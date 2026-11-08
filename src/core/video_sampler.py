"""Video Sampler: Decodes video, monitors UI-anchor stability to filter out 3D moving backgrounds,
and extracts sharp stationary frames for each player dwell period.
"""
from typing import List, Tuple, Optional, Generator
import cv2
import numpy as np

from src.core.roi_manager import ROIManager


class StableFrame:
    def __init__(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, sharpness: float):
        self.frame = frame
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec
        self.sharpness = sharpness


class VideoSampler:
    def __init__(self, sample_fps: float = 15.0, stability_threshold: float = 1.4, min_dwell_frames: int = 2):
        self.sample_fps = sample_fps
        self.stability_threshold = stability_threshold
        self.min_dwell_frames = min_dwell_frames

    @staticmethod
    def calculate_sharpness(img_bgr: np.ndarray) -> float:
        """Calculate Laplacian variance as an image focus / sharpness score."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def get_ui_anchor(frame: np.ndarray) -> np.ndarray:
        """Extract a combined grayscale image of UI regions that are static during a player view,
        completely ignoring the 3D moving video background.
        """
        h, w = frame.shape[:2]
        # Crop Rating circle, Player card header, and Player full name
        c1 = ROIManager.crop_roi(frame, ROIManager.STABILITY_ANCHOR_RATING)
        c2 = ROIManager.crop_roi(frame, ROIManager.STABILITY_ANCHOR_STAT_HEADER)
        c3 = ROIManager.crop_roi(frame, ROIManager.PLAYER_FULL_NAME)

        g1 = cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY) if c1.size > 0 else np.zeros((10, 10), dtype=np.uint8)
        g2 = cv2.cvtColor(c2, cv2.COLOR_BGR2GRAY) if c2.size > 0 else np.zeros((10, 10), dtype=np.uint8)
        g3 = cv2.cvtColor(c3, cv2.COLOR_BGR2GRAY) if c3.size > 0 else np.zeros((10, 10), dtype=np.uint8)

        # Standardize sizes
        g1_res = cv2.resize(g1, (80, 80))
        g2_res = cv2.resize(g2, (80, 40))
        g3_res = cv2.resize(g3, (80, 40))
        return np.vstack([g1_res, g2_res, g3_res])

    @staticmethod
    def _detect_16_9_offset(cap: cv2.VideoCapture, raw_w: int, raw_h: int) -> int:
        """Detect the horizontal offset x0 of the 16:9 UI container for ultrawide or windowed recordings."""
        target_ar = 16.0 / 9.0
        w_16_9 = int(round(raw_h * target_ar))
        fallback_x0 = max(0, (raw_w - w_16_9) // 2)

        try:
            from src.ocr.ocr_engine import OCREngine
            import re
            ocr = OCREngine.get_instance()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            total_sec = count / fps if fps > 0 else 0.0

            for sec in [5.0, 10.0, 15.0, 20.0, 2.0]:
                if total_sec > 0 and sec >= total_sec:
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                hdr = frame[:int(raw_h * 0.15), :]
                boxes = ocr.recognize_raw_boxes(hdr)
                for b in boxes:
                    txt = b[1].lower().replace(" ", "")
                    if any(k in txt for k in ["playerperformance", "matchfacts", "matchoverview", "teammanagement", "postmatch"]):
                        xmin = min(p[0] for p in b[0])
                        cand_x0 = int(round(xmin - 0.032 * w_16_9))
                        cand_x0 = max(0, min(cand_x0, raw_w - w_16_9))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        return cand_x0
                    m = re.search(r'\d+\s*[-–—]\s*\d+', b[1])
                    if m and not re.search(r'\d{2}:\d{2}', b[1]):
                        x_center = (min(p[0] for p in b[0]) + max(p[0] for p in b[0])) / 2.0
                        cand_x0 = int(round(x_center - 0.50 * w_16_9))
                        cand_x0 = max(0, min(cand_x0, raw_w - w_16_9))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        return cand_x0
        except Exception:
            pass

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return fallback_x0

    def sample_stable_frames(self, video_path: str) -> List[StableFrame]:
        """Process video sequentially and yield one sharpest frame per stationary dwell period."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(video_fps / self.sample_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x_offset = None
        if raw_h > 0 and (raw_w / float(raw_h)) > (16.0 / 9.0) + 0.01:
            x_offset = self._detect_16_9_offset(cap, raw_w, raw_h)

        stable_clusters: List[List[Tuple[int, float, np.ndarray, float]]] = []
        current_cluster: List[Tuple[int, float, np.ndarray, float]] = []

        prev_anchor: Optional[np.ndarray] = None
        current_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Process only sampled frames
            if current_frame_idx % frame_interval == 0:
                frame = ROIManager.normalize_to_canonical_16_9(frame, x_offset=x_offset)

                timestamp_sec = current_frame_idx / video_fps
                curr_anchor = self.get_ui_anchor(frame)
                sharpness = self.calculate_sharpness(frame)

                if prev_anchor is not None:
                    diff = float(cv2.absdiff(prev_anchor, curr_anchor).mean())
                    if diff <= self.stability_threshold:
                        current_cluster.append((current_frame_idx, timestamp_sec, frame, sharpness))
                    else:
                        if len(current_cluster) >= self.min_dwell_frames:
                            stable_clusters.append(current_cluster)
                        current_cluster = [(current_frame_idx, timestamp_sec, frame, sharpness)]
                else:
                    current_cluster.append((current_frame_idx, timestamp_sec, frame, sharpness))

                prev_anchor = curr_anchor

            current_frame_idx += 1

        if len(current_cluster) >= self.min_dwell_frames:
            stable_clusters.append(current_cluster)

        cap.release()

        # For each stable dwell period, pick the interior frame with maximum sharpness
        best_frames: List[StableFrame] = []
        for cluster in stable_clusters:
            # Avoid the first and last boundary frames if cluster is large enough
            pool = cluster[1:-1] if len(cluster) >= 3 else cluster
            best = max(pool, key=lambda item: item[3])
            best_frames.append(StableFrame(
                frame=best[2],
                frame_idx=best[0],
                timestamp_sec=best[1],
                sharpness=best[3]
            ))

        return best_frames

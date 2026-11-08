"""ROI Manager: Normalized region definitions for 16:9 screens.
All regions are specified as [ymin, xmin, ymax, xmax] in normalized (0.0 to 1.0) coordinates.
"""
from typing import Tuple, Dict, Optional
import cv2
import numpy as np


class ROIManager:
    # Top Bar Header
    HEADER_TITLE = (0.025, 0.02, 0.09, 0.28)          # "Player Performance" or Match Facts
    HEADER_TEAM_RIGHT = (0.035, 0.60, 0.105, 0.88)
    HEADER_SCORE = (0.02, 0.20, 0.12, 0.80)           # Top match banner in Match Facts

    # Player Card (Top Left)
    PLAYER_OVR = (0.16, 0.03, 0.26, 0.095)            # Overall rating
    PLAYER_FULL_NAME = (0.155, 0.12, 0.245, 0.265)
    PLAYER_RATING_CIRCLE = (0.145, 0.26, 0.275, 0.335) # Match rating badge
    PLAYER_PERF_TOTAL_RATING_BANNER = (0.19, 0.36, 0.28, 0.60) # Total Rating & potm badge
    PLAYER_DISCIPLINE_CARDS = (0.245, 0.145, 0.290, 0.220)     # Yellow / Red card icon under player name

    # Squad List (Left Column)
    SQUAD_TABLE = (0.28, 0.02, 0.93, 0.35)            # Full squad list area
    SQUAD_HEADER = (0.28, 0.02, 0.33, 0.35)           # "POS Name RR G AST"
    
    # Navigation Bar
    TAB_NAV = (0.12, 0.36, 0.175, 0.90)               # "Summary Possession Goalkeeping"

    # Outfield Summary Stats (Right Table)
    OUTFIELD_STAT_LABELS = (0.245, 0.66, 0.89, 0.895) # Stat name labels column
    OUTFIELD_PLAYER_VALS = (0.245, 0.895, 0.89, 0.935) # Player stats (under player face icon)
    OUTFIELD_TEAM_VALS = (0.245, 0.935, 0.89, 0.975)   # Team stats (under crest icon)

    # Goalkeeping Tab Stats (Right Table: "Overall Saving")
    GK_SAVING_TITLE = (0.21, 0.74, 0.26, 0.88)        # "Overall Saving"
    GK_SAVING_LABELS = (0.27, 0.66, 0.68, 0.90)       # Shots Against, Saves, Goals Conceded, etc.
    GK_SAVING_VALS = (0.27, 0.91, 0.68, 0.96)         # GK specific values

    # Match Facts Summary Center Rows
    MATCH_FACTS_ROWS = (0.18, 0.30, 0.90, 0.70)       # Center stat comparison table

    # UI Stability Anchors (high-contrast regions used to detect player switching)
    # These regions ignore the moving background video
    STABILITY_ANCHOR_HEADER = (0.03, 0.03, 0.08, 0.25)
    STABILITY_ANCHOR_RATING = (0.155, 0.265, 0.265, 0.33)
    STABILITY_ANCHOR_SQUAD_HEADER = (0.285, 0.025, 0.325, 0.34)
    STABILITY_ANCHOR_STAT_HEADER = (0.185, 0.88, 0.245, 0.96) # Avatar & badge icons area

    @staticmethod
    def normalize_to_canonical_16_9(
        frame: np.ndarray,
        target_w: int = 1920,
        target_h: int = 1080,
        x_offset: Optional[int] = None,
    ) -> np.ndarray:
        """Crop pillarboxes / letterboxes from non-16:9 recordings and scale to standard 16:9
        FC UI renders in a 16:9 container, which may be centered or offset on custom/windowed recordings.
        """
        if frame is None or frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        ar = w / float(h)
        target_ar = 16.0 / 9.0

        if ar > target_ar + 0.01:
            w_16_9 = int(round(h * target_ar))
            if x_offset is not None:
                x0 = max(0, min(x_offset, w - w_16_9))
            else:
                x0 = (w - w_16_9) // 2
            frame = frame[:, x0 : x0 + w_16_9]
        elif ar < target_ar - 0.01:
            # Letterboxed (e.g. 16:10, 4:3)
            h_16_9 = int(round(w / target_ar))
            y0 = (h - h_16_9) // 2
            frame = frame[y0 : y0 + h_16_9, :]

        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        return frame

    @staticmethod
    def crop_roi(frame: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
        """Crop normalized ROI [ymin, xmin, ymax, xmax] from a frame."""
        h, w = frame.shape[:2]
        ymin, xmin, ymax, xmax = roi
        y1 = max(0, int(round(ymin * h)))
        y2 = min(h, int(round(ymax * h)))
        x1 = max(0, int(round(xmin * w)))
        x2 = min(w, int(round(xmax * w)))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def get_pixel_coords(frame_shape: Tuple[int, int], roi: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """Convert normalized ROI to (x1, y1, x2, y2)."""
        h, w = frame_shape[:2]
        ymin, xmin, ymax, xmax = roi
        return (
            max(0, int(round(xmin * w))),
            max(0, int(round(ymin * h))),
            min(w, int(round(xmax * w))),
            min(h, int(round(ymax * h))),
        )

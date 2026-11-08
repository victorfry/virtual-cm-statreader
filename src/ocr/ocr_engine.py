import re
import unicodedata
from typing import Optional, List, Tuple
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR


# Mapping of visually similar OCR misrecognitions to correct digits
DIGIT_MAP = {
    'O': '0', 'o': '0', 'D': '0', 'Q': '0', 'a': '0',
    'l': '1', 'I': '1', '|': '1', 'i': '1',
    'z': '2', 'Z': '2',
    's': '5', 'S': '5',
    'b': '6',
    'B': '8',
}


class OCREngine:
    _instance: Optional['OCREngine'] = None

    def __init__(self):
        self.ocr = RapidOCR()

    @classmethod
    def get_instance(cls) -> 'OCREngine':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _pad_aspect_ratio(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return img_bgr
        aspect = w / float(h)
        if aspect > 3.0:
            target_h = int(w / 2.5)
            pad = (target_h - h) // 2
            if pad > 0:
                img_bgr = cv2.copyMakeBorder(img_bgr, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif aspect < 0.33:
            target_w = int(h * 0.4)
            pad = (target_w - w) // 2
            if pad > 0:
                img_bgr = cv2.copyMakeBorder(img_bgr, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_bgr

    def recognize_text(self, img_bgr: np.ndarray) -> str:
        if img_bgr is None or img_bgr.size == 0:
            return ""
        padded = self._pad_aspect_ratio(img_bgr)
        results, _ = self.ocr(padded)
        if not results:
            return ""
        return " ".join(r[1] for r in results).strip()

    def recognize_raw_boxes(self, img_bgr: np.ndarray) -> List[Tuple[list, str, float]]:
        if img_bgr is None or img_bgr.size == 0:
            return []
        padded = self._pad_aspect_ratio(img_bgr)
        results, _ = self.ocr(padded)
        return [(r[0], r[1], float(r[2])) for r in results] if results else []

    def recognize_line(self, img_bgr: np.ndarray) -> str:
        if img_bgr is None or img_bgr.size == 0:
            return ""
        h, w = img_bgr.shape[:2]
        if h < 2 or w < 2:
            return ""
        target_w = max(16, int(w * 48.0 / h))
        resized = cv2.resize(img_bgr, (target_w, 48))
        results, _ = self.ocr.text_recognizer([resized])
        if results and results[0][0]:
            return results[0][0].strip()
        return ""

    def _recognize_tight_crop(self, cell_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]], is_float: bool = False) -> str:
        x_min = max(0, min(b[0] for b in boxes) - 2)
        x_max = min(cell_bgr.shape[1], max(b[0] + b[2] for b in boxes) + 2)
        y_min = max(0, min(b[1] for b in boxes) - 1)
        y_max = min(cell_bgr.shape[0], max(b[1] + b[3] for b in boxes) + 1)

        tight_crop = cell_bgr[y_min:y_max, x_min:x_max]
        total_w = max(b[0] + b[2] for b in boxes) - min(b[0] for b in boxes)

        candidates = []
        for pad_x in [24, 16, 12, 8]:
            padded = cv2.copyMakeBorder(tight_crop, 4, 4, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            target_w = max(16, int(padded.shape[1] * 48.0 / padded.shape[0]))
            resized = cv2.resize(padded, (target_w, 48))
            results, _ = self.ocr.text_recognizer([resized])
            if results and results[0][0]:
                txt = results[0][0]
                conf = float(results[0][1])
                clean = self.clean_digits(txt, is_float=is_float)
                if clean and clean != "0":
                    candidates.append((clean, conf, len(clean)))

        if not candidates:
            if len(boxes) == 1 and boxes[0][2] >= 9:
                return "0"
            return ""

        if total_w >= 18 and not is_float:
            multi = [c for c in candidates if c[2] > 1]
            if multi:
                best = max(multi, key=lambda c: c[1])
                return best[0]

        best = max(candidates, key=lambda c: c[1])
        if len(boxes) == 1 and not is_float and len(best[0]) > 1 and total_w < 18:
            return best[0][0]
        return best[0]

    def recognize_single_cell(self, cell_bgr: np.ndarray, is_float: bool = False, min_brightness: int = 120) -> str:
        if cell_bgr is None or cell_bgr.size == 0:
            return "0"

        if len(cell_bgr.shape) == 3 and cell_bgr.shape[2] == 3 and (float(cell_bgr[:, :, 0].mean()) > float(cell_bgr[:, :, 2].mean()) + 15):
            gray = cell_bgr[:, :, 2]
        else:
            gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY) if len(cell_bgr.shape) == 3 else cell_bgr

        if gray.max() < min_brightness:
            return "0"

        dyn_th = int(np.clip(gray.max() * 0.48, 80, 140))
        thresh = cv2.threshold(gray, dyn_th, 255, cv2.THRESH_BINARY)[1]

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = [cv2.boundingRect(c) for c in cnts if cv2.boundingRect(c)[3] >= 8]
        if not boxes:
            return "0"

        w_cell = cell_bgr.shape[1]
        max_h = max(b[3] for b in boxes)
        if max_h >= 14:
            min_char_h = max(12, int(max_h * 0.65)) if not is_float else int(max_h * 0.35)
            filtered = [
                b for b in boxes
                if not ((b[0] <= 2 and b[2] <= 15) or (b[0] + b[2] >= w_cell - 2 and b[2] <= 15))
                and (b[3] >= min_char_h)
            ]
            if filtered:
                boxes = filtered
        boxes.sort(key=lambda b: b[0])

        if not is_float:
            split_boxes = []
            for (bx, by, bw, bh) in boxes:
                if bw >= 17 and bh >= 12:
                    crop = (thresh[by:by+bh, bx:bx+bw] > 0).astype(int)
                    col_sums = crop.sum(axis=0)
                    m_start = max(4, int(bw * 0.25))
                    m_end = min(bw - 4, int(bw * 0.75) + 1)
                    if m_start < m_end:
                        min_idx = m_start + int(np.argmin(col_sums[m_start:m_end]))
                        min_val = col_sums[min_idx]
                        left_peak = col_sums[:min_idx].max() if min_idx > 0 else 0
                        right_peak = col_sums[min_idx:].max() if min_idx < len(col_sums) else 0
                        max_bridge = max(7, int(bh * 0.40))
                        if min_val <= max_bridge and min_val <= min(left_peak, right_peak) * 0.65:
                            b1 = (bx, by, min_idx, bh)
                            b2 = (bx + min_idx, by, bw - min_idx, bh)
                            split_boxes.extend([b1, b2])
                            continue
                split_boxes.append((bx, by, bw, bh))
            boxes = split_boxes

        if not boxes:
            return "0"

        _direct_cache = None
        def get_direct() -> str:
            nonlocal _direct_cache
            if _direct_cache is None:
                direct_raw = self.recognize_text(cell_bgr)
                _direct_cache = self.clean_digits(direct_raw, is_float=is_float)
            return _direct_cache

        if not is_float and get_direct() == "100":
            return "100"

        strokes = [(bw <= 8 and bh >= 11) for (bx, by, bw, bh) in boxes]

        if len(boxes) == 1:
            if strokes[0]:
                return "1"
            tight = self._recognize_tight_crop(cell_bgr, boxes, is_float=is_float)
            if tight:
                return tight
            direct = get_direct()
            if direct and len(direct) == 1:
                return direct
            return "0"

        if len(boxes) == 2 and not is_float:
            if strokes[0] and strokes[1]:
                return "11"
            if strokes[0] and not strokes[1]:
                direct = get_direct()
                if direct == "100":
                    return "100"
                if direct and len(direct) == 2 and direct.startswith('1'):
                    return direct
                sc = self._recognize_tight_crop(cell_bgr, [boxes[1]], is_float=False)
                digit_part = sc[-1] if sc and sc[-1].isdigit() else (direct[-1] if direct and direct[-1].isdigit() else "0")
                return f"1{digit_part}"
            if not strokes[0] and strokes[1]:
                direct = get_direct()
                if direct and len(direct) == 2 and direct.endswith('1'):
                    return direct
                sc = self._recognize_tight_crop(cell_bgr, [boxes[0]], is_float=False)
                digit_part = sc[0] if sc and sc[0].isdigit() else (direct[0] if direct and direct[0].isdigit() else "0")
                return f"{digit_part}1"
            d1 = self._recognize_tight_crop(cell_bgr, [boxes[0]], is_float=False)
            d2 = self._recognize_tight_crop(cell_bgr, [boxes[1]], is_float=False)
            if d1 and d2 and d1.isdigit() and d2.isdigit():
                return f"{d1[-1]}{d2[-1]}"

        if len(boxes) >= 3 and not is_float:
            if get_direct() == "100":
                return "100"

        tight = self._recognize_tight_crop(cell_bgr, boxes, is_float=is_float)
        if tight and len(tight) == len(boxes):
            return tight

        direct = get_direct()
        if direct and len(direct) == len(boxes):
            return direct

        return tight if tight else (direct if direct else "0")

    @staticmethod
    def clean_digits(text: str, is_float: bool = False) -> str:
        """Normalize unicode (including fullwidth digits) and map digit confusions."""
        norm = unicodedata.normalize('NFKC', text)
        for k, v in DIGIT_MAP.items():
            norm = norm.replace(k, v)

        if is_float:
            norm = re.sub(r'[^0-9.]', '', norm)
            try:
                val = float(norm)
                if val > 40.0:
                    norm = f"{val / 10.0:.1f}"
            except ValueError:
                pass
        else:
            norm = re.sub(r'[^0-9]', '', norm)

        return norm if norm else "0"

from enum import Enum
from typing import Tuple
import re
import numpy as np
import cv2

from src.core.roi_manager import ROIManager
from src.ocr.ocr_engine import OCREngine


class ScreenType(Enum):
    MATCH_FACTS_SUMMARY = "MATCH_FACTS_SUMMARY"
    PLAYER_PERF_SUMMARY = "PLAYER_PERF_SUMMARY"
    PLAYER_PERF_GOALKEEPING = "PLAYER_PERF_GOALKEEPING"
    OTHER = "OTHER"


class ScreenClassifier:
    def __init__(self):
        self.ocr = OCREngine.get_instance()

    def classify_frame(self, frame: np.ndarray) -> ScreenType:
        """Classify screen type using targeted anchor regions."""
        if frame is None or frame.size == 0:
            return ScreenType.OTHER

        frame = ROIManager.normalize_to_canonical_16_9(frame)
        h, w = frame.shape[:2]

        hdr_crop = ROIManager.crop_roi(frame, ROIManager.HEADER_TITLE)
        hdr_text = self.ocr.recognize_line(hdr_crop).lower()
        if not hdr_text:
            hdr_text = self.ocr.recognize_text(hdr_crop).lower()

        if "perf" in hdr_text or "pla" in hdr_text:
            table_top_crop = frame[int(h * 0.20):int(h * 0.35), int(w * 0.65):int(w * 0.95)]
            table_top_text = self.ocr.recognize_line(table_top_crop).lower()
            if not table_top_text:
                table_top_text = self.ocr.recognize_text(table_top_crop).lower()

            if "saving" in table_top_text or "shots against" in table_top_text:
                return ScreenType.PLAYER_PERF_GOALKEEPING
            else:
                return ScreenType.PLAYER_PERF_SUMMARY

        tab_crop = frame[int(h * 0.12):int(h * 0.20), int(w * 0.20):int(w * 0.80)]
        tab_text = self.ocr.recognize_line(tab_crop).lower()
        if not any(term in tab_text for term in ["summary", "possession", "shooting", "passing", "defending"]):
            tab_text = self.ocr.recognize_text(tab_crop).lower()

        if any(term in tab_text for term in ["summary", "possession", "shooting", "passing", "defending"]):
            top_banner_crop = frame[int(h * 0.02):int(h * 0.12), int(w * 0.20):int(w * 0.80)]
            top_banner_text = self.ocr.recognize_text(top_banner_crop)
            if re.search(r'\d+\s*[:\-]\s*\d+', top_banner_text) or ":" in top_banner_text or "-" in top_banner_text:
                return ScreenType.MATCH_FACTS_SUMMARY

        hub_left_crop = frame[int(h * 0.15):int(h * 0.45), int(w * 0.05):int(w * 0.25)]
        hub_left_text = self.ocr.recognize_text(hub_left_crop).lower().replace(" ", "")
        if any(term in hub_left_text for term in ["matchfacts", "performance", "highlighter"]):
            top_banner_crop = frame[int(h * 0.02):int(h * 0.15), :]
            top_banner_text = self.ocr.recognize_text(top_banner_crop)
            if re.search(r'\d+\s*[:\-]\s*\d+', top_banner_text) or "-" in top_banner_text:
                return ScreenType.MATCH_FACTS_SUMMARY

        return ScreenType.OTHER

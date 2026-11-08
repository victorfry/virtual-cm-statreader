"""Icon Detector: Detects Man of the Match (MOTM) yellow ball icon and cards."""
from typing import Tuple, Optional
import numpy as np
import cv2


class IconDetector:
    # Yellow/Gold color range in HSV for MOTM ball
    LOWER_YELLOW = np.array([16, 80, 100])
    UPPER_YELLOW = np.array([36, 255, 255])

    # Discipline card color ranges in HSV
    LOWER_CARD_YELLOW = np.array([18, 110, 100])
    UPPER_CARD_YELLOW = np.array([32, 255, 255])

    LOWER_CARD_RED1 = np.array([0, 110, 100])
    UPPER_CARD_RED1 = np.array([10, 255, 255])
    LOWER_CARD_RED2 = np.array([170, 110, 100])
    UPPER_CARD_RED2 = np.array([180, 255, 255])

    @classmethod
    def detect_discipline_cards(cls, card_discipline_roi: np.ndarray) -> Tuple[int, int]:
        """Detect yellow and red card icons in the player modal under the player name.
        Returns: (yellow_cards, red_cards) where yellow is 0 or 1, and red is 0 or 1.
        """
        if card_discipline_roi is None or card_discipline_roi.size == 0:
            return 0, 0

        hsv = cv2.cvtColor(card_discipline_roi, cv2.COLOR_BGR2HSV)

        # 1. Check for Red Card
        mask_red = cv2.inRange(hsv, cls.LOWER_CARD_RED1, cls.UPPER_CARD_RED1) | cv2.inRange(hsv, cls.LOWER_CARD_RED2, cls.UPPER_CARD_RED2)
        cnts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_red:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspect = w / float(h) if h > 0 else 0.0
            if area >= 35 and 0.45 <= aspect <= 1.1 and 7 <= w <= 25 and 10 <= h <= 28:
                return 0, 1

        # 2. Check for Yellow Card
        mask_yellow = cv2.inRange(hsv, cls.LOWER_CARD_YELLOW, cls.UPPER_CARD_YELLOW)
        cnts_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_yellow:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            aspect = w / float(h) if h > 0 else 0.0
            if area >= 35 and 0.45 <= aspect <= 1.1 and 7 <= w <= 25 and 10 <= h <= 28:
                return 1, 0

        return 0, 0

    @classmethod
    def check_motm_icon(cls, squad_row_left_patch: np.ndarray) -> bool:
        """Check if the region immediately to the left of the player's name contains the MOTM yellow ball icon."""
        if squad_row_left_patch is None or squad_row_left_patch.size == 0:
            return False

        hsv = cv2.cvtColor(squad_row_left_patch, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, cls.LOWER_YELLOW, cls.UPPER_YELLOW)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h) if h > 0 else 0.0
            if area >= 100 and 0.65 <= aspect <= 1.5 and 12 <= w <= 32 and 12 <= h <= 32:
                return True
        return False

    @classmethod
    def scan_squad_for_motm(cls, squad_crop: np.ndarray) -> Optional[int]:
        """Scan the squad list icon column (x: 50 to 135) to find the Y-coordinate of the circular MOTM yellow ball."""
        if squad_crop is None or squad_crop.size == 0:
            return None

        icon_col = squad_crop[:, 50:135] if squad_crop.shape[1] > 135 else squad_crop

        hsv = cv2.cvtColor(icon_col, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, cls.LOWER_YELLOW, cls.UPPER_YELLOW)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = cw / float(ch) if ch > 0 else 0.0
            if area >= 120 and 0.65 <= aspect <= 1.5 and 12 <= cw <= 32 and 12 <= ch <= 32:
                return y + ch // 2
        return None

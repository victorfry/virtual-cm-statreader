from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
import re
from rapidfuzz import fuzz

from src.core.roi_manager import ROIManager
from src.ocr.ocr_engine import OCREngine
from src.ocr.icon_detector import IconDetector


class StatExtractor:
    OUTFIELD_STAT_KEYS = [
        "goals", "assists", "shots", "shot_accuracy", "passes", "pass_accuracy",
        "dribbles", "dribble_success_rate", "tackles", "tackle_success_rate", "offsides",
        "fouls_committed", "possession_won", "possession_lost", "minutes_played",
        "distance_covered_km", "distance_sprinted_km"
    ]

    GK_STAT_KEYS = [
        "shots_against", "shots_on_target", "saves", "goals_conceded", "save_success_rate",
        "punch_saves", "rush_saves", "penalty_saves", "penalty_goals_conceded",
        "shootout_saves", "shootout_goals_conceded", "punch_clearances", "cross_claims"
    ]

    POSITIONS = [
        "GK", "RB", "RCB", "CB", "LCB", "LB", "RWB", "LWB",
        "RDM", "CDM", "LDM", "RM", "RCM", "CM", "LCM", "LM",
        "RAM", "CAM", "LAM", "RW", "RS", "ST", "LS", "LW", "SUB"
    ]

    def __init__(self):
        self.ocr = OCREngine.get_instance()

    def extract_player_card(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract OVR, Full Name, and Match Rating from the top-left player card."""
        h, w = frame.shape[:2]

        ovr = None
        ovr_crop = ROIManager.crop_roi(frame, ROIManager.PLAYER_OVR)
        raw_ovr = self.ocr.recognize_single_cell(ovr_crop)
        if raw_ovr.isdigit() and len(raw_ovr) >= 2:
            cand = int(raw_ovr[:2])
            if 40 <= cand <= 99:
                ovr = cand

        card_roi = frame[int(h * 0.14):int(h * 0.28), int(w * 0.02):int(w * 0.35)]
        boxes = self.ocr.recognize_raw_boxes(card_roi)

        match_rating = None
        name_parts = []

        for b, txt, score in boxes:
            txt_clean = txt.strip()
            if re.match(r'^\d+\.\d+$', txt_clean):
                try:
                    r_val = float(txt_clean)
                    if 1.0 <= r_val <= 10.0:
                        match_rating = r_val
                except ValueError:
                    pass
            elif ovr is None and txt_clean.isdigit() and 40 <= int(txt_clean) <= 99:
                ovr = int(txt_clean)
            elif not txt_clean.isdigit() and len(txt_clean) > 1 and not re.match(r'^\d+\.\d+$', txt_clean):
                if txt_clean.upper() not in {"QUIT", "RESUME", "RESTART", "CANCEL", "CONFIRM", "ADVANCE", "BACK", "EXIT", "OPTIONS", "SETTINGS", "REPLAY", "CONTINUE", "MATCHFACTS", "PERFORMANCE"}:
                    name_parts.append(txt_clean)

        if ovr is None:
            ovr_patch = card_roi[:, :int(card_roi.shape[1] * 0.25)]
            raw_ovr = self.ocr.clean_digits(self.ocr.recognize_text(ovr_patch))
            if raw_ovr.isdigit() and 40 <= int(raw_ovr) <= 99:
                ovr = int(raw_ovr)
            else:
                ovr = 80

        if match_rating is None:
            rating_crop = ROIManager.crop_roi(frame, ROIManager.PLAYER_RATING_CIRCLE)
            raw_rating = self.ocr.recognize_text(rating_crop)
            clean_rating = self.ocr.clean_digits(raw_rating, is_float=True)
            try:
                r_val = float(clean_rating)
                if r_val > 10.0:
                    r_val = round(r_val / 10.0, 1)
                match_rating = r_val if 1.0 <= r_val <= 10.0 else 6.0
            except ValueError:
                match_rating = 6.0

        if not name_parts:
            name_crop = ROIManager.crop_roi(frame, ROIManager.PLAYER_FULL_NAME)
            raw_boxes = self.ocr.recognize_raw_boxes(name_crop)
            raw_boxes.sort(key=lambda b: min(p[1] for p in b[0]))
            name_parts = [
                b[1].strip() for b in raw_boxes 
                if b[1].strip() and b[1].strip().upper() not in {"QUIT", "RESUME", "RESTART", "CANCEL", "CONFIRM", "ADVANCE", "BACK", "EXIT", "OPTIONS", "SETTINGS", "REPLAY", "CONTINUE", "MATCHFACTS", "PERFORMANCE"}
            ]

        full_name = " ".join(name_parts) if name_parts else "Unknown Player"

        disc_crop = ROIManager.crop_roi(frame, ROIManager.PLAYER_DISCIPLINE_CARDS)
        y_cards, r_cards = IconDetector.detect_discipline_cards(disc_crop)

        return {
            "overall_rating": ovr,
            "match_rating": match_rating,
            "full_name": full_name,
            "yellow_cards": y_cards,
            "red_cards": r_cards
        }

    def extract_team_name(self, frame: np.ndarray) -> str:
        """Extract team name from the top-right header."""
        team_crop = ROIManager.crop_roi(frame, ROIManager.HEADER_TEAM_RIGHT)
        raw_team = self.ocr.recognize_text(team_crop)
        cleaned = re.sub(r'\b(LT|RT|LB|RB)\b', '', raw_team, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'[^a-zA-Z\s]', '', cleaned).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)

        if len(cleaned) == 2:
            return cleaned.upper()
        return cleaned.title() if len(cleaned) > 2 else "Unknown Team"

    def extract_squad_info(self, frame: np.ndarray, player_full_name: str) -> Dict[str, Any]:
        squad_crop = ROIManager.crop_roi(frame, ROIManager.SQUAD_TABLE)
        h_squad, w_squad = squad_crop.shape[:2]
        boxes = self.ocr.recognize_raw_boxes(squad_crop)

        pos_candidates = []
        name_candidates = []

        for box, txt, score in boxes:
            txt_clean = txt.strip()
            ymin = min(p[1] for p in box)
            xmin = min(p[0] for p in box)
            xmax = max(p[0] for p in box)

            if xmin < w_squad * 0.18:
                for p in self.POSITIONS:
                    if txt_clean == p or txt_clean.startswith(p) or txt_clean.endswith(p):
                        pos_candidates.append((ymin, p, box))
                        break

            elif w_squad * 0.15 <= xmin <= w_squad * 0.55:
                name_candidates.append((ymin, txt_clean, box))

        hsv = cv2.cvtColor(squad_crop, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(hsv, np.array([75, 45, 60]), np.array([108, 255, 255]))
        row_counts = np.sum(cyan_mask > 0, axis=1)
        cyan_y = np.argmax(row_counts) if np.max(row_counts) > 15 else None

        motm_y = IconDetector.scan_squad_for_motm(squad_crop)

        matched_short_name = ""
        matched_pos = "CM"  # Default fallback
        matched_is_motm = False

        best_score = -1.0
        clean_full = re.sub(r'[^a-zA-Z\s]', '', player_full_name).lower()
        full_words = [w for w in clean_full.split() if len(w) >= 2]

        for ymin, name_txt, box in name_candidates:
            clean_name = re.sub(r'[^a-zA-Z\s]', '', name_txt).lower()
            cand_words = [w for w in clean_name.split() if len(w) >= 2]
            s_part = fuzz.partial_ratio(clean_name, clean_full)
            s_token = fuzz.token_set_ratio(clean_name, clean_full)
            score = max(s_part, s_token)

            surname_match = False
            if cand_words and full_words:
                cand_surname = cand_words[-1]
                surname_match = any(fuzz.ratio(cand_surname, w) >= 65 for w in full_words)
            elif not cand_words:
                surname_match = False

            if not surname_match and full_words:
                score -= 35.0

            if cyan_y is not None and abs(ymin - cyan_y) < 30 and surname_match:
                score += 35.0

            if score > best_score and score >= 40.0:
                best_score = score
                matched_short_name = name_txt
                
                closest_p = [p for p in pos_candidates if abs(p[0] - ymin) < 35]
                if closest_p:
                    matched_pos = min(closest_p, key=lambda p: abs(p[0] - ymin))[1]

                if motm_y is not None and abs(ymin - motm_y) < 28:
                    matched_is_motm = True

        if not matched_short_name:
            matched_short_name = player_full_name.split()[0] if player_full_name else "Unknown"
            if cyan_y is not None:
                closest_p = [p for p in pos_candidates if abs(p[0] - cyan_y) < 35]
                if closest_p:
                    matched_pos = min(closest_p, key=lambda p: abs(p[0] - cyan_y))[1]

        motm_banner_crop = ROIManager.crop_roi(frame, ROIManager.PLAYER_PERF_TOTAL_RATING_BANNER)
        motm_banner_txt = self.ocr.recognize_text(motm_banner_crop).lower().replace(" ", "")
        if "playerofthematch" in motm_banner_txt:
            matched_is_motm = True

        return {
            "position": matched_pos,
            "short_name": matched_short_name,
            "man_of_the_match": matched_is_motm,
            "squad_boxes": boxes,
        }

    def extract_outfield_stats(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract all 17 outfield stats from the right-hand stats table."""
        h, w = frame.shape[:2]
        table_crop = frame[int(h * 0.245):int(h * 0.89), int(w * 0.66):int(w * 0.96)]
        h_table, w_table = table_crop.shape[:2]

        p_x1, p_x2 = int(w_table * 0.775), int(w_table * 0.865)
        t_x1, t_x2 = int(w_table * 0.875), int(w_table * 0.965)

        scale = h_table / 697.0
        y_start = 42.5 * scale
        step = 39.0 * scale
        half_h = int(step * 0.32)

        stats: Dict[str, Any] = {}

        for i, key in enumerate(self.OUTFIELD_STAT_KEYS):
            is_float = ("distance" in key)
            cy = int(round(y_start + i * step))
            y1, y2 = max(0, cy - half_h), min(h_table, cy + half_h)

            p_cell = table_crop[y1:y2, p_x1:p_x2]
            raw_val = self.ocr.recognize_single_cell(p_cell, is_float=is_float)

            if is_float:
                try:
                    stats[key] = float(raw_val)
                except ValueError:
                    stats[key] = 0.0
            else:
                try:
                    stats[key] = int(raw_val)
                except ValueError:
                    stats[key] = 0

        return stats

    def extract_team_goals(self, frame: np.ndarray) -> Optional[int]:
        h, w = frame.shape[:2]
        table_crop = frame[int(h * 0.245):int(h * 0.89), int(w * 0.66):int(w * 0.96)]
        h_table, w_table = table_crop.shape[:2]
        t_x1, t_x2 = int(w_table * 0.875), int(w_table * 0.965)
        scale = h_table / 697.0
        y_start = 42.5 * scale
        step = 39.0 * scale
        half_h = int(step * 0.32)
        y1, y2 = max(0, int(y_start - half_h)), min(h_table, int(y_start + half_h))
        t_cell = table_crop[y1:y2, t_x1:t_x2]
        raw_val = self.ocr.recognize_single_cell(t_cell, is_float=False)
        try:
            return int(raw_val)
        except ValueError:
            return None

    def extract_goalkeeper_stats(self, frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        table_crop = frame[int(h * 0.27):int(h * 0.78), int(w * 0.66):int(w * 0.96)]
        h_table, w_table = table_crop.shape[:2]

        val_x1, val_x2 = int(w_table * 0.875), int(w_table * 0.950)

        y_start = h_table * 0.100
        step = (h_table * 0.840) / 12.0
        half_h = int(step * 0.38)

        gk_stats: Dict[str, Any] = {}

        for i, key in enumerate(self.GK_STAT_KEYS):
            cy = int(round(y_start + i * step))
            y1, y2 = max(0, cy - half_h), min(h_table, cy + half_h)

            cell = table_crop[y1:y2, val_x1:val_x2]
            raw_val = self.ocr.recognize_single_cell(cell, is_float=False)

            try:
                gk_stats[key] = int(raw_val)
            except ValueError:
                gk_stats[key] = 0

        return gk_stats

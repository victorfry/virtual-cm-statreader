from typing import Dict, List, Optional, Any
from collections import Counter
import re
import numpy as np

from src.core.video_sampler import StableFrame
from src.core.roi_manager import ROIManager
from src.parser.screen_classifier import ScreenClassifier, ScreenType
from src.ocr.stat_extractor import StatExtractor
from src.ocr.icon_detector import IconDetector
from src.parser.roster_matcher import RosterMatcher
from src.models.match_data import MatchRecord, TeamMatchStats, OutfieldPlayerStats, GoalkeeperStats


class MatchAggregator:
    def __init__(
        self,
        roster_matcher: Optional[RosterMatcher] = None,
        classifier: Optional[ScreenClassifier] = None,
        extractor: Optional[StatExtractor] = None,
    ):
        self.classifier = classifier or ScreenClassifier()
        self.extractor = extractor or StatExtractor()
        self.roster_matcher = roster_matcher or RosterMatcher()

    @staticmethod
    def canonicalize_team(team_str: str, known_teams: List[str]) -> str:
        if not known_teams:
            return team_str
        clean_target = re.sub(r'[^a-zA-Z]', '', team_str).lower()
        if not clean_target:
            return team_str

        best_t = team_str
        best_score = 0
        from rapidfuzz import fuzz
        for kt in known_teams:
            clean_kt = re.sub(r'[^a-zA-Z]', '', kt).lower()
            sc = fuzz.ratio(clean_target, clean_kt)
            if sc > best_score:
                best_score = sc
                best_t = kt

        return best_t if best_score >= 65 else team_str

    def process_frames(
        self,
        stable_frames: List[StableFrame],
        progress_callback: Optional[Any] = None
    ) -> MatchRecord:
        home_team_name: Optional[str] = None
        away_team_name: Optional[str] = None
        home_score = 0
        away_score = 0
        match_time = "90:00"

        team_players: Dict[str, Dict[str, OutfieldPlayerStats]] = {}
        team_gk_saving: Dict[str, Dict[str, Any]] = {}
        team_crest_goals: Dict[str, Counter] = {}
        pending_cards: Dict[str, Dict[str, int]] = {}
        
        motm_votes = Counter()
        last_player_name: Optional[str] = None

        for f_idx, sf in enumerate(stable_frames):
            screen_type = self.classifier.classify_frame(sf.frame)

            if screen_type == ScreenType.MATCH_FACTS_SUMMARY:
                res_hdr = self.extractor.ocr.recognize_raw_boxes(sf.frame[:int(sf.frame.shape[0] * 0.15), :])
                hdr_texts = [r[1] for r in res_hdr]
                combined_hdr = " ".join(hdr_texts)

                score_box = None
                for b in res_hdr:
                    m_time = re.match(r'^(\d{2})[:\uff1a](\d{2})$', b[1].strip())
                    if m_time:
                        mins = int(m_time.group(1))
                        if mins >= 45:
                            match_time = b[1].strip().replace('\uff1a', ':')
                        continue

                    m = re.search(r'(?<!\d)(\d{1,2})\s*[-–—:\uff1a]\s*(\d{1,2})(?!\d)', b[1])
                    if m:
                        g1, g2 = m.group(1), m.group(2)
                        if g1 == '00' or g2 == '00' or (len(g1) == 2 and g1.startswith('0')):
                            continue
                        s1, s2 = int(g1), int(g2)
                        if s1 < 30 and s2 < 30:
                            home_score = s1
                            away_score = s2
                            score_box = b
                            break

                if score_box:
                    score_x_center = (min(p[0] for p in score_box[0]) + max(p[0] for p in score_box[0])) / 2.0

                    def is_game_mode(txt: str) -> bool:
                        return any(gm in txt.lower() for gm in [
                            "friendlies", "online"
                        ])

                    left_boxes = [b for b in res_hdr if max(p[0] for p in b[0]) < score_x_center - 30 and not re.search(r'\d', b[1]) and not is_game_mode(b[1])]
                    right_boxes = [b for b in res_hdr if min(p[0] for p in b[0]) > score_x_center + 30 and not re.search(r'\d', b[1]) and not is_game_mode(b[1])]

                    left_boxes = [b for b in left_boxes if abs(max(p[0] for p in b[0]) - score_x_center) < 550]
                    right_boxes = [b for b in right_boxes if abs(min(p[0] for p in b[0]) - score_x_center) < 550]

                    left_boxes.sort(key=lambda b: min(p[0] for p in b[0]))
                    right_boxes.sort(key=lambda b: min(p[0] for p in b[0]))

                    left_texts = [b[1].strip() for b in left_boxes]
                    right_texts = [b[1].strip() for b in right_boxes]
                    
                    def deduplicate_tokens(tokens: List[str]) -> str:
                        res = []
                        for t in tokens:
                            if not t:
                                continue
                            if any(t.lower() in existing.lower() for existing in res):
                                continue
                            res = [e for e in res if e.lower() not in t.lower()]
                            res.append(t)
                        return " ".join(res).strip().title()

                    if left_texts:
                        home_cand = deduplicate_tokens(left_texts)
                        if len(home_cand) >= 2:
                            upper_c = home_cand.upper().replace(' ', '')
                            if upper_c.startswith('BORUSSIA') and len(upper_c) > 8:
                                home_cand = f"Borussia {upper_c[8:].title()}"
                            elif upper_c.startswith('SL') and len(upper_c) > 3:
                                home_cand = f"SL {upper_c[2:].title()}"
                            elif upper_c.startswith('NOTTM') and len(upper_c) > 5:
                                home_cand = f"Nottm {upper_c[5:].title()}"
                            elif upper_c.endswith('CF') and len(upper_c) > 3:
                                home_cand = f"{upper_c[:-2].title()} CF"
                            elif upper_c.endswith('FC') and len(upper_c) > 3:
                                home_cand = f"{upper_c[:-2].title()} FC"
                            elif len(home_cand) == 2:
                                home_cand = home_cand.upper()
                            home_team_name = home_cand
                    if right_texts:
                        away_cand = deduplicate_tokens(right_texts)
                        if len(away_cand) >= 2:
                            upper_c = away_cand.upper().replace(' ', '')
                            if upper_c.startswith('BORUSSIA') and len(upper_c) > 8:
                                away_cand = f"Borussia {upper_c[8:].title()}"
                            elif upper_c.startswith('SL') and len(upper_c) > 3:
                                away_cand = f"SL {upper_c[2:].title()}"
                            elif upper_c.startswith('NOTTM') and len(upper_c) > 5:
                                away_cand = f"Nottm {upper_c[5:].title()}"
                            elif upper_c.endswith('CF') and len(upper_c) > 3:
                                away_cand = f"{upper_c[:-2].title()} CF"
                            elif upper_c.endswith('FC') and len(upper_c) > 3:
                                away_cand = f"{upper_c[:-2].title()} FC"
                            elif len(away_cand) == 2:
                                away_cand = away_cand.upper()
                            away_team_name = away_cand

                for t in hdr_texts:
                    m = re.match(r'^(\d{2}):(\d{2})$', t.strip())
                    if m and int(m.group(1)) >= 45:
                        match_time = t.strip()

                if progress_callback:
                    progress_callback(f_idx + 1, len(stable_frames), sf.timestamp_sec, f"Match Facts Screen ({home_score}-{away_score})")

            elif screen_type == ScreenType.PLAYER_PERF_SUMMARY:
                raw_team_name = self.extractor.extract_team_name(sf.frame)
                known_teams = [t for t in [home_team_name, away_team_name] if t]
                team_name = self.canonicalize_team(raw_team_name, known_teams)

                card_info = self.extractor.extract_player_card(sf.frame)
                full_name = card_info["full_name"]

                clean_check = re.sub(r'[^a-zA-Z]', '', full_name).upper()
                if not full_name or len(full_name) < 2 or clean_check in {"QUIT", "RESUME", "RESTART", "CANCEL", "CONFIRM", "ADVANCE", "BACK", "EXIT", "UNKNOWNPLAYER"}:
                    if progress_callback:
                        progress_callback(f_idx + 1, len(stable_frames), sf.timestamp_sec, "Skipping menu / exit screen")
                    continue

                clean_full = re.sub(r'[^a-zA-Z]', '', full_name).lower()
                prev_p = team_players.get(team_name, {}).get(full_name)
                if not prev_p and team_name in team_players:
                    for k, v in team_players[team_name].items():
                        if re.sub(r'[^a-zA-Z]', '', k).lower() == clean_full:
                            prev_p = v
                            break

                is_multi_frame_early = False
                if f_idx + 1 < len(stable_frames):
                    next_frame = stable_frames[f_idx + 1].frame
                    next_st = self.classifier.classify_frame(next_frame)
                    if next_st == ScreenType.PLAYER_PERF_SUMMARY:
                        next_h, next_w = next_frame.shape[:2]
                        next_name_crop = next_frame[int(next_h * 0.17):int(next_h * 0.22), int(next_w * 0.09):int(next_w * 0.40)]
                        next_line = self.extractor.ocr.recognize_line(next_name_crop)
                        clean_next = re.sub(r'[^a-zA-Z]', '', next_line).lower()
                        if clean_next and (clean_next in clean_full or clean_full in clean_next):
                            is_multi_frame_early = True

                if is_multi_frame_early:
                    if clean_full not in pending_cards:
                        pending_cards[clean_full] = {"yellow": 0, "red": 0}
                    if card_info.get("yellow_cards", 0) > pending_cards[clean_full]["yellow"]:
                        pending_cards[clean_full]["yellow"] = card_info["yellow_cards"]
                    if card_info.get("red_cards", 0) > pending_cards[clean_full]["red"]:
                        pending_cards[clean_full]["red"] = card_info["red_cards"]
                    if progress_callback:
                        progress_callback(f_idx + 1, len(stable_frames), sf.timestamp_sec, f"[{team_name}] {full_name} - Unclean frame, skipping")
                    continue

                if clean_full in pending_cards:
                    if pending_cards[clean_full]["yellow"] > card_info.get("yellow_cards", 0):
                        card_info["yellow_cards"] = pending_cards[clean_full]["yellow"]
                    if pending_cards[clean_full]["red"] > card_info.get("red_cards", 0):
                        card_info["red_cards"] = pending_cards[clean_full]["red"]

                last_player_name = clean_full

                squad_info = self.extractor.extract_squad_info(sf.frame, full_name)
                stats = self.extractor.extract_outfield_stats(sf.frame)

                crest_g = self.extractor.extract_team_goals(sf.frame)
                if crest_g is not None:
                    if team_name not in team_crest_goals:
                        team_crest_goals[team_name] = Counter()
                    team_crest_goals[team_name][crest_g] += 1

                p_id = None
                p_id_cand, matched_name, conf = self.roster_matcher.match_player(
                    full_name=full_name,
                    short_name=squad_info["short_name"],
                    position=squad_info["position"],
                    player_team=team_name
                )
                if conf >= 70.0:
                    p_id = p_id_cand

                squad_crop = ROIManager.crop_roi(sf.frame, ROIManager.SQUAD_TABLE)
                motm_y = IconDetector.scan_squad_for_motm(squad_crop)
                if motm_y is not None:
                    boxes = squad_info.get("squad_boxes")
                    if not boxes:
                        boxes = self.extractor.ocr.recognize_raw_boxes(squad_crop)
                    w_sq = squad_crop.shape[1]
                    name_boxes = [b for b in boxes if (w_sq * 0.15) <= min(p[0] for p in b[0]) <= (w_sq * 0.55)]
                    closest = [b for b in name_boxes if abs(min(p[1] for p in b[0]) - motm_y) < 25]
                    if closest:
                        motm_votes[closest[0][1].strip()] += 1

                if squad_info.get("man_of_the_match", False):
                    motm_votes[full_name] += 5

                player_obj = OutfieldPlayerStats(
                    full_name=full_name,
                    short_name=squad_info["short_name"],
                    team_name=team_name,
                    position=squad_info["position"],
                    overall_rating=card_info["overall_rating"],
                    match_rating=card_info["match_rating"],
                    man_of_the_match=False,
                    clean_sheet=False,
                    yellow_cards=card_info.get("yellow_cards", 0),
                    red_cards=card_info.get("red_cards", 0),
                    **stats
                )

                if team_name not in team_players:
                    team_players[team_name] = {}
                
                norm_p_key = re.sub(r'[^a-zA-Z]', '', full_name).lower()
                existing_key = None
                for k in team_players[team_name].keys():
                    clean_k = re.sub(r'[^a-zA-Z]', '', k).lower()
                    if clean_k == norm_p_key:
                        existing_key = k
                        break
                    old_p = team_players[team_name][k]
                    if (old_p.overall_rating == player_obj.overall_rating
                            and abs(old_p.match_rating - player_obj.match_rating) < 0.15
                            and old_p.minutes_played == player_obj.minutes_played
                            and (clean_k in norm_p_key or norm_p_key in clean_k)):
                        existing_key = k
                        break

                if not existing_key:
                    team_players[team_name][full_name] = player_obj
                else:
                    old_obj = team_players[team_name][existing_key]
                    longer_name = full_name if len(full_name) >= len(existing_key) else existing_key
                    pos_to_keep = player_obj.position
                    if player_obj.position in ("CM", "CB") and old_obj.position in ("ST", "CF", "LW", "RW", "RS", "LS", "RM", "LM", "RB", "LB", "RWB", "LWB", "CAM", "CDM"):
                        pos_to_keep = old_obj.position
                    player_obj.full_name = longer_name
                    player_obj.position = pos_to_keep
                    player_obj.yellow_cards = max(old_obj.yellow_cards, player_obj.yellow_cards)
                    player_obj.red_cards = max(old_obj.red_cards, player_obj.red_cards)
                    if existing_key != longer_name:
                        del team_players[team_name][existing_key]
                    team_players[team_name][longer_name] = player_obj

                if progress_callback:
                    progress_callback(f_idx + 1, len(stable_frames), sf.timestamp_sec, f"[{team_name}] {full_name} ({squad_info['position']}, OVR {card_info['overall_rating']}) - Extracted")

            elif screen_type == ScreenType.PLAYER_PERF_GOALKEEPING:
                raw_team_name = self.extractor.extract_team_name(sf.frame)
                known_teams = [t for t in [home_team_name, away_team_name] if t]
                team_name = self.canonicalize_team(raw_team_name, known_teams)

                card_info = self.extractor.extract_player_card(sf.frame)
                full_name = card_info["full_name"]

                if not full_name or len(full_name) < 2:
                    continue

                gk_saving = self.extractor.extract_goalkeeper_stats(sf.frame)

                is_gk = False
                squad_info = self.extractor.extract_squad_info(sf.frame, full_name)
                if squad_info.get("position") == "GK":
                    is_gk = True
                elif team_name in team_players:
                    p_obj = team_players[team_name].get(full_name)
                    if p_obj and p_obj.position == "GK":
                        is_gk = True

                prev = team_gk_saving.get(team_name)
                prev_is_gk = prev.get("is_gk", False) if prev else False

                should_update = False
                if prev is None:
                    should_update = True
                elif is_gk and not prev_is_gk:
                    should_update = True
                elif is_gk == prev_is_gk and sf.sharpness > prev["sharpness"]:
                    should_update = True

                if should_update:
                    team_gk_saving[team_name] = {
                        "full_name": full_name,
                        "card_info": card_info,
                        "saving_stats": gk_saving,
                        "sharpness": sf.sharpness,
                        "is_gk": is_gk
                    }

                if progress_callback:
                    progress_callback(f_idx + 1, len(stable_frames), sf.timestamp_sec, f"[{team_name}] {full_name} (Goalkeeping) - Extracted")

            else:
                if progress_callback:
                    progress_callback(f_idx + 1, len(stable_frames), sf.timestamp_sec, "Skipping non-stat screen")

        if len(team_players) > 2:
            from rapidfuzz import fuzz
            keys = list(team_players.keys())
            for i in range(len(keys)):
                k1 = keys[i]
                if k1 not in team_players:
                    continue
                clean_k1 = re.sub(r'[^a-zA-Z]', '', k1).lower()
                for j in range(i + 1, len(keys)):
                    k2 = keys[j]
                    if k2 not in team_players:
                        continue
                    clean_k2 = re.sub(r'[^a-zA-Z]', '', k2).lower()
                    if fuzz.ratio(clean_k1, clean_k2) >= 65:
                        target_key = k1 if " " in k1 else k2
                        merge_key = k2 if target_key == k1 else k1
                        for p_name, p_data in team_players[merge_key].items():
                            p_data.team_name = target_key
                            if p_name not in team_players[target_key] or p_data.minutes_played > 0:
                                team_players[target_key][p_name] = p_data
                        del team_players[merge_key]
                        if merge_key in team_gk_saving and target_key not in team_gk_saving:
                            team_gk_saving[target_key] = team_gk_saving.pop(merge_key)

        for t_name in list(team_players.keys()):
            team_players[t_name] = {
                k: p for k, p in team_players[t_name].items() if p.minutes_played > 0
            }

        identified_teams = list(team_players.keys())
        home_name = home_team_name if home_team_name and home_team_name in team_players else (identified_teams[0] if len(identified_teams) > 0 else "Home Team")
        remaining_teams = [t for t in identified_teams if t != home_name]
        away_name = away_team_name if away_team_name and away_team_name in team_players else (remaining_teams[0] if len(remaining_teams) > 0 else "Away Team")

        home_players_list = list(team_players.get(home_name, {}).values())
        away_players_list = list(team_players.get(away_name, {}).values())

        if home_score == 0 and away_score == 0:
            home_player_goals = sum(p.goals for p in home_players_list)
            away_player_goals = sum(p.goals for p in away_players_list)

            home_crest_g = team_crest_goals[home_name].most_common(1)[0][0] if (home_name in team_crest_goals and team_crest_goals[home_name]) else 0
            away_crest_g = team_crest_goals[away_name].most_common(1)[0][0] if (away_name in team_crest_goals and team_crest_goals[away_name]) else 0

            home_gk_info = team_gk_saving.get(home_name)
            away_gk_info = team_gk_saving.get(away_name)

            home_gk_conceded = 0
            if home_gk_info:
                s_stats = home_gk_info["saving_stats"]
                home_gk_conceded = s_stats.get("goals_conceded", 0) + s_stats.get("penalty_goals_conceded", 0)

            away_gk_conceded = 0
            if away_gk_info:
                s_stats = away_gk_info["saving_stats"]
                away_gk_conceded = s_stats.get("goals_conceded", 0) + s_stats.get("penalty_goals_conceded", 0)

            home_score = max(home_player_goals, home_crest_g, away_gk_conceded)
            away_score = max(away_player_goals, away_crest_g, home_gk_conceded)

            all_mins = [p.minutes_played for p in home_players_list + away_players_list]
            max_mins = max(all_mins) if all_mins else 90
            if max_mins > 0:
                match_time = f"{max_mins}:00"

        home_clean_sheet = (away_score == 0)
        away_clean_sheet = (home_score == 0)

        def resolve_discipline_cards(player: OutfieldPlayerStats) -> None:
            """Infer double yellow vs. straight red card:
            - If red card detected:
                - fouls_committed >= 2: Double yellow -> 1 YC, 1 RC
                - fouls_committed <= 1: Straight red  -> 0 YC, 1 RC
            - If yellow card detected:
                - Standard yellow booking -> 1 YC, 0 RC
            - If no card detected:
                - Clean discipline -> 0 YC, 0 RC
            """
            if player.red_cards >= 1:
                if player.fouls_committed >= 2:
                    player.yellow_cards = 1
                    player.red_cards = 1
                else:
                    player.yellow_cards = 0
                    player.red_cards = 1
            elif player.yellow_cards >= 1:
                player.yellow_cards = 1
                player.red_cards = 0
            else:
                player.yellow_cards = 0
                player.red_cards = 0

        for p in home_players_list:
            p.clean_sheet = home_clean_sheet
            resolve_discipline_cards(p)
        for p in away_players_list:
            p.clean_sheet = away_clean_sheet
            resolve_discipline_cards(p)

        team_gk: Dict[str, Optional[GoalkeeperStats]] = {}
        for t_name, p_list in [(home_name, home_players_list), (away_name, away_players_list)]:
            opp_score = away_score if t_name == home_name else home_score
            opp_players = away_players_list if t_name == home_name else home_players_list
            gk_info = team_gk_saving.get(t_name)
            gk_player = next((p for p in p_list if p.position == "GK"), None)

            if gk_info and gk_player:
                clean_gk = re.sub(r'[^a-zA-Z]', '', gk_player.full_name).lower()
                clean_info = re.sub(r'[^a-zA-Z]', '', gk_info.get("full_name", "")).lower()
                from rapidfuzz import fuzz
                if fuzz.ratio(clean_gk, clean_info) < 50:
                    gk_info = None

            if not gk_player and gk_info:
                gk_player = next((p for p in p_list if p.full_name == gk_info["full_name"]), None)

            if gk_player:
                gk_name = gk_player.full_name
                if gk_info:
                    s_dict = dict(gk_info["saving_stats"])
                    total_conceded = s_dict.get("goals_conceded", 0) + s_dict.get("penalty_goals_conceded", 0)
                    s_dict["goals_conceded"] = max(total_conceded, opp_score)
                    s_dict["calculated_gk_stats"] = None
                else:
                    opp_shots = sum(p.shots for p in opp_players)
                    opp_sot = sum(int(round(p.shots * (p.shot_accuracy / 100.0))) for p in opp_players)
                    if opp_sot < opp_score:
                        opp_sot = opp_score
                    if opp_shots < opp_sot:
                        opp_shots = opp_sot
                    saves = max(0, opp_sot - opp_score)
                    save_pct = int(round((saves / opp_sot) * 100.0)) if opp_sot > 0 else 0

                    s_dict = {
                        "shots_against": opp_shots,
                        "shots_on_target": opp_sot,
                        "saves": saves,
                        "goals_conceded": opp_score,
                        "save_success_rate": save_pct,
                        "calculated_gk_stats": True
                    }

                team_gk[t_name] = GoalkeeperStats(
                    full_name=gk_name,
                    **s_dict
                )
            elif gk_info:
                gk_name = gk_info["full_name"]
                s_dict = dict(gk_info["saving_stats"])
                total_conceded = s_dict.get("goals_conceded", 0) + s_dict.get("penalty_goals_conceded", 0)
                s_dict["goals_conceded"] = max(total_conceded, opp_score)
                s_dict["calculated_gk_stats"] = None

                p_mins = 90
                if match_time and ":" in match_time:
                    try:
                        p_mins = int(match_time.split(":")[0])
                    except ValueError:
                        p_mins = 90

                p_obj = OutfieldPlayerStats(
                    full_name=gk_name,
                    short_name=gk_name.split()[0],
                    team_name=t_name,
                    position="GK",
                    overall_rating=gk_info["card_info"]["overall_rating"],
                    match_rating=gk_info["card_info"]["match_rating"],
                    man_of_the_match=False,
                    clean_sheet=(away_clean_sheet if t_name == away_name else home_clean_sheet),
                    yellow_cards=gk_info["card_info"].get("yellow_cards", 0),
                    red_cards=gk_info["card_info"].get("red_cards", 0),
                    minutes_played=p_mins
                )
                resolve_discipline_cards(p_obj)
                p_list.append(p_obj)

                team_gk[t_name] = GoalkeeperStats(
                    full_name=gk_name,
                    **s_dict
                )
            else:
                team_gk[t_name] = None

        motm_winner = None
        if motm_votes:
            top_cand, top_count = motm_votes.most_common(1)[0]
            from rapidfuzz import process, fuzz
            all_players = home_players_list + away_players_list
            best_p = None
            best_score = 0.0

            for p in all_players:
                s1 = fuzz.token_set_ratio(top_cand, p.full_name)
                s2 = fuzz.token_set_ratio(top_cand, p.short_name)
                s3 = fuzz.partial_ratio(top_cand.lower(), p.full_name.lower())
                s4 = fuzz.partial_ratio(top_cand.lower(), p.short_name.lower())
                score = max(s1, s2, s3, s4)
                if score > best_score:
                    best_score = score
                    best_p = p

            if best_p and best_score >= 50.0:
                best_p.man_of_the_match = True
                motm_winner = best_p.full_name

        home_team_stats = TeamMatchStats(
            team_name=home_name,
            score=home_score,
            players=home_players_list,
            goalkeeper=team_gk.get(home_name),
            clean_sheet=home_clean_sheet
        )

        away_team_stats = TeamMatchStats(
            team_name=away_name,
            score=away_score,
            players=away_players_list,
            goalkeeper=team_gk.get(away_name),
            clean_sheet=away_clean_sheet
        )

        return MatchRecord(
            home_team=home_team_stats,
            away_team=away_team_stats,
            man_of_the_match_player=motm_winner,
            match_time=match_time
        )

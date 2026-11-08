import re
from typing import Dict, List, Optional, Tuple, Any
from rapidfuzz import fuzz, process


class RosterMatcher:
    def __init__(self, roster_dict: Optional[Dict[str, str]] = None, team_name: Optional[str] = None):
        self.roster_dict = roster_dict or {}
        self.team_name = team_name
        self._reverse_map: Dict[str, str] = {v: k for k, v in self.roster_dict.items()}
        self._choices: List[str] = list(self.roster_dict.values())

    def update_roster(self, roster_dict: Dict[str, str], team_name: Optional[str] = None):
        self.roster_dict = roster_dict
        if team_name:
            self.team_name = team_name
        self._reverse_map = {v: k for k, v in self.roster_dict.items()}
        self._choices = list(self.roster_dict.values())

    @staticmethod
    def clean_name(name: str) -> str:
        return re.sub(r'[^a-zA-Z\s]', '', name).strip().lower()

    def match_player(self, full_name: str, short_name: str, position: str = "", player_team: Optional[str] = None) -> Tuple[Optional[str], str, float]:
        if not self._choices:
            return None, full_name, 0.0

        if self.team_name and player_team:
            from rapidfuzz import fuzz
            if fuzz.partial_ratio(self.team_name.lower(), player_team.lower()) < 60:
                return None, full_name, 0.0

        candidates = [
            full_name,
            short_name,
            f"{short_name} ({position})" if position else short_name,
            full_name.split()[-1] if full_name else "", # Last name
            full_name.split()[0] if full_name else "",  # First name
        ]
        candidates = [c for c in candidates if len(c.strip()) > 1]

        best_match_str = ""
        best_score = 0.0

        for q in candidates:
            m1 = process.extractOne(q, self._choices, scorer=fuzz.token_set_ratio)
            m2 = process.extractOne(q, self._choices, scorer=fuzz.partial_ratio)

            for m in [m1, m2]:
                if m and m[1] > best_score:
                    best_score = float(m[1])
                    best_match_str = m[0]

        if best_score >= 60.0 and best_match_str:
            p_id = self._reverse_map.get(best_match_str)
            return p_id, best_match_str, best_score

        return None, full_name, best_score
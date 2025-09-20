import re
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
from config.settings import config


class OmniscienceDataParser:
    """Parser that strictly maps 5-line, 4-line, and splits blocks and normalizes
    all numeric outputs to implied probability (IP)."""

    def __init__(self, cfg=config):
        self.cfg = cfg
        self.parsed_data: List[Dict[str, Any]] = []
        self.splits_data: List[Dict[str, Any]] = []

    # --------------------------
    # Helpers: odds conversions
    # --------------------------
    @staticmethod
    def _sanitize_int_token(tok: Optional[str]) -> Optional[int]:
        if tok is None:
            return None
        s = str(tok).strip()
        if s == "":
            return None
        if s.lower() == "even":
            return 100
        s = s.replace('+', '')
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else None

    @staticmethod
    def _american_to_prob_raw(odds: Optional[int]) -> Optional[float]:
        if odds is None:
            return None
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return float(-odds) / (float(-odds) + 100.0)

    @staticmethod
    def _normalize_two_way(a_odds: Optional[int], b_odds: Optional[int]) -> (Optional[float], Optional[float]):
        """Return normalized probabilities (p_a, p_b) that sum to 1 given two American odds.
        If one side missing, return raw prob for present side and None for other.
        """
        if a_odds is None and b_odds is None:
            return None, None
        a_raw = OmniscienceDataParser._american_to_prob_raw(a_odds) if a_odds is not None else 0.0
        b_raw = OmniscienceDataParser._american_to_prob_raw(b_odds) if b_odds is not None else 0.0
        total = (a_raw or 0.0) + (b_raw or 0.0)
        if total == 0:
            # fallback: if one side is present and other None
            return (a_raw, None) if b_odds is None else (None, b_raw)
        return (a_raw / total, b_raw / total)

    @staticmethod
    def _calc_opposite_vig(vig: Optional[int]) -> Optional[int]:
        if vig is None:
            return None
        return -220 - vig

    @staticmethod
    def _parse_percentage(tok: str) -> Optional[float]:
        if not tok:
            return None
        t = str(tok).strip().replace('%', '')
        try:
            return float(t)
        except Exception:
            m = re.search(r"\d+\.?\d*", t)
            return float(m.group(0)) if m else None

    # --------------------------
    # Feed parsing
    # --------------------------
    def parse_feed(self, text_feed: str, block_type: str = 'auto') -> pd.DataFrame:
        lines = [ln.strip() for ln in text_feed.splitlines() if ln.strip() != '']
        if not lines:
            return pd.DataFrame()

        start_idx = 1 if self.cfg.ignore_header_rows and len(lines) > 0 else 0
        cur: List[str] = []
        blocks: List[List[str]] = []
        i = start_idx
        while i < len(lines):
            cur.append(lines[i])
            i += 1
            # decide flush
            if block_type == '5line':
                expect = 5
            elif block_type == '4line':
                expect = 4
            else:
                expect = 5 if len(cur) >= 5 and self.cfg.parse_5_line_blocks else (4 if len(cur) >= 4 and self.cfg.parse_4_line_blocks else None)
            if expect and len(cur) == expect:
                blocks.append(cur.copy())
                cur = []
        if cur and len(cur) in (4,5):
            blocks.append(cur.copy())

        parsed = []
        for b in blocks:
            if len(b) == 5 and self.cfg.parse_5_line_blocks:
                p = self._parse_5line(b)
            elif len(b) == 4 and self.cfg.parse_4line_blocks:
                p = self._parse_4line(b)
            else:
                p = None
            if p:
                parsed.append(p)
                self.parsed_data.append(p)

        return pd.DataFrame(parsed)

    # --------------------------
    # 5-line mapping
    # --------------------------
    def _parse_5line(self, block: List[str]) -> Optional[Dict[str, Any]]:
        # block lines: 1 date time favorite spread | 2 spread vig | 3 total o/u | 4 total vig | 5 awayML homeML
        try:
            t1 = block[0].split()
            if len(t1) < 4:
                return None
            date_token, time_token = t1[0], t1[1]
            favorite_team = t1[2]
            spread_points = self._extract_point_value(t1[3])

            spread_vig_raw = block[1].strip()
            spread_vig = self._sanitize_int_token(spread_vig_raw)
            spread_vig_opp = self._calc_opposite_vig(spread_vig)
            fav_ip_raw, dog_ip_raw = self._normalize_two_way(spread_vig, spread_vig_opp)

            total_token = block[2].strip()
            total_side = None
            if len(total_token) >= 2 and total_token[0].lower() in ('o', 'u'):
                total_side = 'over' if total_token[0].lower() == 'o' else 'under'
                total_points = self._extract_point_value(total_token[1:])
            else:
                total_points = self._extract_point_value(total_token)

            total_vig = self._sanitize_int_token(block[3])
            total_vig_opp = self._calc_opposite_vig(total_vig)
            over_ip_raw, under_ip_raw = self._normalize_two_way(total_vig, total_vig_opp)

            ml_matches = re.findall(r"[+-]?\d+", block[4])
            away_ml = self._sanitize_int_token(ml_matches[0]) if len(ml_matches) > 0 else None
            home_ml = self._sanitize_int_token(ml_matches[1]) if len(ml_matches) > 1 else None
            away_ml_ip_raw, home_ml_ip_raw = self._normalize_two_way(away_ml, home_ml)

            parsed = {
                'game_id': f"{date_token}|{time_token}|{favorite_team}",
                'date': date_token,
                'time': time_token,
                'favorite_team': favorite_team,
                'spread_points': spread_points,
                'favorite_ip_raw': fav_ip_raw,
                'dog_ip_raw': dog_ip_raw,
                'spread_vig': spread_vig,
                'spread_vig_opp': spread_vig_opp,
                'total_points': total_points,
                'total_side': total_side,
                'over_ip_raw': over_ip_raw,
                'under_ip_raw': under_ip_raw,
                'total_vig': total_vig,
                'total_vig_opp': total_vig_opp,
                'away_ml': away_ml,
                'home_ml': home_ml,
                'away_ml_ip_raw': away_ml_ip_raw,
                'home_ml_ip_raw': home_ml_ip_raw,
                'parsed_at': datetime.utcnow().isoformat()
            }
            return parsed
        except Exception:
            return None

    # --------------------------
    # 4-line mapping
    # --------------------------
    def _parse_4line(self, block: List[str]) -> Optional[Dict[str, Any]]:
        # line1: date time awayML homeML total | 2 total vig | 3 team runline | 4 runline vig
        try:
            t1 = block[0].split()
            if len(t1) < 5:
                return None
            date_token, time_token = t1[0], t1[1]
            away_ml = self._sanitize_int_token(t1[2])
            home_ml = self._sanitize_int_token(t1[3])
            away_ml_ip_raw, home_ml_ip_raw = self._normalize_two_way(away_ml, home_ml)
            total_token = t1[4]
            total_points = self._extract_point_value(total_token)

            total_vig = self._sanitize_int_token(block[1])
            total_vig_opp = self._calc_opposite_vig(total_vig)
            over_ip_raw, under_ip_raw = self._normalize_two_way(total_vig, total_vig_opp)

            rtokens = block[2].split()
            runline_points = self._extract_point_value(rtokens[1]) if len(rtokens) >= 2 else None
            runline_vig = self._sanitize_int_token(block[3])
            runline_vig_opp = self._calc_opposite_vig(runline_vig)
            runline_vig_ip, runline_vig_opp_ip = self._normalize_two_way(runline_vig, runline_vig_opp)

            parsed = {
                'game_id': f"{date_token}|{time_token}|{away_ml}|{home_ml}",
                'date': date_token,
                'time': time_token,
                'away_ml': away_ml,
                'home_ml': home_ml,
                'away_ml_ip_raw': away_ml_ip_raw,
                'home_ml_ip_raw': home_ml_ip_raw,
                'total_points': total_points,
                'total_vig': total_vig,
                'total_vig_opp': total_vig_opp,
                'over_ip_raw': over_ip_raw,
                'under_ip_raw': under_ip_raw,
                'runline_points': runline_points,
                'runline_vig': runline_vig,
                'runline_vig_opp': runline_vig_opp,
                'parsed_at': datetime.utcnow().isoformat()
            }
            return parsed
        except Exception:
            return None

    # --------------------------
    # Splits parser: 8-line block, lines 1-3 ignored, 4-7 contain percentages
    # --------------------------
    def parse_splits_block(self, text_block: str) -> Optional[Dict[str, Any]]:
        lines = [l.strip() for l in text_block.splitlines() if l.strip() != '']
        if len(lines) < 7:
            return None
        away_bet_pct = self._parse_percentage(lines[3])
        home_bet_pct = self._parse_percentage(lines[4])
        away_money_pct = self._parse_percentage(lines[5])
        home_money_pct = self._parse_percentage(lines[6])
        parsed = {
            'away_bet_pct': away_bet_pct,
            'home_bet_pct': home_bet_pct,
            'away_money_pct': away_money_pct,
            'home_money_pct': home_money_pct,
            'parsed_at': datetime.utcnow().isoformat()
        }
        self.splits_data.append(parsed)
        return parsed

    # --------------------------
    # small helper
    # --------------------------
    @staticmethod
    def _extract_point_value(token: str) -> Optional[float]:
        if token is None:
            return None
        m = re.search(r"-?\d+\.?\d*", str(token))
        return float(m.group(0)) if m else None

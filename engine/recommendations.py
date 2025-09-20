from typing import Dict, Any
from datetime import datetime
from config.settings import config
import numpy as np


class RecommendationEngine:
    def __init__(self, cfg=config):
        self.cfg = cfg

    def generate_recommendation(self, game_row: Dict[str, Any], ta_indicators: Dict[str, Any], forecasts: Dict[str, Any]) -> Dict[str, Any]:
        score = 0.0
        narrative = []
        triggered = []

        # Use spread signals if available
        if 'spread' in ta_indicators and ta_indicators['spread']:
            s = ta_indicators['spread']
            mom = s.get('momentum', {})
            mom_v = mom.get('MOM_V') if mom else 0.0
            mom_a = mom.get('MOM_A') if mom else 0.0
            if mom_v is not None:
                score += np.sign(mom_v) * min(0.5, abs(mom_v) * 100)
                narrative.append(f"Spread MOM-V: {mom_v:.6f}")
                triggered.append('spread_momentum')
            # steam
            steam = s.get('steam_detection', {})
            if steam and steam.get('steam'):
                score += 0.3 * steam.get('confidence', 0.0)
                narrative.append(f"Steam detected (conf {steam.get('confidence',0):.2f})")
                triggered.append('steam')

        # Moneyline signals
        for side in ('away_ml', 'home_ml'):
            if side in ta_indicators and ta_indicators[side]:
                t = ta_indicators[side]
                momv = t.get('momentum', {}).get('MOM_V') if t.get('momentum') else 0.0
                if momv:
                    score += np.sign(momv) * min(0.4, abs(momv) * 120)
                    triggered.append(f"{side}_momentum")

        # Forecasts: if forecasts indicate imminent movement that improves edge, increase score
        for mk, f in forecasts.items():
            if not f:
                continue
            pm = f.get('projected_point_move')
            conf = f.get('confidence', 0.0)
            if pm and abs(pm) > 0.3 and conf > 0.6:
                narrative.append(f"Forecast {mk}: move {pm:.2f} pts (conf {conf:.2f})")
                score += 0.2 * np.sign(pm)

        # Normalize score into confidence
        confidence = 0.5 + score
        if confidence > 0.95:
            confidence = 0.95
        if confidence < 0.05:
            confidence = 0.05

        # Decide action
        if confidence >= self.cfg.strong_confidence_threshold:
            action = 'Back'
        elif confidence <= (self.cfg.min_confidence_for_action - 0.1):
            action = 'Fade'
        else:
            action = 'Hold'

        # crude EV and Kelly
        ev = (confidence - 0.5) * 0.12
        kelly = max(0.0, min(self.cfg.kelly_fraction_cap, ev / 0.05))

        return {
            'game_id': game_row.get('game_id'),
            'action': action,
            'confidence': float(confidence),
            'expected_value': float(ev),
            'kelly_stake': float(kelly),
            'narrative': '\n'.join(narrative),
            'triggered_indicators': triggered,
            'indicator_summary': ta_indicators,
            'timestamp': datetime.utcnow().isoformat()
        }

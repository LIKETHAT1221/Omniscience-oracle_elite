import numpy as np
from typing import List, Dict, Any, Optional

# Basic IP/point ta utilities for Omniscience

def momentum_from_ips(values: List[float], period: int = 3) -> Dict[str, Optional[float]]:
    arr = np.array(values)
    if len(arr) < period + 1:
        return {'MOM_V': None, 'MOM_A': None}
    mom_v = (arr[-1] - arr[-period-1]) / period
    mom_a = None
    if len(arr) >= 2 * period + 1:
        prev_v = (arr[-period-1] - arr[-2*period-1]) / period
        mom_a = (mom_v - prev_v) / period
    return {'MOM_V': float(mom_v), 'MOM_A': float(mom_a) if mom_a is not None else None}


def rsi_from_ips(values: List[float], period: int = 14) -> Optional[float]:
    x = np.array(values)
    if len(x) < period + 1:
        return None
    deltas = np.diff(x)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def z_score(values: List[float], lookback: int = 20) -> Optional[float]:
    arr = np.array(values)
    if len(arr) < lookback:
        return None
    mu = np.mean(arr[-lookback:])
    sd = np.std(arr[-lookback:])
    if sd == 0:
        return 0.0
    return float((arr[-1] - mu) / sd)


def sma(values: List[float], period: int = 10) -> Optional[float]:
    if len(values) < 1:
        return None
    return float(np.mean(values[-period:]))


def ema(values: List[float], period: int = 10) -> Optional[float]:
    arr = np.array(values)
    if len(arr) < 1:
        return None
    alpha = 2.0 / (period + 1.0)
    ema_v = arr[0]
    for v in arr[1:]:
        ema_v = alpha * v + (1 - alpha) * ema_v
    return float(ema_v)


def adaptive_ma(values: List[float], base_period: int = 10, max_period: int = 30, sensitivity: float = 2.0) -> Optional[float]:
    if not values:
        return None
    arr = np.array(values)
    if len(arr) < base_period:
        return float(np.mean(arr))
    vol = np.std(arr[-base_period:])
    if vol == 0:
        eff = 0.0
    else:
        direction = abs(arr[-1] - arr[-base_period])
        eff = direction / (vol * np.sqrt(base_period))
    adaptive_period = base_period + int((max_period - base_period) * eff * sensitivity)
    adaptive_period = max(base_period, min(max_period, adaptive_period))
    return float(np.mean(arr[-adaptive_period:]))


def bollinger_width(values: List[float], lookback: int = 20) -> Optional[float]:
    arr = np.array(values)
    if len(arr) < lookback:
        return None
    mu = np.mean(arr[-lookback:])
    sd = np.std(arr[-lookback:])
    return float((mu + 2 * sd) - (mu - 2 * sd))


def atr_on_points(points: List[float], lookback: int = 14) -> Optional[float]:
    if len(points) < 2:
        return None
    tr = np.abs(np.diff(points))
    if len(tr) < lookback:
        return float(np.mean(tr))
    return float(np.mean(tr[-lookback:]))


def fibonacci_levels(points: List[float], lookback: int = 50) -> Dict[str, float]:
    out = {}
    if len(points) < 2:
        return out
    s = points[-lookback:]
    high = max(s)
    low = min(s)
    diff = high - low if high != low else 1.0
    out = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }
    return out


def fibonacci_extensions(points: List[float], lookback: int = 50) -> Dict[str, float]:
    out = {}
    if len(points) < 2:
        return out
    s = points[-lookback:]
    high = max(s)
    low = min(s)
    diff = high - low if high != low else 1.0
    out = {
        '1.272': high + 0.272 * diff,
        '1.414': high + 0.414 * diff,
        '1.618': high + 0.618 * diff,
        '2.0': high + 1.0 * diff,
        '2.618': high + 1.618 * diff
    }
    return out


def detect_steam_movement_advanced(values_ip: List[float], values_points: Optional[List[float]] = None, splits: Optional[Dict] = None) -> Dict[str, Any]:
    """Return steam detection summary. Uses z-score, momentum, volatility, and splits if present."""
    out = {'steam': False, 'confidence': 0.0, 'signals': []}
    if not values_ip or len(values_ip) < 6:
        return out
    zs = z_score(values_ip, lookback=20) or 0.0
    mom = momentum_from_ips(values_ip, period=3)
    mom_v = mom.get('MOM_V') or 0.0
    vol = np.std(values_ip[-10:]) if len(values_ip) >= 10 else np.std(values_ip)

    score = 0.0
    if abs(zs) > 2.0:
        score += 1.0
        out['signals'].append('zscore')
    if abs(mom_v) > 0.0005:
        score += 1.0
        out['signals'].append('momentum')
    if vol > 0.02:
        score += 0.5
        out['signals'].append('volatility')

    # incorporate splits if provided (heavy weight)
    if splits and ('away_money_pct' in splits and 'home_money_pct' in splits):
        if max(splits['away_money_pct'] or 0, splits['home_money_pct'] or 0) > 60:
            score += 1.0
            out['signals'].append('sharp_splits')

    out['confidence'] = min(1.0, score / 3.0)
    out['steam'] = out['confidence'] > 0.4
    return out


def calculate_greeks_estimate(values_ip: List[float], points: Optional[List[float]] = None) -> Dict[str, Optional[float]]:
    """Estimate Greeks-like sensitivities: delta (dIP/dPoint), gamma (d2IP/dPoint2), vega ~ sensitivity to vol."""
    out = {'delta': None, 'gamma': None, 'vega': None}
    if not values_ip or points is None or len(values_ip) < 3 or len(points) < 3:
        return out
    dip = np.diff(values_ip)
    dp = np.diff(points)
    ratios = [a / b for a, b in zip(dip, dp) if abs(b) > 1e-8]
    if not ratios:
        return out
    out['delta'] = float(np.median(ratios))
    # gamma ~ change in delta
    if len(ratios) >= 2:
        out['gamma'] = float(np.median(np.diff(ratios)))
    out['vega'] = float(np.std(dip))
    return out


def implied_volatility_simple(values_ip: List[float]) -> Optional[float]:
    if not values_ip or len(values_ip) < 2:
        return None
    returns = np.diff(values_ip)
    return float(np.std(returns))


def calculate_all_ta_indicators(series_data: List[Dict], field: str = 'ip', point_field: str = 'point') -> Dict[str, Any]:
    values_ip = [d[field] for d in series_data if field in d and d[field] is not None]
    values_points = [d[point_field] for d in series_data if point_field in d and d[point_field] is not None]
    out: Dict[str, Any] = {}
    if not values_ip:
        return out
    out['current_ip'] = float(values_ip[-1])
    out['data_points'] = len(values_ip)
    out['momentum'] = momentum_from_ips(values_ip, period=3)
    out['rsi'] = rsi_from_ips(values_ip, period=14)
    out['z_score'] = z_score(values_ip, lookback=20)
    out['sma'] = sma(values_ip, period=10)
    out['ema'] = ema(values_ip, period=10)
    out['adaptive_ma'] = adaptive_ma(values_ip)
    out['bollinger_width'] = bollinger_width(values_ip, lookback=20)
    out['atr'] = atr_on_points(values_points, lookback=14) if values_points else None
    out['fib_retracement'] = fibonacci_levels(values_points, lookback=50) if values_points else {}
    out['fib_extensions'] = fibonacci_extensions(values_points, lookback=50) if values_points else {}
    out['steam_detection'] = detect_steam_movement_advanced(values_ip, values_points)
    out['greeks'] = calculate_greeks_estimate(values_ip, values_points) if values_points else {}
    out['implied_volatility'] = implied_volatility_simple(values_ip)
    out['series'] = values_ip[-500:]
    return out

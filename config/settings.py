class Config:
    # Parsing toggles
    ignore_header_rows = True
    parse_5_line_blocks = True
    parse_4_line_blocks = True

    # Recommendation thresholds
    strong_confidence_threshold = 0.80
    min_confidence_for_action = 0.55

    # LMF defaults (minutes)
    lmf_horizons = [30, 60, 90]

    # TA defaults
    default_history_len = 300
    rsi_period = 14
    zscore_lookback = 20
    bollinger_lookback = 20
    atr_lookback = 14

    # Kelly parameters
    kelly_fraction_cap = 0.20

config = Config()

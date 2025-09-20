import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# add repo root so imports work when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from parser.odds_parser import OmniscienceDataParser
from ta.ta_engine import calculate_all_ta_indicators, calculate_all_ta_indicators as ta_wrapper  # alias
from engine.recommendations import RecommendationEngine
from config.settings import config


st.set_page_config(page_title='Omniscience Oracle', layout='wide')
st.title('🔮 Omniscience Oracle - Modular')

# init
if 'parser' not in st.session_state:
    st.session_state['parser'] = OmniscienceDataParser()
if 'rec_engine' not in st.session_state:
    st.session_state['rec_engine'] = RecommendationEngine()
if 'buffers' not in st.session_state:
    st.session_state['buffers'] = {}

with st.sidebar:
    parser_mode = st.radio('Parser mode', options=['5line', '4line'])
    run_auto = st.checkbox('Auto TA+Rec after parse', value=True)
    clear_buf = st.button('Clear buffers')
    if clear_buf:
        st.session_state['buffers'] = {}
        st.success('buffers cleared')

st.markdown('Paste your odds feed below (one feed type at a time).')
feed = st.text_area('Odds feed', height=240)

st.markdown('Optional splits feed (separate box)')
splits = st.text_area('Splits feed', height=120)

col1, col2 = st.columns(2)
if col1.button('Parse Feed'):
    if not feed.strip():
        st.warning('No feed provided')
    else:
        df = st.session_state['parser'].parse_feed(feed, block_type=parser_mode)
        if df.empty:
            st.warning('No blocks parsed')
        else:
            st.dataframe(df.fillna(''))
            st.success(f'Parsed {len(df)} blocks')

            # push to buffers
            now = pd.Timestamp.utcnow()
            for _, r in df.iterrows():
                gid = r.get('game_id')
                def push(market, ip_val, point_val=None):
                    if ip_val is None:
                        return
                    key = f"{gid}|{market}"
                    buf = st.session_state['buffers'].setdefault(key, [])
                    buf.append({'ip': float(ip_val), 'point': float(point_val) if point_val is not None else None, 'ts': now.isoformat()})
                    if len(buf) > config.default_history_len:
                        st.session_state['buffers'][key] = buf[-config.default_history_len:]

                push('away_ml', r.get('away_ml_ip_raw'))
                push('home_ml', r.get('home_ml_ip_raw'))
                push('spread', r.get('favorite_ip_raw'), r.get('spread_points'))
                push('total', r.get('over_ip_raw'), r.get('total_points'))

            if run_auto:
                recs = []
                for _, r in df.iterrows():
                    gid = r.get('game_id')
                    ta_indicators = {}
                    forecasts = {}
                    for market in ['away_ml','home_ml','spread','total']:
                        key = f"{gid}|{market}"
                        buf = st.session_state['buffers'].get(key, [])
                        if buf:
                            series = [{'ip': p['ip'], 'point': p['point']} for p in buf]
                            ta = calculate_all_ta_indicators(series, field='ip', point_field='point')
                            ta_indicators[market] = ta
                            if market in ('spread','total'):
                                pts = [p['point'] for p in buf if p.get('point') is not None]
                                ips = [p['ip'] for p in buf]
                                if pts and ips:
                                    # use lmf from ta_engine by calling its functions if exported (already inside ta module)
                                    from ta.ta_engine import lmf_forecast
                                    f = lmf_forecast(pts, ips, horizon_minutes=60)
                                    forecasts[market] = f

                    rec = st.session_state['rec_engine'].generate_recommendation(r.to_dict(), ta_indicators, forecasts)
                    recs.append(rec)

                rec_df = pd.DataFrame(recs)
                st.subheader('Recommendations')
                st.dataframe(rec_df.fillna(''))

if col2.button('Parse Splits'):
    if not splits.strip():
        st.warning('No splits text')
    else:
        blocks = [b for b in splits.split('\n\n') if b.strip()]
        parsed = []
        for b in blocks:
            p = st.session_state['parser'].parse_splits_block(b)
            if p:
                parsed.append(p)
        if parsed:
            st.dataframe(pd.DataFrame(parsed))
        else:
            st.warning('No splits parsed')

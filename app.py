
import streamlit as st
import crypto
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
st.title("Crypto Signal: Decision Tree / Random Forest / XGBoost")
st.caption("Supervised learning on free Binance data. Paper money only, not financial advice.")
uploaded = st.file_uploader("Upload a Binance trades CSV", type=["csv"])
use_demo = st.checkbox("Use random demo data instead (no file needed)")
if uploaded or use_demo:
    if use_demo:
        n=69715
        start=datetime(2026,10,11)
        df_demo = pd.DataFrame({"timestamp":[start+timedelta(seconds=i*30) for i in range(n)], "price":83000+np.cumsum(np.random.randn(n)*10), "quantity":np.random.uniform(0.001,1,n), "side":np.random.choice(["buy","sell"],n)})
        bars = crypto.features(crypto.to_bars(df_demo, 1))
    else:
        bars = crypto.features(crypto.to_bars(crypto.load_trades(uploaded, nrows=500000), 1))
    st.write(f"Loaded {len(bars):,} bars")
    st.line_chart(bars.set_index("timestamp")["price"].tail(5000))
    if st.button("Train models"):
        models = crypto.get_models()
        results = []
        for name, m in models.items():
            acc, base = crypto.train(bars, m)
            results.append({"Model":name, "Accuracy":acc, "Baseline":base})
        st.table(results)
    model_choice = st.selectbox("Model for live signal", list(crypto.get_models().keys()))
    if st.button("Get live signal"):
        models = crypto.get_models()
        crypto.train(bars, models[model_choice])
        sig, conf, price = crypto.live_signal(models[model_choice])
        c1,c2,c3 = st.columns(3)
        c1.metric("Signal", sig)
        c2.metric("Confidence", f"{conf:.0%}")
        c3.metric("Price", f"{price:,.2f}")
        st.info(crypto.paper_plan(sig, price))

import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="AEO Scanner", layout="wide")
st.title("AEO Scanner — NAB (Credit Cards)")

# --- load the saved analysis (reads disk, not the notebook kernel) ---
with open("nab_credit_cards_analysis.json") as f:
    analysis_results = json.load(f)

# flatten to one row per entity
rows = []
for r in analysis_results:
    for e in r["entities"]:
        rows.append({
            "model_requested": r["model_requested"],
            "question": r["question"],
            "name": e["name"],
            "isCompetitor": e["isCompetitor"],
            "sentiment": e["sentiment"],
        })
entities_df = pd.DataFrame(rows)

# --- Panel 1: NAB subject view ---
st.header("NAB — how the brand itself is covered")
nab = entities_df[entities_df["name"] == "NAB"]
col1, col2 = st.columns(2)
with col1:
    st.metric("Answers mentioning NAB", f"{nab.shape[0]} / {len(analysis_results)}")
with col2:
    st.metric("Positive mentions", (nab["sentiment"] == "positive").sum())
st.bar_chart(nab["sentiment"].value_counts())

# --- Panel 2: competitor field ---
st.header("The competitive field")
competitors = entities_df[entities_df["isCompetitor"] == True]
share = competitors["name"].value_counts()
st.bar_chart(share)

st.subheader("Sentiment by competitor")
sentiment_pivot = (
    competitors.groupby("name")["sentiment"]
    .value_counts()
    .unstack(fill_value=0)
)
st.dataframe(sentiment_pivot)
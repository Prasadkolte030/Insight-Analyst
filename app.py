from __future__ import annotations

import pandas as pd
import streamlit as st

from analyst import analyze, load_kaggle_data, make_demo_data, prepare_dataset

st.set_page_config(page_title="Insight Analyst", page_icon="📊", layout="wide")

st.title("Insight Analyst")
st.caption(
    "Ask a question about a dataset. The analyst makes its action plan visible, "
    "chooses analysis tools, and explains what the results mean."
)
st.caption(
    "Uses the provided Kaggle Global Superstore CSV by default. "
    "You can also upload another CSV or switch to synthetic demo data."
)

with st.sidebar:
    st.header("Your data")
    source = st.radio(
        "Choose a dataset",
        ("Provided Kaggle data", "Upload a CSV", "Try synthetic demo"),
    )
    uploaded = None
    date_note = None
    if source == "Provided Kaggle data":
        try:
            frame, date_note = prepare_dataset(load_kaggle_data())
        except FileNotFoundError:
            frame = None
            st.error(
                "The bundled dataset is missing. Include "
                "`Global Superstore.csv` in the deployed repository, "
                "or choose another data source."
            )
        except pd.errors.EmptyDataError:
            frame = None
            st.error(
                "The bundled dataset file is empty. Replace "
                "`Global Superstore.csv` in the deployed repository with "
                "the full CSV, then redeploy."
            )
        else:
            st.success(f"Kaggle data ready: {len(frame):,} rows.")
            if date_note:
                st.warning(date_note)
    elif source == "Upload a CSV":
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        st.caption("Your file is processed in this app session and is not sent to an LLM.")
        frame = None
        if uploaded is not None:
            try:
                frame, date_note = prepare_dataset(pd.read_csv(uploaded))
            except (
                UnicodeDecodeError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                ValueError,
            ) as error:
                st.error(f"Could not read that CSV: {error}")
        if frame is not None:
            st.success(f"Loaded {len(frame):,} rows and {len(frame.columns)} columns.")
    else:
        frame = make_demo_data()
        st.success(f"Demo data ready: {len(frame):,} rows.")
    st.caption(
        "Understands fields such as Sales, Profit, Order Date, "
        "Region, Category, Sub-Category, and Product Name."
    )

if frame is None:
    st.info("Choose an available data source in the sidebar to start analyzing.")
    st.stop()
if frame.empty or frame.shape[1] == 0:
    st.error("This dataset is empty or has no columns.")
    st.stop()

with st.expander("Preview data"):
    st.dataframe(frame.head(20), use_container_width=True)
    st.caption(f"Columns: {', '.join(str(column) for column in frame.columns)}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_dataset" not in st.session_state:
    st.session_state.last_dataset = None
dataset_key = (
    "provided-kaggle-data"
    if source == "Provided Kaggle data"
    else "demo"
    if source == "Try synthetic demo"
    else f"{uploaded.name}:{uploaded.size}" if uploaded is not None else None
)
if dataset_key != st.session_state.last_dataset:
    st.session_state.messages = []
    st.session_state.last_dataset = dataset_key

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("plan"):
            with st.expander("Inspect the analysis plan and tool results"):
                st.markdown(message["plan"])
                for item in message["results"]:
                    st.markdown(f"**{item['tool'].title()} tool**")
                    st.json(item["data"])

question = st.chat_input("e.g. Top 3 products by sales last quarter")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.status("Planning and analyzing the data...", expanded=True) as status:
            result = analyze(question, frame)
            st.markdown(f"**Plan:** {result['plan_text']}")
            for item in result["results"]:
                st.write(f"Ran the **{item['tool']}** tool.")
                st.json(item["data"])
            status.update(label="Analysis complete", state="complete", expanded=False)
        st.markdown(result["answer"])
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "plan": result["plan_text"],
            "results": result["results"],
        }
    )

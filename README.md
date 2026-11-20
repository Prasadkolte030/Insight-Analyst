---
title: Insight Analyst
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Insight Analyst

A Streamlit chatbot orchestrated with LangGraph and the user-provided Kaggle `Global Superstore.csv` (51,290 rows). The agent maps fields such as `Sales`, `Profit`, `Region`, `Category`, `Sub-Category`, and `Product Name`, then uses a visible plan and distinct tools for product rankings, group comparisons, seasonal trends, performance drivers, dataset profiles, and IQR anomaly detection. It synthesizes results in plain language and does not expose private chain-of-thought.

## Try it

The app loads the provided Kaggle data by default. Example questions:

- “What were the top 3 products by sales last quarter?”
- “Is there a seasonal trend in sales for category Technology?”
- “Compare East and West performance and tell me what's driving the difference.”
- “Summarize anything unusual in this data.”

The bundled source file is `Global Superstore.csv`. The supplied file has unusable `Order Date` and `Ship Date` values (`00:00.0` throughout), but contains `Year` and `weeknum`. The app reconstructs approximate weekly midpoints for time analyses and labels quarter/seasonality results as estimates; it does not pretend those are exact dates. Choose **Upload a CSV** for another file or **Try synthetic demo** to see a clean date-series example. The app does not require an LLM or API key.

## Public deployment (free Hugging Face Spaces)

1. Create a **public Docker Space** at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Clone its Git repository, copy this project into it, and push the files to the Space's `main` branch.
3. Wait for the Space to build; its public page is the app URL. Free Spaces may sleep after inactivity and take a short time to wake when opened.

No API keys or hosting secrets are needed. This Space includes the provided Kaggle CSV as its default dataset; do not add private datasets to a public Space.

## Run locally (optional)

Requires Python 3.11 or newer:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## How the graph works

`plan` selects one or more analyses from the question and maps likely measure, product, group, and time columns. It supports latest-quarter ranking, month-of-year seasonality, filters for named categories, and region-driver breakdowns. For the supplied CSV, these time results are estimates derived from `Year` and `weeknum`. A conditional `dispatch` edge routes each selection to its own tool node, then routes again for remaining tools. `synthesize` turns the structured results into an insight. The plan and tool outputs are available in the chat's inspection panel.

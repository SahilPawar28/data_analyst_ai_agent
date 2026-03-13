import streamlit as st
import pandas as pd
import json
import time
import requests

st.set_page_config(
    page_title="🤖 Agentic Data Analyst",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.agent-card {
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    border: 1px solid #4a4a6a;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    color: white;
}
.agent-running { border-left: 4px solid #f59e0b; }
.agent-done    { border-left: 4px solid #10b981; }
.agent-wait    { border-left: 4px solid #6b7280; }
.insight-box {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #3b82f6;
    border-radius: 12px;
    padding: 24px;
    color: #e2e8f0;
    font-size: 15px;
    line-height: 1.8;
    white-space: pre-wrap;
}
.stApp { background-color: #0d0d1a; }
h1, h2, h3 { color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic AI Data Analyst")
st.caption("Upload your dataset → Multi-agent pipeline → AI Insights (Free via OpenRouter)")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenRouter API Key", type="password",
                            help="Get free key at openrouter.ai")
    st.divider()
    st.markdown("**Pipeline Agents**")
    st.markdown("1. 📥 Ingestion Agent\n2. 🔍 Schema Agent\n3. 🧹 Quality Agent\n4. 📊 Stats Agent\n5. 🔗 Correlation Agent\n6. 🧠 AI Insight Agent")
    st.divider()
    st.info("Works with CSV, Excel, JSON")
    st.markdown("🔑 Get **free** API key:\n[openrouter.ai](https://openrouter.ai)")
    st.markdown("**Free model used:**\n`stepfun/step-3.5-flash:free`")

uploaded = st.file_uploader("Upload your dataset", type=["csv","xlsx","xls","json"])

def load_file(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    elif name.endswith((".xlsx",".xls")):
        return pd.read_excel(f)
    elif name.endswith(".json"):
        data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            try:
                return pd.DataFrame(data)
            except Exception:
                return pd.DataFrame([data])
    raise ValueError("Unsupported file type")

def agent_ingestion(df):
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "size_kb": round(df.memory_usage(deep=True).sum() / 1024, 2)
    }

def agent_schema(df):
    schema = {}
    for col in df.columns:
        schema[col] = {
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isnull().sum()),
            "unique": int(df[col].nunique())
        }
    return schema

def agent_quality(df):
    issues = []
    for col, cnt in df.isnull().sum().items():
        if cnt > 0:
            pct = round(cnt / len(df) * 100, 1)
            issues.append(f"'{col}' has {cnt} missing values ({pct}%)")
    dups = int(df.duplicated().sum())
    if dups:
        issues.append(f"{dups} duplicate rows found")
    const = [c for c in df.columns if df[c].nunique() <= 1]
    if const:
        issues.append(f"Constant/empty columns: {const}")
    return {"issues": issues, "quality_score": max(0, 100 - len(issues) * 10)}

def agent_stats(df):
    stats = {}
    num_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(include=["object","category","bool"])
    if not num_df.empty:
        stats["numeric_summary"] = num_df.describe().round(3).to_dict()
    if not cat_df.empty:
        cat_summary = {}
        for col in cat_df.columns[:8]:
            top = df[col].value_counts().head(3).to_dict()
            cat_summary[col] = {str(k): int(v) for k, v in top.items()}
        stats["categorical_summary"] = cat_summary
    return stats

def agent_correlation(df):
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return {"note": "Not enough numeric columns for correlation"}
    corr = num_df.corr().round(3)
    strong = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            v = corr.iloc[i, j]
            if abs(v) >= 0.5:
                strong.append({"col1": cols[i], "col2": cols[j], "r": round(v,3)})
    strong.sort(key=lambda x: abs(x["r"]), reverse=True)
    return {"strong_correlations": strong[:10], "matrix_shape": corr.shape[0]}

def agent_llm_insights(summary_text, api_key):
    prompt = f"""You are a senior data scientist. Analyze the following dataset summary produced by an automated multi-agent pipeline and provide comprehensive, actionable insights.

{summary_text}

Please provide:
1. Dataset Overview - what this dataset appears to be about
2. Data Quality Assessment - issues found and recommendations
3. Key Statistical Findings - notable patterns, distributions, outliers
4. Relationships & Correlations - important variable relationships
5. Actionable Recommendations - what analysis or actions to take next
6. Potential Use Cases - what business questions this data can answer

Be specific, insightful, and concise. Use bullet points where helpful."""

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://agentic-data-analyst.streamlit.app",
            "X-Title": "Agentic Data Analyst",
            "Accept-Charset": "utf-8"
        },
        json={
            "model": "stepfun/step-3.5-flash:free",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        },
        stream=True
    )

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    for line in response.iter_lines(decode_unicode=True):
        if line:
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue

if uploaded:
    try:
        df = load_file(uploaded)
        st.success(f"✅ File loaded: **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")

        if st.button("🚀 Run Agentic Pipeline", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your OpenRouter API key in the sidebar.")
                st.stop()

            results = {}
            agents = [
                ("📥 Ingestion Agent",   "Parsing file & extracting metadata",    agent_ingestion),
                ("🔍 Schema Agent",       "Analysing column types & cardinality",  agent_schema),
                ("🧹 Quality Agent",      "Detecting missing values & duplicates", agent_quality),
                ("📊 Stats Agent",        "Computing descriptive statistics",       agent_stats),
                ("🔗 Correlation Agent",  "Finding variable relationships",        agent_correlation),
            ]

            st.markdown("## 🔄 Agent Pipeline")
            placeholders = []
            for name, desc, _ in agents:
                ph = st.empty()
                ph.markdown(f'<div class="agent-card agent-wait">⏳ <b>{name}</b><br><small>{desc}</small></div>', unsafe_allow_html=True)
                placeholders.append(ph)

            for idx, (name, desc, fn) in enumerate(agents):
                placeholders[idx].markdown(
                    f'<div class="agent-card agent-running">🔄 <b>{name}</b><br><small>{desc} — running…</small></div>',
                    unsafe_allow_html=True)
                time.sleep(0.4)
                key = name.split(" ")[1].lower()
                results[key] = fn(df)
                placeholders[idx].markdown(
                    f'<div class="agent-card agent-done">✅ <b>{name}</b><br><small>{desc} — complete</small></div>',
                    unsafe_allow_html=True)

            st.markdown("## 📋 Agent Results")
            t1, t2, t3, t4, t5 = st.tabs(["Ingestion","Schema","Quality","Stats","Correlations"])
            with t1: st.json(results["ingestion"])
            with t2: st.json(results["schema"])
            with t3:
                q = results["quality"]
                score = q["quality_score"]
                color = "🟢" if score >= 80 else "🟡" if score >= 50 else "🔴"
                st.metric("Quality Score", f"{color} {score}/100")
                if q["issues"]:
                    for iss in q["issues"]: st.warning(iss)
                else:
                    st.success("No data quality issues found!")
            with t4: st.json(results["stats"])
            with t5:
                corr = results["correlation"]
                if "strong_correlations" in corr and corr["strong_correlations"]:
                    st.dataframe(pd.DataFrame(corr["strong_correlations"]), use_container_width=True)
                else:
                    st.info(corr.get("note","No strong correlations found."))

            summary = f"""
FILE: {uploaded.name}
SHAPE: {results['ingestion']['rows']} rows x {results['ingestion']['cols']} columns
COLUMNS: {', '.join(results['ingestion']['columns'])}
MEMORY: {results['ingestion']['size_kb']} KB

SCHEMA:
{json.dumps(results['schema'], indent=2)}

DATA QUALITY (score: {results['quality']['quality_score']}/100):
Issues: {results['quality']['issues'] if results['quality']['issues'] else 'None'}

STATISTICS:
{json.dumps(results['stats'], indent=2)[:3000]}

CORRELATIONS:
{json.dumps(results['correlation'], indent=2)}
"""

            st.markdown("## 🧠 AI Insight Agent")
            llm_ph = st.empty()
            llm_ph.markdown('<div class="agent-card agent-running">🔄 <b>🧠 AI Insight Agent</b><br><small>Sending results to StepFun 3.5 Flash for deep analysis…</small></div>', unsafe_allow_html=True)

            insight_box = st.empty()
            full_text = ""
            try:
                for chunk in agent_llm_insights(summary, api_key):
                    full_text += chunk
                    insight_box.markdown(f'<div class="insight-box">{full_text}▌</div>', unsafe_allow_html=True)
                insight_box.markdown(f'<div class="insight-box">{full_text}</div>', unsafe_allow_html=True)
                llm_ph.markdown('<div class="agent-card agent-done">✅ <b>🧠 AI Insight Agent</b><br><small>Insights generated successfully!</small></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"AI error: {e}")
                st.info("Make sure your OpenRouter API key is valid. Get a free key at openrouter.ai")

            report = f"# Agentic AI Data Analysis Report\n\nFile: {uploaded.name}\n\n## Pipeline Results\n\n```json\n{json.dumps(results, indent=2, default=str)}\n```\n\n## AI Insights\n\n{full_text}"
            st.download_button("📥 Download Full Report", report, file_name="data_insights_report.md", mime="text/markdown")

    except Exception as e:
        st.error(f"Error loading file: {e}")

else:
    st.markdown("""
    ### How it works
    1. **Upload** a CSV, Excel, or JSON file
    2. **5 AI agents** analyse your data in sequence
    3. **StepFun 3.5 Flash** (free via OpenRouter) synthesises everything into deep insights

    👈 Enter your **free** OpenRouter API key in the sidebar to get started.
    """)
    col1, col2, col3 = st.columns(3)
    with col1: st.info("📥 **Ingestion Agent**\nParses file metadata")
    with col2: st.info("🔍 **Schema Agent**\nAnalyses column types")
    with col3: st.info("🧹 **Quality Agent**\nDetects data issues")
    col4, col5 = st.columns(2)
    with col4: st.info("📊 **Stats Agent**\nDescriptive statistics")
    with col5: st.info("🔗 **Correlation Agent**\nFinds relationships")
    st.success("🧠 **AI Insight Agent** — StepFun 3.5 Flash (FREE) synthesises all findings into actionable insights")

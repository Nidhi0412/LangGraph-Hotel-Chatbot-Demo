# --- Imports ---
import os
import sys
import io
import json
import logging
import streamlit as st
from openai import OpenAI
from openai import AsyncOpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from s3_handler_new import S3FileHandler
import time
import tiktoken
import boto3
from botocore.exceptions import ClientError
import re
import asyncio
import collections
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- ✅ Global configs ---
GPT_MODEL = "gpt-4-turbo"
MAX_TOKENS = 6000
MAX_RAW_CHART_LENGTH = 1500  # Characters threshold for summarization
# Do not commit a real API key. Set OPENAI_API_KEY in .env (see .env.example).
# OPENAI_API_KEY = "sk-proj-..."  # local only, never push
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHART_DESCRIPTION_FILE = os.getenv("CHART_DESCRIPTION_FILE", "/home/nidhi/Langchain_tutorials/Practice/GenAI_APP/Innalytics2_chatbot/New_embedding_Ai_innalytics_Charts_Details.xlsx")

# --- Config & Constants ---
HOTEL_CODE = "12939"
USER_ID = "589164"
# HOTEL_CODE="34739"
# USER_ID="2058503"
SELECTED_CHART_IDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 28, 29, 30, 31, 33, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]


# --- Logging Setup ---
logger = logging.getLogger(__name__)
log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Prompt & Helper Functions ---
INSIGHT_PROMPT = """You are expert agentic Business-Intelligence Analyst working inside Innalytics—the leading BI platform for hotels.
Your audience is the hotel's owner, general manager, department heads, and frontline staff.
Your mission: read the data we supply, decide which points matter most, and produce an catagories wise insight-rich report with clear, hotel-friendly language.

INPUT YOU WILL RECEIVE
• One or more data tables or JSON blocks showing KPIs (e.g., RevPAR, ADR, Occupancy, Website Visitors).
• Time context for every metric (e.g., "today", "last 7 days", "M-1", "same day last year").
• Hotel context such as country, star rating, room count, currency, market segment labels, channel labels, etc.
• Optional comparatives (vs. budget, last year, same period last week).

You may use any, all, or just the crucial subset of these data points.
Ignore anything that is irrelevant to the analysis you are preparing.

YOUR TASK

Digest the data. Identify significant patterns, anomalies, and outliers.

Produce a detailed deep-dive insights in the structure below.

Revenue & Profitability

Booking Pace & Forecasting

Market & Channel Segmentation

Operational Status & Housekeeping

Direct-Channel Funnel & Website Performance

For each theme:
• 2–4 headline insights (short paragraphs) in new lines as bullet points
• 2 "What to watch" metrics (leading indicator)
• 3 Actionable recommendations – practical, specific, and prioritized (⭐️⭐️⭐️ = highest impact).

Actionability first. For every weakness or negative variance, propose concrete next steps (e.g., "Deploy 10 % OTA flash sale for 0–3 day lead-time shoulder nights.").

STYLE & FORMATTING RULES
• Write in plain English, hotel-industry tone; avoid jargon outsiders won't know.
• Keep each bullet or paragraph crisp (≤ 25 words). No fluff.
• Use bold for KPI names and ↑/↓ arrows for directionality.
• All numbers must include the unit or currency (%, ₹, ₹k, room-nights, etc.).
• Never reveal internal reasoning—only final, polished insights.
• End with a one-line "Overall verdict" summarising current business health.
Rules:
• Don't generate SWOT format or title (no Strengths, Weaknesses, etc.).
• Only include insights based on actual chart content.
• Use bullet points (•) for each insight.
• Highlight important KPIs using **bold** and directional arrows ↑/↓.

EXAMPLES = 
=====================
📊 FORMAT & STYLE EXAMPLES
=====================

🧾 Use this format for insights under each business theme:

Revenue & Profitability
• 💰 **ADR Jump** ADR rose by ₹800 this week, adding ₹3.2L to overall revenue. Maintain dynamic pricing on weekends.
• 🔴 **Revenue Drop** RevPAR declined by 25% vs last month (₹1,500 → ₹1,125). Optimize rate parity and OTA visibility.

Booking Pace & Forecasting
• 📅 **Pickup Surge** Bookings for next weekend are up 32% vs LY. Reinforce high-demand days with minimum stay rules.
• 🔍 **Lead Time Shift** Average booking lead time decreased to 2.1 days. Consider last-minute rate automation.

Market & Channel Segmentation
• 🌍 **Local Guest Spike** 72% of bookings are from the domestic market this week, led by Gujarat and Maharashtra.
• 📱 **OTA Dominance** 64% of total bookings came via Booking.com. Monitor OTA dependency and promote direct booking discounts.

Operational Status & Housekeeping
• 🧹 **Room Readiness Lag** 8 rooms were unready by check-in time. Strengthen coordination between housekeeping and front desk.
• 🚪 **High Walk-ins** Walk-in guests reached 15% of total traffic yesterday. Prepare buffer rooms and pre-checkin flow.

Direct-Channel Funnel & Website Performance
• 🚀 **Direct Booking Surge** Website-exclusive deals brought in ₹15K this month, a 15% MoM increase.
• 📱 **Mobile Revenue Growth** 85% of city bookings came from mobile, driving ₹3.5L in revenue. Launch mobile-only offers.


BEFORE YOU RESPOND
Think step-by-step:

1. Parse data → 2. Spot key drivers & gaps → 3. Draft insights → 4. Check for clarity/actionability → 5. Output in required structure.
"""

# --- ✅ Initialize OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- ✅ Chart Categories and Analysis Logic ---

# Do not commit real AWS credentials. Set in .env (see .env.example): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET, AWS_BASE_PREFIX.
# s3_handler = S3FileHandler(access_key="...", secret_key="...", bucket_name="...", base_prefix="...")  # local only, never push
_s3_access = os.getenv("AWS_ACCESS_KEY_ID", "")
_s3_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
_s3_bucket = os.getenv("AWS_BUCKET", "hospitality-pvt-stg")
_s3_prefix = os.getenv("AWS_BASE_PREFIX", "day1/innalyticsStore/live/")
s3_handler = S3FileHandler(
    access_key=_s3_access,
    secret_key=_s3_secret,
    bucket_name=_s3_bucket,
    base_prefix=_s3_prefix
) if (_s3_access and _s3_secret) else None


# --- ✅ Token counting function ---
def num_tokens_from_string(string, model="gpt-4"):
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except:
        return len(string.split()) * 1.3  # Rough estimate
    
def show_sidebar_token_info():
    st.sidebar.markdown("### 📊 Token Usage Stats")
    tokens = st.session_state.get("total_tokens", {"input": 0, "output": 0, "total": 0})
    if tokens["total"] > 0:
        st.sidebar.markdown(f"""
        - **📥 Input Tokens**: {tokens['input']:,}
        - **📤 Output Tokens**: {tokens['output']:,}
        - **🔢 Total Tokens**: {tokens['total']:,}
        """)
        input_cost = (tokens["input"] / 1000) * 0.01
        output_cost = (tokens["output"] / 1000) * 0.03
        total_cost = input_cost + output_cost
        st.sidebar.markdown("### 💰 Estimated Cost")
        st.sidebar.markdown(f"""
        - **Input Cost**: ${input_cost:.4f}
        - **Output Cost**: ${output_cost:.4f}
        - **Total Cost**: ${total_cost:.4f} (₹{total_cost*83:.2f})
        """)
    else:
        st.sidebar.info("Run analysis to view token usage.")


def load_chart_metadata():
    try:
        df = pd.read_excel(CHART_DESCRIPTION_FILE)
    except Exception as e:
        logger.warning(f"Chart metadata file not found or invalid ({CHART_DESCRIPTION_FILE}): {e}. Using empty metadata.")
        return {}
    df = df[df['Chart_ID'].apply(lambda x: str(x).strip().isdigit())]
    df['Chart_ID'] = df['Chart_ID'].astype(int).astype(str)

    metadata = {}
    for _, row in df.iterrows():
        # Use the category name directly from Excel
        category = str(row.get("Chart Catagories","uncategorized")).strip()
        metadata[row['Chart_ID']] = {
            "description": row.get("Chart Description", ""),
            "category": category,
            "time_filter": row.get("Filter Time Span", ""),
            "graph_name": row.get("Graph_Name", ""),
            "chart_type": row.get("Chart Type", ""),
            "key_metrics": row.get("Key_Metrics", "").split(",") if row.get("Key_Metrics") else []
        }
    print("Chart descriptions loaded from Excel:")
    for k, v in metadata.items():
        print(f"ID: {k}, Category: {v.get('category')}, Description: {v.get('description')}")
    return metadata


def get_available_charts_with_metadata(hotel_code, user_id, metadata):
    if s3_handler is None:
        return []
    files = s3_handler.list_files_by_hotel_user(hotel_code, user_id, chart_ids=SELECTED_CHART_IDS)
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {
            executor.submit(s3_handler.read_file_content, file): file
            for file in files
        }
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                content = future.result()
                if content:
                    parts = os.path.basename(file).replace('.json', '').split('_')
                    if len(parts) >= 4 and parts[0] == 'live' and parts[1] == 'dataset':
                        chart_id = parts[2]
                        if chart_id in metadata:
                            results.append({"id": chart_id, "content": content, **metadata[chart_id]})
            except Exception as e:
                st.error(f"Error reading file {file}: {e}")
    print("Fetched chart IDs and their categories:")
    for chart in results:
        chart_info = metadata.get(chart['id'], {})
        print(f"Chart ID: {chart['id']}, Category in metadata: {chart_info.get('category')}")
    return results

def build_category_context(category, categorized_charts, info):
    # Use the category name directly as it appears in Excel
    context = f"{category}\n"
    context += f"Time Filter: {info.get('Filter Time Span', '')}\n"
    if 'key_metrics' in info and info['key_metrics']:
        context += f"Key Metrics: {', '.join(info['key_metrics'])}\n"
    category_charts = categorized_charts.get(category, [])
    if category_charts:
        context += "\nChart Data:\n"
        for chart in category_charts:
            chart_id = chart['id']
            content = chart['content']
            description = chart['description']
            time_filter = chart['time_filter']
            content_to_pass = content  # Already summarized if needed
            context += (
                f"\nChart {chart_id}:\n"
                f"Content: {content_to_pass}\n"
                f"Description: {description}\n"
                f"Time Filter: {time_filter}\n"
            )
    return context

    

# --- Streamlit UI ---
st.title("🌟 InnaGenie | Category-wise Hotel Insights")
st.subheader("Deep-dive business intelligence insights by theme")


def render_category_block(title, insights, watch_metric, recommendations):
    st.markdown(f"### 🧾 {title}")
    for point in insights:
        st.markdown(f"• {point}")
    st.markdown(f"**🔎 What to watch:** {watch_metric}")
    st.markdown("**📌 Recommendations:**")
    for reco in recommendations:
        st.markdown(f"⭐️ {reco}")
    st.markdown("---")

def parse_llm_report(report_text):
    # Split by category headers (e.g., "Revenue & Profitability")
    category_pattern = r"^(.*?)\\n(?=\\w|$)"
    categories = re.split(r"(?m)^([A-Za-z &]+)\\n", report_text)[1:]  # [title, body, title, body, ...]
    parsed = []
    for i in range(0, len(categories), 2):
        title = categories[i].strip()
        body = categories[i+1].strip()
        # Extract insights (bullets), what to watch, and recommendations
        insights = re.findall(r"• (.+)", body)
        watch_match = re.search(r"What to watch: (.+)", body)
        watch_metric = watch_match.group(1).strip() if watch_match else ""
        recos = re.findall(r"⭐️ (.+)", body)
        parsed.append({
            "title": title,
            "insights": insights,
            "watch_metric": watch_metric,
            "recommendations": recos
        })
    return parsed

# --- Initialize token tracking ---
if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = {"input": 0, "output": 0, "total": 0}

# --- Theme-specific prompt template ---
THEME_PROMPT_TEMPLATE = """
ONLY FOCUS ON THE THEME: {theme}
Do NOT include other business themes in your response.
Follow all instructions in the main prompt below, but restrict your analysis to this theme only.
"""

# --- Async LLM call for category-wise insights ---
async def get_insight_for_category_async(client, category, context, main_prompt, model, max_tokens=1000, temperature=0.5):
    theme_prompt = THEME_PROMPT_TEMPLATE.format(theme=category)
    full_prompt = theme_prompt + "\n" + main_prompt
    messages = [
        {"role": "system", "content": "You are a hotel business analyst writing clear, impactful insights."},
        {"role": "user", "content": full_prompt},
        {"role": "user", "content": context}
    ]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

async def process_all_categories_async(category_contexts, main_prompt, model, containers, progress_bar):
    async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    total = len(category_contexts)
    completed = 0
    tasks = []
    results = [None] * total

    async def process_one(idx, category, context):
        containers[idx].info(f"Generating insights for {category}...")
        try:
            result = await get_insight_for_category_async(async_client, category, context, main_prompt, model)
            containers[idx].success(f"Insights ready for {category}!")
            containers[idx].markdown(result)
            results[idx] = result
        except Exception as e:
            containers[idx].error(f"Error: {e}")
            results[idx] = None
        nonlocal completed
        completed += 1
        progress_bar.progress(completed / total)

    for idx, (category, context) in enumerate(category_contexts):
        tasks.append(process_one(idx, category, context))
    await asyncio.gather(*tasks)
    progress_bar.empty()
    return results

# --- Async summarization for long charts ---
async def summarize_chart_content_async(client, content, category):
    # Use the same summarization logic as summarize_chart_content, but async
    if category == "Direct-Channel Funnel & Website Performance":
        focus_points = "visitor traffic, search counts, booking conversions, and funnel performance"
    elif category == "Market & Channel Segmentation":
        focus_points = "geographic markets, booking segments, traveler types, device usage, and channel mix"
    elif category == "Operational Status":
        focus_points = "arrivals, departures, housekeeping status, guest traffic, cancellations, and payments"
    elif category == "Booking Pace & Forecasting":
        focus_points = "booking curves, pace, lead times, pickup trends, and forecasting gaps"
    elif category == "Revenue & Profitability":
        focus_points = "ADR, RevPAR, total revenue, profitability, and performance of rate plans and packages"
    else:
        focus_points = "key metrics, trends, and notable patterns"
    prompt = f"""Analyze this {category} chart data and provide ONLY the key insights.\n\nDO NOT include phrases like:\n- \"Here is a concise summary...\"\n- \"This summary provides...\"\n- \"In conclusion...\"\n\nDO:\n- Focus on {focus_points}\n- Include specific numbers and dates\n- Highlight min/max values and anomalies\n- Use direct, factual statements\n- Start immediately with the analysis\n- Be concise but informative\n- FORMAT IN SIMPLE TEXT ONLY\n\nIMPORTANT: Keep the summary under {MAX_RAW_CHART_LENGTH} characters total. Write in a direct style that can be passed to another system.\n\nChart data:\n{content[:2000]}"""
    async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    response = await async_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a data analyst summarizing hotel chart data. Provide ONLY the actual analysis without any introductory phrases or conclusions. Write in a direct, factual style that another AI system can use directly."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )
    summary = response.choices[0].message.content.strip()
    if len(summary) > MAX_RAW_CHART_LENGTH:
        summary = summary[:MAX_RAW_CHART_LENGTH - 3] + "..."
    return f"Summarized data (original exceeded {MAX_RAW_CHART_LENGTH} chars): {summary}"

# --- Efficient async chart processing and insight generation ---
async def process_charts_and_generate_insights(available_charts, chart_descriptions, main_prompt, model, containers, progress_bar):
    # 1. Map metadata to each chart
    charts = []
    for chart in available_charts:
        chart_info = chart_descriptions.get(chart['id'], {})
        chart_with_meta = {**chart, **chart_info}
        charts.append(chart_with_meta)

    # 2. Summarize long charts in parallel
    async def summarize_if_needed(chart):
        if len(chart['content']) > MAX_RAW_CHART_LENGTH:
            chart['content'] = await summarize_chart_content_async(None, chart['content'], chart['category'])
        return chart
    charts = await asyncio.gather(*(summarize_if_needed(chart) for chart in charts))

    # 3. Group by business theme (category)
    categorized_charts = collections.defaultdict(list)
    for chart in charts:
        categorized_charts[chart['category']].append(chart)

    # 4. For each business theme, build context and generate insights in parallel
    category_contexts = []
    for category, charts_in_cat in categorized_charts.items():
        if not charts_in_cat:
            continue  # Skip categories with no charts
        info = {
            'time_filter': charts_in_cat[0].get('time_filter', ''),
            'key_metrics': charts_in_cat[0].get('key_metrics', [])
        }
        context = build_category_context(category, categorized_charts, info)
        category_contexts.append((category, context))

    # 5. Generate insights per theme in parallel (reuse your async insight logic)
    results = await process_all_categories_async(category_contexts, main_prompt, model, containers, progress_bar)
    return results, category_contexts

def safe_markdown(text):
    return text.replace('*', '\\*').replace('_', '\\_')

try:
    st.write("🔄 Loading chart descriptions...")
    chart_descriptions = load_chart_metadata()

    st.success("✅ Chart descriptions loaded.")

    st.write("🔄 Fetching available charts from S3...")
    
    available_charts = get_available_charts_with_metadata(HOTEL_CODE, USER_ID, chart_descriptions)

    st.success(f"✅ Fetched {len(available_charts)} charts")

    if not available_charts:
        st.warning("⚠️ No charts found! Check your S3 credentials or chart IDs.")
    else:
        if st.button("🚀 Generate Insight Report"):
            with st.spinner("Analyzing charts and generating report (category-wise async)..."):
                start_time = time.time()
                # Efficient async chart processing and insight generation
                containers = [st.empty() for _ in range(10)]  # Will be replaced with actual number after grouping
                progress_bar = st.progress(0)
                results = None
                category_contexts = None
                try:
                    # Streamlit Cloud sometimes doesn't allow asyncio.run, so fallback if needed
                    try:
                        results, category_contexts = asyncio.run(process_charts_and_generate_insights(available_charts, chart_descriptions, INSIGHT_PROMPT, GPT_MODEL, containers, progress_bar))
                    except RuntimeError:
                        results, category_contexts = asyncio.get_event_loop().run_until_complete(process_charts_and_generate_insights(available_charts, chart_descriptions, INSIGHT_PROMPT, GPT_MODEL, containers, progress_bar))
                except Exception as e:
                    st.error(f"Async processing failed: {e}")
                end_time = time.time()
                total_runtime = end_time - start_time
                st.sidebar.markdown("### ⏱️ Total Runtime")
                st.sidebar.markdown(f"**{total_runtime:.2f} seconds**")
                # --- Token counting for sidebar ---
                if results and category_contexts:
                    input_tokens = sum(num_tokens_from_string(ctx) for _, ctx in category_contexts)
                    output_tokens = sum(num_tokens_from_string(res) for res in results if res)
                    total_tokens = input_tokens + output_tokens
                    st.session_state["total_tokens"] = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": total_tokens
                    }
                show_sidebar_token_info()
                st.subheader("📋 Final Hotel Insight Report")
                if results and any(results):
                    for result in results:
                        st.markdown(safe_markdown(result))
                else:
                    st.info("No insights generated.")
except Exception as e:
    st.error(f"🚨 Critical error: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("InnaGenie Insight Engine 2.2 | Developed by Innalytics AI Team")

# wrapper for insights generation
async def get_insights_for_langgraph(hotel_code: str, user_id: str, category: str = None) -> dict:
    """
    LangGraph-compatible wrapper for insights generation.
    Bypasses Streamlit UI and runs core async logic.
    """
    try:
        chart_descriptions = load_chart_metadata()
        available_charts = get_available_charts_with_metadata(hotel_code, user_id, chart_descriptions)

        if not available_charts:
            return {
                "insights": [],
                "metadata": {"error": "No charts found"}
            }

        # Dummy UI container replacement
        class DummyContainer:
            def info(self, msg): pass
            def success(self, msg): pass
            def markdown(self, msg): pass
            def error(self, msg): pass
            def progress(self, value): pass
            def empty(self): pass

        containers = [DummyContainer() for _ in range(10)]
        progress_bar = DummyContainer()

        results, category_contexts = await process_charts_and_generate_insights(
            available_charts,
            chart_descriptions,
            INSIGHT_PROMPT,
            GPT_MODEL,
            containers,
            progress_bar
        )

        if category:
            # Filter by category if provided
            category_indices = [i for i, (cat, _) in enumerate(category_contexts) if cat == category]
            if category_indices:
                results = [results[i] for i in category_indices]
                category_contexts = [category_contexts[i] for i in category_indices]

        return {
            "insights": results,
            "metadata": {
                "charts_used": [chart['id'] for chart in available_charts],
                "categories_analyzed": [cat for cat, _ in category_contexts]
            }
        }

    except Exception as e:
        logger.error(f"Error in LangGraph insight function: {e}")
        return {
            "insights": [],
            "metadata": {"error": str(e)}
        }

# streamlit run Insights_new.py

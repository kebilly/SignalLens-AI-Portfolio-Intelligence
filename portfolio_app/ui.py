from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portfolio_app.analysis import (
    aggregate_portfolio_sentiment,
    build_ai_prompt,
    build_deterministic_integrated_report,
    build_deterministic_sentiment_report,
    build_integrated_prompt,
    build_sentiment_prompt,
    calculate_portfolio_risk,
    calculate_risk_profile_score,
    combined_state_label,
    portfolio_table,
    risk_band,
)
from portfolio_app.config import Settings
from portfolio_app.demo import demo_asset_summary, demo_news
from portfolio_app.etf import (
    HoldingsFormatError,
    comparison_table,
    country_exposure,
    overlap_metrics,
    parse_holdings,
)
from portfolio_app.ptp import PTPFormatError, parse_ptp_pdf, screen_portfolio
from portfolio_app.quant import quantitative_metrics, risk_contributions
from portfolio_app.reports import portfolio_report_pdf
from portfolio_app.risk_assessment import (
    DIMENSION_META,
    DIMENSION_WEIGHTS,
    RISK_QUESTIONS,
    calculate_assessment,
    questions_for,
)
from portfolio_app.sentiment import (
    SENTIMENT_COLORS,
    SENTIMENT_ORDER,
    calculate_sentiment_stats,
    daily_sentiment,
    sentiment_label_zh,
    sentiment_price_correlation,
    source_distribution,
    topic_distribution,
)
from portfolio_app.sentiment_provider import AlphaVantageClient
from portfolio_app.services import (
    AIAnalysisResult,
    ExternalServiceError,
    FMPClient,
    OpenAIClient,
    PerplexityClient,
)
from portfolio_app.validation import normalize_symbol, validate_portfolio

logger = logging.getLogger(__name__)

AI_OPTIONS = [
    "自動選擇（建議）",
    "OpenAI · gpt-5-mini",
    "Perplexity · sonar",
    "Perplexity · sonar-reasoning-pro",
    "不使用 AI",
]

NAVIGATION_LABELS = {
    "總覽": "總覽",
    "風險評估": "風險評估",
    "整合分析": "整合分析",
    "投資組合風險": "投資組合風險",
    "新聞情緒研究": "新聞情緒研究",
    "ETF 曝險比較": "ETF 曝險比較",
    "產品警示": "產品警示",
    "研究報告": "研究報告",
}


def _ai_meta(result: AIAnalysisResult) -> dict[str, Any]:
    return {
        "provider": result.provider,
        "model": result.model,
        "completed": result.completed,
        "continued": result.continued,
        "finish_reason": result.finish_reason,
    }


def _render_ai_completion_status(meta: dict[str, Any] | None) -> None:
    if not meta:
        return
    if not meta.get("completed", True):
        st.warning("AI 服務自動續寫後仍達輸出上限；本頁已保留所有收到的內容，但報告可能仍不完整。")
    elif meta.get("continued"):
        st.info("本報告第一次生成時達到輸出上限，系統已自動續寫並合併完整內容。")


@st.cache_resource(show_spinner=False)
def services(settings: Settings) -> tuple[FMPClient, PerplexityClient]:
    return FMPClient(settings), PerplexityClient(settings)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_asset(symbol: str, include_beta: bool, settings: Settings) -> dict[str, Any]:
    fmp, _ = services(settings)
    return fmp.asset_summary(symbol, include_beta)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(symbol: str, settings: Settings) -> list[dict[str, Any]]:
    fmp, _ = services(settings)
    return fmp.price_history(symbol)


@st.cache_data(ttl=7200, show_spinner=False)
def cached_news(symbol: str, limit: int, settings: Settings) -> list[dict[str, Any]]:
    return AlphaVantageClient(settings.alpha_vantage_api_key, settings.request_timeout_seconds).news_sentiment(symbol, limit)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_alpha_prices(symbol: str, settings: Settings) -> list[dict[str, Any]]:
    return AlphaVantageClient(
        settings.alpha_vantage_api_key, settings.request_timeout_seconds
    ).daily_prices(symbol)


def ai_service(settings: Settings, selection: str, task: str):
    """Choose a model explicitly or by task, returning client, label and reason."""
    if selection == "不使用 AI":
        raise ExternalServiceError("目前已選擇不使用 AI。", "AI disabled by user")
    if selection == "OpenAI · gpt-5-mini":
        if not settings.openai_api_key:
            raise ExternalServiceError("尚未設定 OPENAI_API_KEY。", "OpenAI selected without key")
        return OpenAIClient(settings, "gpt-5-mini"), "OpenAI · gpt-5-mini", "適合結構化中文報告，速度與內容完整度較平衡。"
    if selection == "Perplexity · sonar":
        if not settings.perplexity_api_key:
            raise ExternalServiceError("尚未設定 PERPLEXITY_API_KEY。", "Perplexity selected without key")
        return PerplexityClient(settings, "sonar"), "Perplexity · sonar", "適合快速摘要，回應較快、篇幅通常較精簡。"
    if selection == "Perplexity · sonar-reasoning-pro":
        if not settings.perplexity_api_key:
            raise ExternalServiceError("尚未設定 PERPLEXITY_API_KEY。", "Perplexity selected without key")
        return PerplexityClient(settings, "sonar-reasoning-pro"), "Perplexity · sonar-reasoning-pro", "適合多持倉與產業風險的深度分析，但等待時間較長。"
    if task == "portfolio" and settings.perplexity_api_key:
        return PerplexityClient(settings, "sonar-reasoning-pro"), "Perplexity · sonar-reasoning-pro", "自動選擇：投資組合風險需要較深入的推理。"
    if settings.openai_api_key:
        return OpenAIClient(settings, "gpt-5-mini"), "OpenAI · gpt-5-mini", "自動選擇：新聞與整合報告適合穩定的結構化輸出。"
    if settings.perplexity_api_key:
        model = "sonar-reasoning-pro" if task == "integrated" else "sonar"
        reason = "自動備援：未設定 OpenAI，改用 Perplexity 深度模型。" if task == "integrated" else "自動備援：未設定 OpenAI，改用 Perplexity 快速模型。"
        return PerplexityClient(settings, model), f"Perplexity · {model}", reason
    raise ExternalServiceError("尚未設定可用的 AI 服務。", "No AI provider configured")


def initialize_state() -> None:
    st.session_state.setdefault("portfolio", [
        {"symbol": "AAPL", "allocation": 30},
        {"symbol": "MSFT", "allocation": 30},
        {"symbol": "JNJ", "allocation": 20},
    ])
    st.session_state.setdefault("cash_position", 20)
    st.session_state.setdefault("risk_questionnaire_answers", {})
    st.session_state.setdefault("risk_assessment_step", 0)


def render_application(settings: Settings | None) -> None:
    _inject_theme()
    navigation, profile = sidebar(settings)
    st.markdown('<div class="hero"><div class="hero-kicker">SIGNALLENS</div><h1>投資組合智慧分析平台</h1><p>整合價格風險、持倉新聞情緒與曝險研究，讓數據比結論更透明。</p></div>', unsafe_allow_html=True)
    pages = {
        "總覽": lambda: overview(settings),
        "風險評估": risk_assessment_page,
        "整合分析": lambda: integrated_analysis_page(settings, profile),
        "投資組合風險": lambda: portfolio_risk_page(settings, profile),
        "新聞情緒研究": lambda: sentiment_page(settings, profile),
        "ETF 曝險比較": etf_page,
        "產品警示": ptp_page,
        "研究報告": reports_page,
    }
    pages[navigation]()


def _inject_theme() -> None:
    st.markdown("""
    <style>
    :root { --ink:#172b3a; --navy:#15344f; --blue:#2e6680; --paper:#f7f8fa; --gold:#d5a247; }
    .stApp { background:linear-gradient(180deg,#fafbfc 0%,#f4f6f8 100%); color:var(--ink); }
    .block-container { max-width:1280px; padding-top:1.6rem; padding-bottom:2.5rem; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#102a43 0%,#183f5a 100%); }
    [data-testid="stSidebar"] * { color:#eef8f5; }
    [data-testid="stSidebarUserContent"] { padding:.65rem 1rem 1.5rem !important; margin-top:-3rem; }
    .sidebar-brand { display:flex; align-items:center; gap:.72rem; padding:.15rem .38rem .8rem; }
    .sidebar-brand-mark { width:2.15rem; height:2.15rem; display:grid; place-items:center; flex:0 0 auto;
      border:1px solid rgba(164,220,231,.5); border-radius:10px; background:rgba(89,172,194,.15);
      box-shadow:inset 0 1px 0 rgba(255,255,255,.13); color:#b9e5eb; font-size:1.18rem; }
    .sidebar-brand-name { color:#f6fbfc; font-family:Georgia,'Times New Roman',serif; font-size:1.42rem;
      line-height:1.05; font-weight:700; letter-spacing:.01em; }
    .sidebar-brand-subtitle { margin-top:.25rem; color:#9eb9c7; font-size:.75rem; letter-spacing:.06em; }
    .sidebar-section-label { margin:.3rem .45rem .45rem; color:#83a6b7; font-size:.69rem;
      font-weight:700; letter-spacing:.13em; }
    .risk-score-card { margin:.45rem 0 .8rem; padding:.82rem .9rem; border:1px solid rgba(133,205,219,.24);
      border-radius:12px; background:linear-gradient(135deg,rgba(42,112,137,.48),rgba(22,67,91,.72));
      box-shadow:0 7px 18px rgba(3,20,32,.16),inset 0 1px 0 rgba(255,255,255,.08); }
    .risk-score-label { color:#b9d3dc; font-size:.73rem; font-weight:650; letter-spacing:.04em; }
    .risk-score-row { display:flex; align-items:baseline; justify-content:space-between; gap:.5rem; margin-top:.18rem; }
    .risk-score-value { color:#fff; font-size:1.72rem; line-height:1; font-weight:800; letter-spacing:-.02em;
      text-shadow:0 1px 10px rgba(0,0,0,.2); }
    .risk-score-total { color:#bcd3dc; font-size:.82rem; font-weight:600; }
    .risk-score-type { padding:.22rem .48rem; border-radius:999px; background:rgba(115,203,219,.16);
      color:#dff8fb; font-size:.72rem; font-weight:700; border:1px solid rgba(137,218,230,.18); }
    .assessment-step { margin:.15rem 0 1rem; padding:.9rem 1rem; border:1px solid #dce7ec; border-radius:13px;
      background:linear-gradient(135deg,#fff,#f4f8fa); box-shadow:0 5px 16px rgba(23,52,79,.05); }
    .assessment-step strong { color:#1e536b; font-size:1.03rem; }
    .assessment-step span { display:block; margin-top:.2rem; color:#667f8b; font-size:.84rem; }
    .assessment-result { padding:1.15rem 1.25rem; border-radius:15px; color:#fff;
      background:linear-gradient(125deg,#173851,#287087); box-shadow:0 10px 26px rgba(23,56,81,.16); }
    .assessment-result-label { color:#b9d7df; font-size:.75rem; letter-spacing:.08em; }
    .assessment-result-score { margin-top:.15rem; font-size:2.25rem; line-height:1.1; font-weight:800; }
    .assessment-result-score span { font-size:.95rem; color:#c8e0e6; font-weight:600; }
    .st-key-primary_navigation [role="radiogroup"] { gap:.28rem; }
    .st-key-primary_navigation label[data-testid="stRadioOption"] { width:100%; min-height:2.55rem; padding:.62rem .78rem;
      border:1px solid transparent; border-radius:10px; transition:background .15s ease,border-color .15s ease,transform .15s ease; }
    .st-key-primary_navigation label[data-testid="stRadioOption"]:hover { background:rgba(255,255,255,.07);
      border-color:rgba(172,218,228,.12); transform:translateX(2px); }
    .st-key-primary_navigation label[data-testid="stRadioOption"] > div > div > div:first-child { display:none; }
    .st-key-primary_navigation label[data-testid="stRadioOption"] p { margin:0; color:#d8e7ec; font-size:.92rem;
      line-height:1.2; font-weight:520; }
    .st-key-primary_navigation label[data-testid="stRadioOption"][data-selected="true"] { background:linear-gradient(90deg,rgba(48,135,158,.38),rgba(68,145,162,.16));
      border-color:rgba(134,205,219,.25); box-shadow:inset 3px 0 0 #67c1d2,0 5px 16px rgba(4,19,31,.12); }
    .st-key-primary_navigation label[data-testid="stRadioOption"][data-selected="true"] p { color:#fff; font-weight:700; }
    .st-key-primary_navigation { margin-bottom:.35rem; }
    [data-testid="stSidebar"] .stButton > button { background:rgba(255,255,255,.075); color:#eef8f5 !important;
      border:1px solid rgba(160,211,222,.28); box-shadow:none; }
    [data-testid="stSidebar"] .stButton > button:hover { background:rgba(91,174,194,.18);
      border-color:rgba(158,221,232,.48); }
    [data-testid="stSidebar"] .stButton > button p { color:#eef8f5 !important; font-weight:700; }
    .stApp [data-testid="stSidebar"] input { color:#183941 !important; -webkit-text-fill-color:#183941 !important; opacity:1 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background:#ffffff; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color:#183941 !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] { background:#ffffff; }
    [data-testid="stSidebar"] [data-testid="stExpander"] * { color:#183941 !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#9bd8e7; }
    [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.18); }
    .hero { padding:1.45rem 1.8rem; margin:0 0 1rem; border-radius:18px;
      background:radial-gradient(circle at 88% 15%,rgba(101,177,199,.25),transparent 30%),linear-gradient(125deg,#173851,#285f73);
      box-shadow:0 12px 28px rgba(25,52,76,.14); color:white; }
    .hero h1 { margin:.12rem 0 .3rem; font-size:1.9rem; line-height:1.25; letter-spacing:.01em; color:white; }
    .hero p { margin:0; color:#dbeaf0; font-size:.96rem; }
    .hero-kicker { color:#e5bc68; font-size:.68rem; letter-spacing:.22em; font-weight:700; }
    [data-testid="stMetric"] { background:white; border:1px solid #e1e7ec; border-radius:13px; padding:.82rem .9rem;
      box-shadow:0 4px 14px rgba(23,52,79,.055); }
    [data-testid="stMetricLabel"] { color:#587078; }
    [data-testid="stMetricValue"] { color:#1f5870; font-size:1.72rem; }
    .stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:650; border:1px solid #6e9db1; }
    .stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {
      background:linear-gradient(90deg,#245c78,#337c91); color:white; border:0; box-shadow:0 6px 14px rgba(36,92,120,.19); }
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { background:white; border:1px solid #e1e7ec; border-radius:12px; padding:.25rem; }
    [data-testid="stExpander"] { background:white; border:1px solid #e1e7ec; border-radius:11px; }
    h1, h2, h3 { color:#193d55; letter-spacing:.01em; }
    h2 { font-size:1.55rem !important; margin-top:.6rem !important; margin-bottom:.4rem !important; }
    h3 { font-size:1.18rem !important; margin-top:.75rem !important; margin-bottom:.35rem !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { line-height:1.72; }
    [data-testid="stTabs"] [role="tablist"] { gap:.3rem; }
    [data-testid="stTabs"] [role="tab"] { padding:.55rem .85rem; }
    </style>
    """, unsafe_allow_html=True)


def sidebar(settings: Settings | None) -> tuple[str, dict[str, Any]]:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><div class="sidebar-brand-mark">◇</div>'
            '<div><div class="sidebar-brand-name">SignalLens</div>'
            '<div class="sidebar-brand-subtitle">投資研究工作台</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-section-label">功能導覽</div>', unsafe_allow_html=True)
        navigation = st.radio(
            "功能導覽",
            list(NAVIGATION_LABELS),
            format_func=NAVIGATION_LABELS.get,
            label_visibility="collapsed",
            key="primary_navigation",
        )
        st.divider()
        st.markdown("### 投資人風險屬性")
        applied_assessment = st.session_state.get("risk_assessment_applied")
        risk_types = ["保守型", "穩健型", "平衡型", "成長型", "積極型"]
        if applied_assessment:
            risk_profile = st.selectbox(
                "風險類型", risk_types, index=risk_types.index(applied_assessment["risk_type"]), disabled=True
            )
        else:
            risk_profile = st.selectbox("風險類型", risk_types, index=2)
        capital = st.number_input("可投資資金（美元）", 1_000, value=100_000, step=1_000)
        if applied_assessment:
            scores = applied_assessment["dimension_scores"]
            risk_score = applied_assessment["score"]
            with st.expander("查看正式問卷四維分數"):
                for key in DIMENSION_WEIGHTS:
                    st.progress(int(scores[key]), text=f"{DIMENSION_META[key][0]}　{scores[key]:.0f}")
            st.button("重新進行風險評估", width="stretch", on_click=_begin_new_risk_assessment)
        else:
            with st.expander("快速設定（尚未完成正式問卷）"):
                scores = {
                    "financial_status": st.slider("財務能力", 0, 100, 50),
                    "investment_experience": st.slider("投資經驗", 0, 100, 50),
                    "investment_goal": st.slider("目標明確度", 0, 100, 50),
                    "risk_tolerance": st.slider("損失承受度", 0, 100, 50),
                }
            risk_score = calculate_risk_profile_score(scores)
            st.button("進行完整風險評估", width="stretch", on_click=_open_risk_assessment)
        st.markdown(
            f'<div class="risk-score-card"><div class="risk-score-label">風險承受分數</div>'
            f'<div class="risk-score-row"><div><span class="risk-score-value">{risk_score:.0f}</span>'
            f'<span class="risk-score-total"> / 100</span></div>'
            f'<span class="risk-score-type">{risk_profile}</span></div></div>',
            unsafe_allow_html=True,
        )
        include_beta = st.toggle("納入 Beta 分析", value=False)
        st.markdown("### AI 分析設定")
        ai_selection = st.selectbox("模型選擇", AI_OPTIONS, index=0)
        if ai_selection == "自動選擇（建議）":
            st.caption("風險分析偏向深度模型；新聞與整合報告偏向穩定的結構化模型。")
        elif ai_selection == "不使用 AI":
            st.caption("只顯示 Python 計算結果與固定教育性摘要，不產生 API 費用。")
        else:
            st.caption("手動模式會固定使用指定模型；若未設定對應 API Key，系統會顯示提示。")
        st.divider()
        if settings and (settings.portfolio_enabled or settings.sentiment_enabled or settings.ai_enabled):
            st.success("即時資料服務已設定")
            if not settings.sentiment_enabled:
                st.info("尚未設定 Alpha Vantage，可使用離線展示模式。")
        else:
            st.warning("尚未設定即時服務，離線展示功能仍可使用。")
        st.caption("僅供研究與教育，不構成投資或財務建議。")
    return navigation, {"risk_profile": risk_profile, "capital": capital, "scores": scores, "risk_score": risk_score, "include_beta": include_beta, "ai_selection": ai_selection}


def _open_risk_assessment() -> None:
    st.session_state.primary_navigation = "風險評估"


def _restart_risk_assessment() -> None:
    st.session_state.risk_questionnaire_answers = {}
    st.session_state.risk_assessment_step = 0
    st.session_state.pop("risk_assessment_result", None)
    for question in RISK_QUESTIONS:
        st.session_state.pop(f"assessment_{question['id']}", None)


def _begin_new_risk_assessment() -> None:
    _restart_risk_assessment()
    _open_risk_assessment()


def risk_assessment_page() -> None:
    st.header("投資人風險評估", divider="rainbow")
    st.caption("以 20 題問卷分別衡量財務能力、投資經驗、投資目標與心理承受度；分數由固定規則計算，不使用 AI。")
    result = st.session_state.get("risk_assessment_result")
    if result:
        _render_assessment_result(result)
        return

    dimensions = list(DIMENSION_WEIGHTS)
    step = min(int(st.session_state.risk_assessment_step), len(dimensions) - 1)
    dimension = dimensions[step]
    title, description = DIMENSION_META[dimension]
    st.progress((step + 1) / len(dimensions), text=f"步驟 {step + 1} / {len(dimensions)}")
    st.markdown(
        f'<div class="assessment-step"><strong>{title}</strong><span>{description}</span></div>',
        unsafe_allow_html=True,
    )
    stored_answers = st.session_state.risk_questionnaire_answers
    current_answers: dict[str, Any] = {}
    with st.form(f"risk_assessment_form_{dimension}", border=False):
        for number, question in enumerate(questions_for(dimension), 1):
            question_id = question["id"]
            label = f"{number}. {question['question']}"
            stored = stored_answers.get(question_id)
            if question.get("kind") == "multiselect":
                current_answers[question_id] = st.multiselect(
                    label,
                    question["options"],
                    default=stored or [],
                    key=f"assessment_{question_id}",
                )
            else:
                options = list(question["options"])
                index = options.index(stored) if stored in options else None
                current_answers[question_id] = st.radio(
                    label,
                    options,
                    index=index,
                    key=f"assessment_{question_id}",
                )
            st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)
        previous_col, next_col = st.columns([1, 2])
        previous = previous_col.form_submit_button(
            "上一步", disabled=step == 0, width="stretch"
        )
        next_label = "計算評估結果" if step == len(dimensions) - 1 else "儲存並繼續"
        next_step = next_col.form_submit_button(next_label, type="primary", width="stretch")

    if previous:
        st.session_state.risk_questionnaire_answers.update(
            {key: value for key, value in current_answers.items() if value}
        )
        st.session_state.risk_assessment_step = max(0, step - 1)
        st.rerun()
    if next_step:
        missing = [
            question["question"]
            for question in questions_for(dimension)
            if not current_answers.get(question["id"])
        ]
        if missing:
            st.error("請完成本頁所有題目後再繼續。")
            return
        known_tools = current_answers.get("known_instruments", [])
        if "目前都不熟悉" in known_tools and len(known_tools) > 1:
            st.error("「目前都不熟悉」不能與其他投資工具同時選擇，請擇一。")
            return
        st.session_state.risk_questionnaire_answers.update(current_answers)
        if step < len(dimensions) - 1:
            st.session_state.risk_assessment_step = step + 1
            st.rerun()
        assessment = calculate_assessment(st.session_state.risk_questionnaire_answers)
        st.session_state.risk_assessment_result = assessment
        st.session_state.risk_assessment_step = len(dimensions)
        st.rerun()


def _render_assessment_result(result: dict[str, Any]) -> None:
    score = float(result["score"])
    left, right = st.columns([2, 3])
    with left:
        st.markdown(
            f'<div class="assessment-result"><div class="assessment-result-label">綜合風險承受分數</div>'
            f'<div class="assessment-result-score">{score:.0f}<span> / 100</span></div>'
            f'<div style="margin-top:.55rem;font-weight:700">{result["risk_type"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("此分類用於調整研究內容的呈現方式，不代表適合購買特定金融商品。")
    with right:
        dimension_frame = pd.DataFrame(
            {
                "評估維度": [DIMENSION_META[key][0] for key in DIMENSION_WEIGHTS],
                "分數": [result["dimension_scores"][key] for key in DIMENSION_WEIGHTS],
            }
        )
        figure = px.bar(
            dimension_frame,
            x="分數",
            y="評估維度",
            orientation="h",
            range_x=[0, 100],
            color="分數",
            color_continuous_scale=["#9ec7d2", "#2c7189"],
            text_auto=".0f",
        )
        figure.update_layout(height=285, coloraxis_showscale=False, margin={"t": 15, "b": 20})
        st.plotly_chart(figure, width="stretch")
    if result.get("alignment_warning"):
        st.warning("四個維度的分數差距較大，代表客觀財務能力、投資目標或心理承受度可能不一致；進行分析時應分開解讀。")
    applied = st.session_state.get("risk_assessment_applied")
    action_col, reset_col = st.columns([2, 1])
    if action_col.button(
        "套用至投資分析",
        type="primary",
        width="stretch",
        disabled=bool(applied and applied.get("answers") == result.get("answers")),
    ):
        st.session_state.risk_assessment_applied = result
        st.success("已套用。投資組合風險、整合分析與研究報告將使用這份評估結果。")
        st.rerun()
    if reset_col.button("重新填寫", width="stretch"):
        _restart_risk_assessment()
        st.rerun()
    if applied and applied.get("answers") == result.get("answers"):
        st.success("目前投資分析正在使用這份正式問卷結果。")
    with st.expander("查看計分方法"):
        st.write("財務狀況 25%｜投資經驗 20%｜投資目標 20%｜風險心理承受度 35%。")
        st.write("各題先依固定選項換算為 0–100 分，再計算各維度平均與加權總分；AI 不參與評分。")
    st.warning("本問卷僅供教育與研究，不構成適合度審查、投資建議或財務建議。")


def overview(settings: Settings | None) -> None:
    st.header("研究工作台總覽", divider="rainbow")
    latest = st.session_state.get("latest_analysis")
    sentiment = st.session_state.get("latest_sentiment")
    cards = st.columns(4)
    cards[0].metric("股票持倉數", len(st.session_state.portfolio))
    cards[1].metric("最近風險分數", f"{latest['portfolio_risk']:.1f}/100" if latest else "尚未分析")
    cards[2].metric("最近新聞情緒", f"{sentiment['stats']['average_ticker_score']:+.3f}" if sentiment else "尚未分析")
    cards[3].metric("資料模式", "即時服務可用" if settings and (settings.portfolio_enabled or settings.sentiment_enabled) else "離線展示")
    st.markdown("""
### 平台核心能力

- 可追溯的風險指標，而非不透明的 AI 評分
- 伺服器端憑證管理與具容錯能力的 API 用戶端
- 整合投資組合、ETF、新聞情緒與產品警示工作流程
- 將確定性數值與 AI 文字解讀清楚分開
- 不需要 API Key 也能完整操作的離線展示模式
""")
    st.info("價格風險與新聞情緒是兩個獨立維度；本平台不會把情緒轉換成買進或賣出分數。")


def integrated_analysis_page(settings: Settings | None, profile: dict[str, Any]) -> None:
    st.header("整合式投資組合分析", divider="rainbow")
    st.caption("在同一流程中分析價格風險與各持倉新聞情緒，並維持兩種量尺的獨立性。")
    portfolio = portfolio_editor()
    mode_column, limit_column = st.columns(2)
    live_ready = bool(settings and settings.portfolio_enabled and settings.sentiment_enabled)
    mode = mode_column.radio(
        "資料模式", ["即時服務", "離線展示"], index=0 if live_ready else 1, horizontal=True
    )
    article_limit = limit_column.slider("每檔持倉新聞數", 1, 50, 10)
    if not st.button("開始整合分析", type="primary", width="stretch"):
        if st.session_state.get("latest_integrated"):
            render_integrated(st.session_state.latest_integrated)
        return
    errors = validate_portfolio(portfolio)
    if errors:
        st.error("請先修正股票代碼與資產配置警示。")
        return
    if mode == "即時服務" and not live_ready:
        st.error("即時整合分析需要設定 FMP_API_KEY 與 ALPHA_VANTAGE_API_KEY。")
        return
    progress = st.progress(0)
    try:
        summaries = []
        sentiment_results = {}
        total_steps = max(len(portfolio) * 2, 1)
        step = 0
        for asset in portfolio:
            if asset["symbol"] == "CASH":
                summary = FMPClient.cash_summary()
            elif mode == "離線展示":
                summary = demo_asset_summary(asset["symbol"])
            else:
                assert settings is not None
                summary = cached_asset(asset["symbol"], profile["include_beta"], settings)
            summary["allocation"] = asset["allocation"]
            summaries.append(summary)
            step += 1
            progress.progress(min(step / total_steps, 0.9))
        for asset in portfolio:
            symbol = asset["symbol"]
            if symbol == "CASH":
                continue
            if mode == "離線展示":
                news = demo_news(symbol)
            else:
                assert settings is not None
                news = cached_news(symbol, article_limit, settings)
            stats = calculate_sentiment_stats(news)
            if mode == "離線展示":
                prices = []
            else:
                assert settings is not None
                prices = cached_alpha_prices(symbol, settings)
            sentiment_results[symbol] = {
                "symbol": symbol,
                "mode": mode,
                "news": news,
                "stats": stats,
                "prices": prices,
                "price_context": _price_context(prices),
                "narrative": "已納入整合研究報告。",
            }
            step += 1
            progress.progress(min(step / total_steps, 0.9))
        risk_result = {
            "profile": profile,
            "assets": summaries,
            "portfolio_risk": calculate_portfolio_risk(summaries),
            "quant": quantitative_metrics(summaries),
            "contributions": risk_contributions(summaries),
        }
        narrative = build_deterministic_integrated_report(risk_result, sentiment_results)
        provider_name = "離線固定摘要" if mode == "離線展示" else "未使用 AI"
        provider_reason = "離線模式不呼叫外部 AI。" if mode == "離線展示" else "目前設定只顯示 Python 計算結果與固定教育性摘要。"
        ai_meta = None
        if mode == "即時服務" and settings and settings.ai_enabled and profile["ai_selection"] != "不使用 AI":
            system, prompt = build_integrated_prompt(risk_result, sentiment_results)
            ai, provider_name, provider_reason = ai_service(settings, profile["ai_selection"], "integrated")
            with st.spinner(f"正在使用 {provider_name} 產生整合報告……"):
                ai_result = ai.analyze(system, prompt)
                narrative = ai_result.text
                ai_meta = _ai_meta(ai_result)
        risk_result["report"] = narrative
        risk_result["ai_meta"] = ai_meta
        result = {
            "risk": risk_result,
            "sentiments": sentiment_results,
            "portfolio_sentiment": aggregate_portfolio_sentiment(summaries, sentiment_results),
            "mode": mode,
            "provider": provider_name,
            "provider_reason": provider_reason,
        }
        st.session_state.latest_analysis = risk_result
        st.session_state.latest_integrated = result
        progress.progress(1.0)
        render_integrated(result)
    except ExternalServiceError as exc:
        logger.warning(exc.log_message)
        st.error(exc.public_message)
    except Exception:
        logger.exception("Integrated analysis failed")
        st.error("整合分析未能完成；系統沒有顯示或洩漏任何 API 憑證。")
    finally:
        progress.empty()


def render_integrated(result: dict[str, Any]) -> None:
    risk = result["risk"]
    tone = result["portfolio_sentiment"]
    metrics = st.columns(5)
    metrics[0].metric("組合風險", f"{risk['portfolio_risk']:.1f}/100")
    metrics[1].metric("年化波動率", f"{risk['quant']['annualized_volatility']:.1f}%" if risk["quant"]["annualized_volatility"] is not None else "無資料")
    metrics[2].metric("加權新聞情緒", f"{tone['weighted_ticker_score']:+.3f}")
    metrics[3].metric("新聞覆蓋率", f"{tone['covered_weight']:.0f}%")
    metrics[4].metric("新聞篇數", tone["article_count"])
    st.info(combined_state_label(risk["portfolio_risk"], tone["weighted_ticker_score"]))
    st.caption(f"本次模型：{result['provider']}｜{result.get('provider_reason', '')}")
    report_tab, risk_tab, holdings_tab = st.tabs(["整合研究報告", "風險分析", "持倉情緒"])
    with report_tab, st.container(border=True):
        _render_ai_completion_status(risk.get("ai_meta"))
        st.markdown(risk["report"])
    with risk_tab:
        render_analysis(risk)
    with holdings_tab:
        rows = []
        for symbol, sentiment in result["sentiments"].items():
            stats = sentiment["stats"]
            rows.append({
                "股票代碼": symbol,
                "新聞篇數": stats["count"],
                "股票情緒": stats["average_ticker_score"],
                "文章情緒": stats["average_overall_score"],
                "相關性": stats["average_relevance"],
                "主要情緒": sentiment_label_zh(stats["dominant_label"]),
            })
        frame = pd.DataFrame(rows)
        st.dataframe(frame, hide_index=True, width="stretch")
        if not frame.empty:
            st.plotly_chart(
                px.bar(frame, x="股票代碼", y=["股票情緒", "文章情緒"], barmode="group",
                       title="各持倉新聞情緒比較", labels={"value": "情緒分數", "variable": "情緒類型"}),
                width="stretch",
            )


def portfolio_editor() -> list[dict[str, Any]]:
    cash = st.number_input("現金配置（%）", 0, 100, int(st.session_state.cash_position), 1, key="cash_editor")
    st.session_state.cash_position = cash
    add, remove, _ = st.columns([1, 1, 5])
    if add.button("新增資產", width="stretch"):
        st.session_state.portfolio.append({"symbol": "", "allocation": 0})
        st.rerun()
    if remove.button("移除資產", disabled=len(st.session_state.portfolio) <= 1, width="stretch"):
        st.session_state.portfolio.pop()
        st.rerun()
    rows = []
    for index, item in enumerate(st.session_state.portfolio):
        symbol_col, allocation_col = st.columns([2, 1])
        symbol = normalize_symbol(symbol_col.text_input(f"股票代碼 {index + 1}", item["symbol"], key=f"portfolio_symbol_{index}"))
        allocation = allocation_col.number_input(f"配置比例 {index + 1}（%）", 0, 100, int(item["allocation"]), 1, key=f"portfolio_weight_{index}")
        st.session_state.portfolio[index] = {"symbol": symbol, "allocation": allocation}
        if symbol or allocation:
            rows.append({"symbol": symbol, "allocation": float(allocation)})
    if cash:
        rows.append({"symbol": "CASH", "allocation": float(cash)})
    errors = validate_portfolio(rows)
    for error in errors:
        st.warning(error)
    if not errors:
        st.success("資產配置總和為 100%")
    if rows:
        figure = px.pie(pd.DataFrame(rows), names="symbol", values="allocation", hole=0.45)
        figure.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(figure, width="stretch")
    return rows


def portfolio_risk_page(settings: Settings | None, profile: dict[str, Any]) -> None:
    st.header("投資組合風險分析", divider="rainbow")
    st.caption("以歷史資料檢視風險、集中度與分散效果，不代表未來預測。")
    portfolio = portfolio_editor()
    if not st.button("開始風險分析", type="primary", width="stretch"):
        show_saved_analysis()
        return
    errors = validate_portfolio(portfolio)
    if errors:
        st.error("請先修正股票代碼與資產配置警示。")
        return
    if not settings or not settings.portfolio_enabled:
        st.error("即時投資組合風險分析需要設定 FMP_API_KEY。")
        return
    progress = st.progress(0)
    summaries = []
    try:
        for index, asset in enumerate(portfolio):
            summary = FMPClient.cash_summary() if asset["symbol"] == "CASH" else cached_asset(asset["symbol"], profile["include_beta"], settings)
            summary["allocation"] = asset["allocation"]
            summaries.append(summary)
            progress.progress((index + 1) / (len(portfolio) + 1))
        heuristic_score = calculate_portfolio_risk(summaries)
        quant = quantitative_metrics(summaries)
        contributions = risk_contributions(summaries)
        narrative = "尚未設定 AI 服務；確定性風險指標仍可正常使用。"
        ai_meta = None
        if settings.ai_enabled:
            system, prompt = build_ai_prompt(profile, summaries, heuristic_score)
            ai, provider_name, provider_reason = ai_service(settings, profile["ai_selection"], "portfolio")
            with st.spinner(f"正在使用 {provider_name} 產生風險教育分析……"):
                ai_result = ai.analyze(system, prompt)
                narrative = ai_result.text
                ai_meta = _ai_meta(ai_result)
        result = {"profile": profile, "assets": summaries, "portfolio_risk": heuristic_score, "quant": quant, "contributions": contributions, "report": narrative, "ai_meta": ai_meta}
        st.session_state.latest_analysis = result
        progress.progress(1.0)
        render_analysis(result)
    except ExternalServiceError as exc:
        logger.warning(exc.log_message)
        st.error(exc.public_message)
    except Exception:
        logger.exception("Portfolio analysis failed")
        st.error("分析未能完成；系統沒有顯示或洩漏任何 API 憑證。")
    finally:
        progress.empty()


def show_saved_analysis() -> None:
    result = st.session_state.get("latest_analysis")
    if result:
        st.info("目前顯示本次工作階段最近完成的分析。")
        render_analysis(result)


def render_analysis(result: dict[str, Any]) -> None:
    score, quant = result["portfolio_risk"], result["quant"]
    columns = st.columns(5)
    columns[0].metric("教育性風險分數", f"{score:.1f}/100", risk_band(score))
    columns[1].metric("年化波動率", f"{quant['annualized_volatility']:.1f}%" if quant["annualized_volatility"] is not None else "無資料")
    columns[2].metric("單日 VaR 95%", f"{quant['historical_var_95']:.2f}%" if quant["historical_var_95"] is not None else "無資料")
    columns[3].metric("HHI 集中度", f"{quant['herfindahl_index']:.3f}")
    columns[4].metric("有效持倉數", f"{quant['effective_positions']:.1f}")
    left, right = st.columns([3, 2])
    with left:
        _render_ai_completion_status(result.get("ai_meta"))
        st.markdown(result["report"])
    with right:
        st.dataframe(portfolio_table(result["assets"]), hide_index=True, width="stretch")
        if result["contributions"]:
            st.plotly_chart(px.bar(pd.DataFrame(result["contributions"]), x="symbol", y="risk_contribution", labels={"symbol": "股票代碼", "risk_contribution": "風險貢獻（%）"}), width="stretch")


def sentiment_page(settings: Settings | None, profile: dict[str, Any]) -> None:
    st.header("新聞情緒研究", divider="rainbow")
    portfolio_symbols = [item["symbol"] for item in st.session_state.portfolio if item["symbol"]]
    selected = st.selectbox("選擇投資組合持倉", list(dict.fromkeys(portfolio_symbols + ["自訂股票代碼"])))
    symbol = normalize_symbol(st.text_input("股票代碼", "AAPL")) if selected == "自訂股票代碼" else selected
    mode_col, limit_col = st.columns(2)
    mode = mode_col.radio("資料模式", ["Alpha Vantage 即時資料", "離線展示"], index=0 if settings and settings.sentiment_enabled else 1, horizontal=True)
    limit = limit_col.slider("最多新聞篇數", 10, 100, 50, 10)
    if not st.button("開始新聞情緒分析", type="primary", width="stretch"):
        if st.session_state.get("latest_sentiment"):
            render_sentiment(st.session_state.latest_sentiment)
        return
    try:
        if mode == "Alpha Vantage 即時資料":
            if not settings or not settings.sentiment_enabled:
                st.error("請設定 ALPHA_VANTAGE_API_KEY，或改用離線展示模式。")
                return
            with st.spinner("正在取得並驗證新聞情緒資料……"):
                news = cached_news(symbol, limit, settings)
        else:
            news = demo_news(symbol)
        stats = calculate_sentiment_stats(news)
        # Offline demo must remain fully offline even when live secrets exist.
        prices = cached_alpha_prices(symbol, settings) if settings and mode == "Alpha Vantage 即時資料" else []
        price_context = _price_context(prices)
        narrative = build_deterministic_sentiment_report(symbol, news, stats)
        ai_meta = None
        if settings and settings.ai_enabled and profile["ai_selection"] != "不使用 AI" and mode == "Alpha Vantage 即時資料":
            system, prompt = build_sentiment_prompt(symbol, news, stats, price_context)
            ai, provider_name, provider_reason = ai_service(settings, profile["ai_selection"], "news")
            with st.spinner(f"正在使用 {provider_name} 產生情緒研究摘要……"):
                ai_result = ai.analyze(system, prompt)
                narrative = ai_result.text
                ai_meta = _ai_meta(ai_result)
        result = {"symbol": symbol, "mode": mode, "news": news, "stats": stats, "prices": prices, "price_context": price_context, "narrative": narrative, "ai_meta": ai_meta}
        st.session_state.latest_sentiment = result
        render_sentiment(result)
    except ExternalServiceError as exc:
        logger.warning(exc.log_message)
        st.error(exc.public_message)
    except Exception:
        logger.exception("Sentiment analysis failed")
        st.error("新聞情緒分析未能完成，可改用離線展示模式檢查介面。")


def render_sentiment(result: dict[str, Any]) -> None:
    stats = result["stats"]
    st.caption(f"資料模式：{result['mode']} · 股票代碼：{result['symbol']}")
    metrics = st.columns(5)
    metrics[0].metric("新聞篇數", stats["count"])
    metrics[1].metric("股票情緒", f"{stats['average_ticker_score']:+.3f}")
    metrics[2].metric("文章情緒", f"{stats['average_overall_score']:+.3f}")
    metrics[3].metric("平均相關性", f"{stats['average_relevance']:.3f}")
    metrics[4].metric("主要情緒", sentiment_label_zh(stats["dominant_label"]))
    if st.session_state.get("latest_analysis"):
        st.info(combined_state_label(st.session_state.latest_analysis["portfolio_risk"], stats["average_ticker_score"]))
    overview_tab, timeline_tab, articles_tab, narrative_tab = st.tabs(["情緒概覽", "價格與趨勢", "新聞明細", "AI 解讀"])
    with overview_tab:
        left, right = st.columns(2)
        ticker_distribution = pd.DataFrame({"Label": [sentiment_label_zh(x) for x in SENTIMENT_ORDER], "Articles": [stats["distribution"][x] for x in SENTIMENT_ORDER], "Color key": SENTIMENT_ORDER})
        article_distribution = pd.DataFrame({"Label": [sentiment_label_zh(x) for x in SENTIMENT_ORDER], "Articles": [stats["overall_distribution"][x] for x in SENTIMENT_ORDER], "Color key": SENTIMENT_ORDER})
        chinese_colors = {sentiment_label_zh(key): value for key, value in SENTIMENT_COLORS.items()}
        left.plotly_chart(px.pie(ticker_distribution, names="Label", values="Articles", color="Label", color_discrete_map=chinese_colors, hole=0.45, title="股票情緒分布"), width="stretch")
        right.plotly_chart(px.pie(article_distribution, names="Label", values="Articles", color="Label", color_discrete_map=chinese_colors, hole=0.45, title="文章情緒分布"), width="stretch")
        st.plotly_chart(px.bar(source_distribution(result["news"]).head(10), x="Articles", y="Source", orientation="h", title="新聞來源分布"), width="stretch")
        topics = topic_distribution(result["news"]).head(10)
        if not topics.empty:
            st.plotly_chart(px.bar(topics, x="Weighted relevance", y="Topic", orientation="h", labels={"Weighted relevance": "加權相關性", "Topic": "主題"}), width="stretch")
    with timeline_tab:
        if result["prices"]:
            _price_chart(result["prices"], result["symbol"])
            with st.expander("股價明細與移動平均線"):
                st.dataframe(_price_frame(result["prices"]).tail(30), hide_index=True, width="stretch")
        timeline = daily_sentiment(result["news"])
        if not timeline.empty:
            figure = px.line(timeline, x="Date", y="Ticker sentiment", markers=True, hover_data=["Articles", "Relevance"], labels={"Date": "日期", "Ticker sentiment": "股票情緒", "Articles": "新聞篇數", "Relevance": "相關性"})
            figure.add_hline(y=0, line_dash="dot", line_color="#95A5A6")
            st.plotly_chart(figure, width="stretch")
        st.caption("價格與情緒僅按日期對齊呈現；圖表及相關係數不代表因果關係。")
        relationship = sentiment_price_correlation(result["news"], result["prices"])
        correlation = relationship["correlation"]
        st.metric(
            "情緒與次交易日報酬相關係數",
            f"{correlation:+.3f}" if correlation is not None else "資料不足",
            help=f"Aligned observations: {relationship['observations']}. Correlation is descriptive, not causal.",
        )
    with articles_tab:
        for index, item in enumerate(sorted(result["news"], key=lambda row: row["relevance"], reverse=True)[:20], 1):
            with st.expander(f"{index}. {item['title']} · {item['ticker_label']} · relevance {item['relevance']:.2f}"):
                st.caption(f"{item['source']} · {item['published_at'].isoformat() if item['published_at'] else 'Unknown time'}")
                detail_columns = st.columns(3)
                detail_columns[0].metric("股票情緒", f"{item['ticker_score']:+.3f}", sentiment_label_zh(item["ticker_label"]))
                detail_columns[1].metric("文章情緒", f"{item['overall_score']:+.3f}", sentiment_label_zh(item["overall_label"]))
                detail_columns[2].metric("相關性", f"{item['relevance']:.3f}", item["relevance_label"])
                if item["topics"]:
                    st.caption("相關主題：" + " · ".join(f"{topic['topic']} ({topic['weighted_relevance']:.3f})" for topic in item["topics"][:3]))
                st.write(item["summary"])
                if item["url"]:
                    st.link_button("開啟新聞原文", item["url"])
    with narrative_tab:
        _render_ai_completion_status(result.get("ai_meta"))
        st.markdown(result["narrative"])
        st.warning("新聞情緒屬於描述性資料，不代表預期報酬或未來走勢。")


def _price_chart(prices: list[dict[str, Any]], symbol: str) -> None:
    full_frame = _price_frame(prices)
    frame = full_frame.tail(30)
    figure = go.Figure(data=[go.Candlestick(x=frame["date"], open=frame["open"], high=frame["high"], low=frame["low"], close=frame["close"], name=symbol)])
    for window, color in ((5, "orange"), (10, "blue"), (20, "red"), (60, "purple")):
        figure.add_trace(go.Scatter(x=frame["date"], y=frame[f"MA{window}"], name=f"MA{window}", line={"color": color}))
    figure.update_layout(xaxis_rangeslider_visible=False, height=500, legend={"orientation": "h", "y": 1.1})
    st.plotly_chart(figure, width="stretch")


def _price_frame(prices: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(prices).sort_values("date").copy()
    for window in (5, 10, 20, 60):
        frame[f"MA{window}"] = frame["close"].rolling(window).mean()
    return frame


def _price_context(prices: list[dict[str, Any]]) -> dict[str, Any]:
    if len(prices) < 2:
        return {}
    frame = pd.DataFrame(prices)
    close = frame["close"].astype(float)
    returns = close.pct_change().dropna()
    return {"last_close": float(close.iloc[-1]), "period_return_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100), "annualized_volatility_pct": float(returns.std() * (252**0.5) * 100), "observations": int(len(frame))}


def etf_page() -> None:
    st.header("ETF 曝險比較", divider="rainbow")
    st.caption("上傳兩份 FMP 格式持倉 JSON；系統會依 ISIN、CUSIP 與股票代碼別名進行配對。")
    left_col, right_col = st.columns(2)
    left_file = left_col.file_uploader("ETF A 持倉檔", ["txt", "json"])
    right_file = right_col.file_uploader("ETF B 持倉檔", ["txt", "json"])
    if not left_file or not right_file:
        st.info("請同時上傳兩份檔案，以比較持倉重疊及推估國家曝險。")
        return
    try:
        left_symbol, left = parse_holdings(left_file.getvalue(), left_file.name)
        right_symbol, right = parse_holdings(right_file.getvalue(), right_file.name)
        metrics = overlap_metrics(left, right)
        columns = st.columns(4)
        columns[0].metric("共同持倉數", f"{metrics['common_holdings']:.0f}")
        columns[1].metric("重疊權重", f"{metrics['overlap_weight']:.1f}%")
        columns[2].metric(f"{left_symbol} top-10", f"{metrics['left_top10']:.1f}%")
        columns[3].metric(f"{right_symbol} top-10", f"{metrics['right_top10']:.1f}%")
        table = comparison_table(left_symbol, left, right_symbol, right)
        st.dataframe(table, hide_index=True, width="stretch")
        a, b = st.columns(2)
        a.plotly_chart(px.bar(country_exposure(left).head(12), x="Weight %", y="Country", orientation="h"), width="stretch")
        b.plotly_chart(px.bar(country_exposure(right).head(12), x="Weight %", y="Country", orientation="h"), width="stretch")
        st.download_button("下載比較結果 CSV", table.to_csv(index=False).encode("utf-8-sig"), f"{left_symbol}-{right_symbol}-comparison.csv", "text/csv")
    except HoldingsFormatError as exc:
        st.error(str(exc))


def ptp_page() -> None:
    st.header("產品與稅務警示", divider="rainbow")
    st.caption("將目前投資組合與使用者提供的最新 PTP 證券清單進行比對。")
    uploaded = st.file_uploader("上傳 PTP 清單 PDF", ["pdf"])
    if not uploaded:
        st.info("PTP 清單可能變動，請使用券商或資料供應商提供的最新文件。")
        return
    portfolio = [{"symbol": item["symbol"], "allocation": item["allocation"]} for item in st.session_state.portfolio]
    try:
        entries = parse_ptp_pdf(uploaded.getvalue())
        matches = screen_portfolio(portfolio, entries)
        st.metric("成功解析筆數", len(entries))
        if matches:
            st.error(f"投資組合中有 {len(matches)} 個股票代碼符合上傳清單。")
            st.dataframe(pd.DataFrame(matches), hide_index=True, width="stretch")
        else:
            st.success("目前投資組合沒有股票代碼符合上傳清單。")
        st.warning("股票代碼符合不代表個人稅務判定，請向合格專業人士確認。")
    except PTPFormatError as exc:
        st.error(str(exc))


def reports_page() -> None:
    st.header("研究報告與資料匯出", divider="rainbow")
    analysis = st.session_state.get("latest_analysis")
    if not analysis:
        st.info("請先完成投資組合風險或整合分析，才能產生報告。")
        return
    integrated = st.session_state.get("latest_integrated")
    sentiment_summary = None
    if integrated:
        sentiment_summary = [
            {
                "symbol": symbol,
                "count": result["stats"]["count"],
                "ticker_score": result["stats"]["average_ticker_score"],
                "overall_score": result["stats"]["average_overall_score"],
                "relevance": result["stats"]["average_relevance"],
                "dominant_label": result["stats"]["dominant_label"],
            }
            for symbol, result in integrated["sentiments"].items()
        ]
    pdf = portfolio_report_pdf(
        analysis["profile"], analysis["assets"], analysis["portfolio_risk"],
        analysis["report"], sentiment_summary,
    )
    st.download_button("下載整合研究 PDF", pdf, "signallens-integrated-research.pdf", "application/pdf", type="primary")
    st.download_button("下載資產指標 CSV", portfolio_table(analysis["assets"]).to_csv(index=False).encode("utf-8-sig"), "signallens-asset-metrics.csv", "text/csv")

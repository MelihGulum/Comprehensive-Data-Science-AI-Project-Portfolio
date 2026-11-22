from __future__ import annotations

import html
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from app.graph.workflow import build_graph
from app.schemas.models import VerificationResult
from app.services.run_store import RunStore
from app.services.data_ingestion import DataIngestionService
from app.services.company_registry import COMPANIES, label_for_ticker
from evaluation.metrics import evaluate

APP_VERSION = "v1.5"
METRIC_LABELS = {
    "revenue_growth": "Revenue growth",
    "gross_margin": "Gross margin",
    "net_margin": "Net margin",
}

st.set_page_config(
    page_title="AI Agent Finance",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --radius: 16px; }
.block-container {padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stSidebar"] {border-right:1px solid rgba(128,128,128,.16);}
[data-testid="stMetric"] {
    border:1px solid rgba(128,128,128,.20);
    border-radius:var(--radius);
    padding:16px 18px;
    background:rgba(128,128,128,.025);
}
[data-testid="stMetricLabel"] {font-weight:600;}
.small-muted {opacity:.64; font-size:.84rem;}
.eyebrow {font-size:.76rem; letter-spacing:.12em; text-transform:uppercase; opacity:.58; font-weight:700;}
.hero {
    border:1px solid rgba(128,128,128,.18); border-radius:18px; padding:20px 22px;
    background:linear-gradient(135deg, rgba(99,102,241,.055), rgba(128,128,128,.015));
    margin-bottom:1rem;
}
.hero-row {display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;}
.hero-title {font-size:1.55rem; font-weight:750; margin:.15rem 0;}
.hero-sub {opacity:.66; font-size:.9rem;}
.pill {display:inline-block; padding:.28rem .6rem; border-radius:999px; font-size:.75rem; font-weight:700;}
.pill-ok {background:rgba(33,150,83,.11); border:1px solid rgba(33,150,83,.28);}
.pill-warn {background:rgba(242,153,74,.11); border:1px solid rgba(242,153,74,.28);}
.agent-card {
    border:1px solid rgba(128,128,128,.18); border-radius:16px; padding:17px 18px;
    margin:.55rem 0; background:rgba(128,128,128,.018);
}
.agent-row {display:flex; align-items:flex-start; gap:.8rem;}
.agent-num {
    width:27px;height:27px;border-radius:50%;display:flex;align-items:center;justify-content:center;
    background:rgba(99,102,241,.12);font-size:.78rem;font-weight:800;flex:0 0 27px;
}
.agent-title {font-weight:700;}
.agent-meta {opacity:.62;font-size:.82rem;margin-top:.18rem;}
.agent-ok {float:right; font-size:.76rem; opacity:.72;}
.ai-box {
    border-left:3px solid rgba(99,102,241,.72); border-radius:10px;
    padding:15px 17px; background:rgba(99,102,241,.055); margin:.7rem 0 1rem 0;
}
.status-pass {padding:.7rem .9rem;border-radius:.65rem;background:rgba(33,150,83,.09);border:1px solid rgba(33,150,83,.28);}
.status-fail {padding:.7rem .9rem;border-radius:.65rem;background:rgba(235,87,87,.09);border:1px solid rgba(235,87,87,.28);}
hr {border-color:rgba(128,128,128,.14)!important;}
</style>
""", unsafe_allow_html=True)


def pct(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.1%}"


def money(value: float) -> str:
    value = float(value or 0)
    return f"${value:.4f}" if value < 0.01 else f"${value:.2f}"


def claims_from(result: dict) -> list[dict]:
    analysis = result.get("analysis", {}) or {}
    return sum([analysis.get(k, []) or [] for k in ("internal_claims", "competitor_claims", "comparison_claims")], [])


def metrics_frame(result: dict) -> pd.DataFrame:
    df = pd.DataFrame(result.get("metrics", []))
    return df.sort_values("year") if not df.empty and "year" in df else df


def llm_claim_count(result: dict) -> int:
    return len(claims_from(result))


def llm_used(result: dict) -> bool:
    return bool(result.get("llm_usage"))


def warning_summary(result: dict) -> tuple[list[str], list[str]]:
    """Separate blocking/data warnings from optional-tool warnings."""
    important, optional = [], []
    for w in result.get("warnings", []) or []:
        text = str(w)
        if "TAVILY_API_KEY" in text or "web research" in text.lower() or "competitor evidence" in text.lower():
            optional.append(text)
        else:
            important.append(text)
    return important, optional


def agent_steps(result: dict, verification: VerificationResult) -> list[tuple[str, str, str]]:
    plan = result.get("research_plan") or {}
    tools = plan.get("tools") or plan.get("selected_tools") or []
    if isinstance(tools, str):
        tools = [tools]
    provider = (result.get("ingestion") or {}).get("provider", "Financial data provider")
    web_ok = bool(result.get("evidence"))
    usage = result.get("llm_usage") or []

    return [
        ("Understand request", "Workflow", "Company, fiscal history and peer scope resolved"),
        ("Plan research", "Research Agent", f"Tool plan: {', '.join(map(str, tools))}" if tools else "Research requirements evaluated"),
        ("Retrieve fundamentals", str(provider), "Primary and competitor financial statements ingested"),
        ("Collect external context", "Research tools", "Evidence collected" if web_ok else "Optional web evidence skipped / unavailable"),
        ("Calculate metrics", "Deterministic Python", "Revenue growth, gross margin and net margin recomputed from source rows"),
        ("Analyze evidence", "Gemini Analysis Agent" if usage else "Deterministic analysis", f"{llm_claim_count(result)} structured claims available"),
        ("Verify claims", "Verification Engine", f"{verification.supported_claims}/{verification.checked_claims} claims supported"),
        ("Generate report", "Reporting layer", "Auditable report assembled from verified outputs"),
    ]


def render_run_header(result: dict, company: str, ticker: str, verification: VerificationResult) -> None:
    passed = verification.passed
    status_cls = "pill-ok" if passed else "pill-warn"
    status = "VERIFIED" if passed else "REVIEW REQUIRED"
    ticker_text = f" · {html.escape(ticker)}" if ticker else ""
    st.markdown(f"""
    <div class="hero">
      <div class="hero-row">
        <div>
          <div class="eyebrow">Financial intelligence workspace</div>
          <div class="hero-title">{html.escape(company)}{ticker_text}</div>
          <div class="hero-sub">Deterministic finance math · agentic research · grounded Gemini analysis · claim-level verification</div>
        </div>
        <div><span class="pill {status_cls}">{status}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_agent_execution(result: dict, verification: VerificationResult, compact: bool = False) -> None:
    steps = agent_steps(result, verification)
    st.markdown("### Agent execution")
    st.caption("What the AI agent decided, which tools were used, and where deterministic checks took over.")
    for i, (title, actor, detail) in enumerate(steps, 1):
        if compact and i not in (2, 3, 6, 7):
            continue
        safe_detail = html.escape(detail)
        st.markdown(
            f"""<div class="agent-card"><div class="agent-row">
            <div class="agent-num">{i}</div><div style="width:100%">
            <span class="agent-ok">✓ COMPLETE</span>
            <div class="agent-title">{html.escape(title)}</div>
            <div class="agent-meta"><b>{html.escape(actor)}</b> · {safe_detail}</div>
            </div></div></div>""",
            unsafe_allow_html=True,
        )


def render_ai_insight(result: dict) -> None:
    usage = result.get("llm_usage") or []
    claims = claims_from(result)
    analysis = result.get("analysis", {}) or {}
    verification = VerificationResult(**result["verification"])

    st.markdown("### AI analyst")
    st.caption("Interpretation answers “what does it mean?”; the underlying financial arithmetic remains deterministic.")

    if not usage:
        st.info("Gemini was not used in this run. No AI interpretation is presented.")
        return

    executive = analysis.get("executive_analysis")
    if not executive:
        summary = analysis.get("executive_summary") or []
        if isinstance(summary, list):
            executive = " ".join(str(x) for x in summary if x)
        elif isinstance(summary, str):
            executive = summary

    if executive:
        st.markdown(
            f'<div class="ai-box"><div class="eyebrow">Executive analysis</div>'
            f'<div style="margin-top:.45rem;line-height:1.65">{html.escape(str(executive))}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Gemini did not produce a higher-level executive interpretation for the available evidence.")

    insights = analysis.get("insights") or []
    if insights:
        st.markdown("#### Key findings")
        cols = st.columns(2)
        for idx, insight in enumerate(insights):
            category = str(insight.get("category", "other")).replace("_", " ").upper()
            title = str(insight.get("title", "Finding"))
            narrative = str(insight.get("narrative", ""))
            status = insight.get("evidence_status", "sufficient")
            claim_ids = insight.get("claim_ids") or []

            with cols[idx % 2]:
                if status == "insufficient":
                    st.markdown(
                        f"""<div class="agent-card">
                        <div class="eyebrow">{html.escape(category)} · INSUFFICIENT EVIDENCE</div>
                        <div class="agent-title" style="margin-top:.35rem">{html.escape(title)}</div>
                        <div class="agent-meta" style="font-size:.9rem;line-height:1.55">{html.escape(narrative)}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    provenance = f"{len(claim_ids)} verified claim{'s' if len(claim_ids) != 1 else ''}"
                    st.markdown(
                        f"""<div class="agent-card">
                        <div class="eyebrow">{html.escape(category)}</div>
                        <div class="agent-title" style="margin-top:.35rem">{html.escape(title)}</div>
                        <div style="opacity:.82;font-size:.91rem;line-height:1.55;margin-top:.35rem">{html.escape(narrative)}</div>
                        <div class="agent-meta">Based on · {html.escape(provenance)}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
    else:
        st.caption("No structured higher-level insights were returned for this run.")

    st.markdown("#### Analysis provenance")
    p1, p2, p3 = st.columns(3)
    p1.metric("Reasoning model", "Gemini")
    p2.metric("Verified claims", f"{verification.supported_claims}/{verification.checked_claims}")
    p3.metric("Structured insights", len(insights))

    with st.expander(f"View verified underlying claims ({len(claims)})", expanded=False):
        for claim in claims:
            text = claim.get("text") or claim.get("claim") or claim.get("statement")
            if text:
                cid = claim.get("claim_id", "")
                st.markdown(f"- {text}")
                if cid:
                    st.caption(cid)

def render_overview(result: dict, ev, latency: float, estimated_cost: float, verification: VerificationResult) -> None:
    df = metrics_frame(result)
    latest = df.iloc[-1] if not df.empty else None
    previous = df.iloc[-2] if len(df) > 1 else None

    st.markdown("### Executive snapshot")
    a, b, c, d = st.columns(4)
    a.metric("Revenue growth", pct(None if latest is None else latest.get("revenue_growth")),
             None if previous is None or pd.isna(previous.get("revenue_growth")) else f"{(latest['revenue_growth']-previous['revenue_growth'])*100:+.1f} pp vs prior year")
    b.metric("Gross margin", pct(None if latest is None else latest.get("gross_margin")),
             None if previous is None else f"{(latest['gross_margin']-previous['gross_margin'])*100:+.1f} pp vs prior year")
    c.metric("Net margin", pct(None if latest is None else latest.get("net_margin")),
             None if previous is None else f"{(latest['net_margin']-previous['net_margin'])*100:+.1f} pp vs prior year")
    d.metric("Evidence coverage", f"{verification.supported_claims}/{verification.checked_claims}",
             "PASS" if verification.passed else "Review required")

    st.markdown("### Financial trends")
    left, right = st.columns([1.45, 1])
    with left:
        if not df.empty:
            margin_df = df.set_index("year")[["gross_margin", "net_margin"]].rename(
                columns={"gross_margin": "Gross margin", "net_margin": "Net margin"}
            ) * 100
            st.line_chart(margin_df, y_label="Percent", x_label="Fiscal year", height=315)
        else:
            st.info("No metric history available.")
    with right:
        if not df.empty:
            growth = df.dropna(subset=["revenue_growth"])[["year", "revenue_growth"]].copy()
            growth["Revenue growth"] = growth["revenue_growth"] * 100
            st.bar_chart(growth.set_index("year")[["Revenue growth"]], y_label="Percent", x_label="Fiscal year", height=315)
        else:
            st.info("No growth history available.")

    st.markdown("### Reliability")
    st.caption("Run-level validation signals — these are workflow checks, not a claim that the AI model is universally accurate.")

    usage = result.get("llm_usage", []) or []
    total_calls = len(usage)

    def _usage_number(item: dict, *keys: str) -> int:
        for key in keys:
            value = item.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return 0

    total_tokens = sum(
        _usage_number(
            item,
            "total_token_count",
            "total_tokens",
            "tokens",
        )
        for item in usage
    )
    if total_tokens == 0:
        # Some Gemini wrappers only expose prompt + candidate/output counts.
        total_tokens = sum(
            _usage_number(item, "prompt_token_count", "input_tokens", "prompt_tokens")
            + _usage_number(item, "candidates_token_count", "output_tokens", "completion_tokens")
            for item in usage
        )

    r1, r2, r3 = st.columns(3)
    r1.metric(
        "Calculation checks",
        f"{ev.numerical_accuracy:.1%}",
        help="Share of checked numerical claims that matched deterministic recomputation.",
    )
    r2.metric(
        "Evidence coverage",
        f"{ev.citation_support_rate:.1%}",
        help="Share of claims requiring support that were linked to available evidence or calculation provenance.",
    )
    r3.metric(
        "Unsupported claims",
        f"{ev.unsupported_claim_rate:.1%}",
        help="Share of evaluated claims that the verification layer could not support.",
    )

    r4, r5, r6 = st.columns(3)
    r4.metric(
        "End-to-end latency",
        f"{latency/1000:.2f}s",
        help="Wall-clock time for ingestion, agent workflow, analysis and verification.",
    )

    if total_calls:
        usage_value = f"{total_calls} call{'s' if total_calls != 1 else ''}"
        usage_delta = f"{total_tokens:,} tokens" if total_tokens else "Token count unavailable"
    else:
        usage_value = "Not used"
        usage_delta = None

    r5.metric(
        "Gemini usage",
        usage_value,
        usage_delta,
        help="Observed Gemini calls and token usage recorded for this run.",
    )

    has_cost_signal = any("estimated_cost_usd" in item and item.get("estimated_cost_usd") is not None for item in usage)
    if not total_calls:
        cost_value = "N/A"
        cost_help = "No Gemini call was recorded for this run."
    elif has_cost_signal:
        cost_value = money(estimated_cost)
        cost_help = "Estimated API cost from recorded Gemini usage metadata. This may be $0 when the configured tier/pricing produces no charge."
    else:
        cost_value = "N/A"
        cost_help = "Gemini ran, but this application did not receive enough pricing metadata to estimate API cost."

    r6.metric(
        "Estimated API cost",
        cost_value,
        help=cost_help,
    )


def render_financials(result: dict) -> None:
    df = metrics_frame(result)
    st.markdown("### Financial performance")
    st.caption("Calculated deterministically from normalized source financials.")
    if df.empty:
        st.info("No financial metrics available.")
        return
    display = df.copy()
    for col in ["revenue_growth", "gross_margin", "net_margin"]:
        if col in display:
            display[col] = display[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
    display = display.rename(columns={
        "year": "Year", "revenue_growth": "Revenue growth",
        "gross_margin": "Gross margin", "net_margin": "Net margin"
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    if {"year", "gross_margin", "net_margin"}.issubset(df.columns):
        st.markdown("#### Margin movement")
        pivot = df.set_index("year")[["gross_margin", "net_margin"]].rename(
            columns={"gross_margin": "Gross margin", "net_margin": "Net margin"}
        ) * 100
        st.line_chart(pivot, y_label="Percent", x_label="Fiscal year", height=350)


def render_competitors(result: dict) -> None:
    st.markdown("### Competitor intelligence")
    st.caption("Comparable peer fundamentals first; AI interpretation only where the available evidence supports it.")

    direct = result.get("competitor_financials", {}) or {}
    primary_df = metrics_frame(result)
    analysis = result.get("analysis", {}) or {}
    evidence = pd.DataFrame(result.get("evidence", []))
    comparison_claims = analysis.get("comparison_claims", []) or []

    # Build a comparable peer frame from direct fundamentals.
    peer_rows = []
    for ticker, rows in direct.items():
        if not rows:
            continue
        rdf = pd.DataFrame(rows)
        if rdf.empty or "year" not in rdf:
            continue
        for _, row in rdf.iterrows():
            revenue = float(row.get("revenue") or 0)
            if not revenue:
                continue
            peer_rows.append({
                "ticker": ticker,
                "company": label_for_ticker(ticker),
                "year": int(row["year"]),
                "revenue_growth": row.get("revenue_growth"),
                "gross_margin": float(row.get("gross_profit") or 0) / revenue,
                "net_margin": float(row.get("net_income") or 0) / revenue,
            })

    peer_df = pd.DataFrame(peer_rows)

    # Add primary company using already-normalized deterministic metrics.
    primary_name = (result.get("ingestion") or {}).get("company") or result.get("company") or "Primary company"
    primary_ticker = result.get("ticker") or (result.get("ingestion") or {}).get("ticker") or ""
    if not primary_df.empty:
        primary_rows = primary_df.copy()
        primary_rows["ticker"] = primary_ticker
        primary_rows["company"] = primary_name
        primary_rows["is_primary"] = True
    else:
        primary_rows = pd.DataFrame()

    if not peer_df.empty:
        peer_df["is_primary"] = False

    combined = pd.concat(
        [x for x in (primary_rows, peer_df) if not x.empty],
        ignore_index=True,
        sort=False,
    ) if (not primary_rows.empty or not peer_df.empty) else pd.DataFrame()

    if combined.empty:
        st.info("No comparable peer fundamentals are available for this run.")
        return

    # Compare only a common fiscal period to avoid silent cross-period comparisons.
    years_by_company = combined.groupby("company")["year"].apply(set).tolist()
    common_years = set.intersection(*years_by_company) if years_by_company else set()
    comparable_year = max(common_years) if common_years else None

    if comparable_year is None:
        st.warning("No common fiscal period exists across the selected peer set. Cross-period ranking has been intentionally suppressed.")
        with st.expander("Available peer fundamentals"):
            st.dataframe(combined.sort_values(["company", "year"]), use_container_width=True, hide_index=True)
        return

    comparable = combined[combined["year"] == comparable_year].copy()

    st.markdown(f"#### Peer benchmark · FY{comparable_year}")
    metric_options = {
        "Gross margin": "gross_margin",
        "Net margin": "net_margin",
        "Revenue growth": "revenue_growth",
    }
    metric_label = st.selectbox("Benchmark metric", list(metric_options.keys()), key="competitor_metric")
    metric = metric_options[metric_label]

    comparable = comparable.dropna(subset=[metric]).copy()
    if comparable.empty:
        st.info(f"No comparable {metric_label.lower()} values are available for FY{comparable_year}.")
        return

    primary_match = comparable[comparable["is_primary"] == True]
    primary_value = float(primary_match.iloc[0][metric]) if not primary_match.empty else None

    # Senior-style benchmark table with deterministic deltas.
    table = comparable[["company", "ticker", "is_primary", metric]].copy()
    table["Company"] = table.apply(
        lambda r: f"{r['company']} · PRIMARY" if r["is_primary"] else r["company"], axis=1
    )
    table[metric_label] = table[metric].map(lambda x: f"{float(x):.1%}")
    if primary_value is not None:
        table["vs primary"] = table.apply(
            lambda r: "—" if r["is_primary"] else f"{(float(r[metric]) - primary_value) * 100:+.1f} pp",
            axis=1,
        )
    else:
        table["vs primary"] = "—"

    left, right = st.columns([1.05, 1.35])
    with left:
        st.dataframe(
            table[["Company", metric_label, "vs primary"]],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        chart = comparable[["company", metric]].copy()
        chart[metric_label] = chart[metric] * 100
        st.bar_chart(
            chart.set_index("company")[[metric_label]],
            y_label="Percent",
            height=320,
        )

    # Relative-position summary is descriptive only; no winner/ranking language.
    st.markdown("#### Relative position")
    peers_only = comparable[comparable["is_primary"] == False]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Primary", pct(primary_value))
    if not peers_only.empty:
        peer_median = float(peers_only[metric].median())
        c2.metric("Peer median", pct(peer_median))
        c3.metric("Difference", f"{(primary_value-peer_median)*100:+.1f} pp" if primary_value is not None else "—")
        expected_peers = len(direct)
        c4.metric("Peer coverage", f"{len(peers_only)}/{expected_peers}" if expected_peers else str(len(peers_only)))
    else:
        c2.metric("Peer median", "—")
        c3.metric("Difference", "—")
        c4.metric("Peer coverage", "0")

    # Verified comparison claims stay visible, but are not allowed to replace fundamentals.
    st.markdown("#### Verified comparisons")
    if comparison_claims:
        for claim in comparison_claims:
            text = claim.get("text") or claim.get("claim") or claim.get("statement")
            if text:
                st.markdown(
                    f"""<div class="agent-card">
                    <div class="eyebrow">VERIFIED COMPARISON</div>
                    <div style="margin-top:.35rem;line-height:1.55">{html.escape(str(text))}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No same-period comparison claims passed into the verified claim set.")

    # Peer interpretation: only show a Gemini peer insight if it was grounded in peer claims.
    peer_insights = [
        x for x in (analysis.get("insights") or [])
        if x.get("category") == "peer_positioning"
    ]
    st.markdown("#### Peer analysis")
    if peer_insights:
        for insight in peer_insights:
            status = insight.get("evidence_status", "sufficient")
            if status == "sufficient":
                st.markdown(
                    f'<div class="ai-box"><div class="eyebrow">Grounded Gemini interpretation</div>'
                    f'<b>{html.escape(str(insight.get("title", "Peer positioning")))}</b><br>'
                    f'{html.escape(str(insight.get("narrative", "")))}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info(str(insight.get("narrative") or "Peer positioning cannot be concluded from the available evidence."))
    else:
        st.info("No grounded peer-positioning interpretation was produced for this run.")

    st.markdown("#### Evidence & coverage")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Comparable period", f"FY{comparable_year}")
    e2.metric("Direct peers", len(peers_only))
    e3.metric("Web evidence", len(evidence))
    e4.metric("Comparison claims", len(comparison_claims))

    if evidence.empty:
        st.caption("External web evidence is unavailable; the benchmark above uses direct normalized competitor fundamentals only.")
    else:
        with st.expander(f"Source evidence ({len(evidence)})", expanded=False):
            cols = [c for c in [
                "company", "metric", "value", "period", "confidence",
                "source_title", "source_url", "source_date"
            ] if c in evidence.columns]
            st.dataframe(evidence[cols], use_container_width=True, hide_index=True)

    with st.expander("Comparison claim register", expanded=False):
        if comparison_claims:
            st.dataframe(pd.DataFrame(comparison_claims), use_container_width=True, hide_index=True)
        else:
            st.caption("No verified comparison claims available.")

def render_sources(result: dict) -> None:
    st.markdown("### Sources & lineage")
    ingestion = result.get("ingestion", {}) or {}
    evidence = pd.DataFrame(result.get("evidence", []))
    news = result.get("news", []) or []
    important, optional = warning_summary(result)

    c1, c2, c3 = st.columns(3)
    c1.metric("Primary provider", str(ingestion.get("provider", "—")))
    c2.metric("External evidence", len(evidence))
    c3.metric("News items", len(news))

    if optional:
        with st.expander("Optional integrations", expanded=False):
            for w in optional:
                st.caption(f"○ {w}")
    if important:
        for w in important:
            st.warning(w)

    if ingestion:
        with st.expander("Data lineage", expanded=True):
            st.json(ingestion)
    if not evidence.empty:
        st.dataframe(evidence, use_container_width=True, hide_index=True)
    if news:
        with st.expander(f"Recent news ({len(news)})"):
            ndf = pd.DataFrame(news)
            cols = [c for c in ["title", "publish_date", "source", "url"] if c in ndf.columns]
            st.dataframe(ndf[cols], use_container_width=True, hide_index=True)


def render_audit(result: dict, run_id: str, latency: float, estimated_cost: float, verification: VerificationResult) -> None:
    st.markdown("### Audit & observability")
    css = "status-pass" if verification.passed else "status-fail"
    st.markdown(
        f'<div class="{css}"><b>{"Verification passed" if verification.passed else "Verification failed"}</b>'
        f' · Run <code>{html.escape(run_id)}</code> · {verification.checked_claims} claims checked</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d, e = st.columns(5)
    a.metric("Supported", verification.supported_claims)
    b.metric("Issues", len(verification.issues))
    c.metric("Revisions", result.get("revision_attempts", result.get("revision_count", 0)) or 0)
    d.metric("Latency", f"{latency:.0f} ms")
    e.metric("LLM cost", money(estimated_cost))

    if verification.issues:
        st.markdown("#### Verification issues")
        st.dataframe(pd.DataFrame([i.model_dump() for i in verification.issues]), use_container_width=True, hide_index=True)

    render_agent_execution(result, verification)

    claims = pd.DataFrame(claims_from(result))
    evidence = pd.DataFrame(result.get("evidence", []))
    usage = pd.DataFrame(result.get("llm_usage", []))
    t1, t2, t3, t4 = st.tabs(["Claims", "Evidence lineage", "LLM usage", "Raw tool plan"])
    with t1:
        st.dataframe(claims, use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(evidence, use_container_width=True, hide_index=True)
    with t3:
        if not usage.empty:
            st.dataframe(usage, use_container_width=True, hide_index=True)
        else:
            st.info("No LLM usage recorded.")
    with t4:
        st.json(result.get("research_plan") or {})


def build_auditable_report_markdown(
    result: dict,
    company: str,
    run_id: str,
    latency: float,
    estimated_cost: float,
) -> str:
    """Build the downloadable report from the same verified state shown in the UI."""
    analysis = result.get("analysis", {}) or {}
    verification = VerificationResult(**result["verification"])
    metrics = metrics_frame(result)
    insights = analysis.get("insights") or []
    claims = claims_from(result)
    evidence = result.get("evidence", []) or []
    ingestion = result.get("ingestion", {}) or {}
    warnings = result.get("warnings", []) or []

    executive = analysis.get("executive_analysis")
    if not executive:
        summary = analysis.get("executive_summary") or []
        executive = " ".join(map(str, summary)) if isinstance(summary, list) else str(summary or "")

    lines = [
        f"# Financial Intelligence Report — {company}",
        "",
        "> Auditable output: deterministic financial calculations + grounded AI interpretation + claim-level verification.",
        "",
        "## Executive analysis",
        "",
        executive or "No higher-level AI interpretation was available for this run.",
        "",
        "## Key findings",
        "",
    ]

    if insights:
        for insight in insights:
            status = insight.get("evidence_status", "sufficient")
            suffix = " — insufficient evidence" if status == "insufficient" else ""
            lines += [
                f"### {insight.get('title', 'Finding')}{suffix}",
                "",
                str(insight.get("narrative", "")),
                "",
            ]
            ids = insight.get("claim_ids") or []
            if ids:
                lines.append(f"Grounded in verified claims: {', '.join(ids)}")
                lines.append("")
    else:
        lines += ["No structured higher-level findings were produced.", ""]

    lines += ["## Financial performance", ""]
    if not metrics.empty:
        lines.append("| Fiscal year | Revenue growth | Gross margin | Net margin |")
        lines.append("|---:|---:|---:|---:|")
        for _, row in metrics.iterrows():
            rg = "—" if pd.isna(row.get("revenue_growth")) else f"{row['revenue_growth']:.1%}"
            gm = "—" if pd.isna(row.get("gross_margin")) else f"{row['gross_margin']:.1%}"
            nm = "—" if pd.isna(row.get("net_margin")) else f"{row['net_margin']:.1%}"
            lines.append(f"| {int(row['year'])} | {rg} | {gm} | {nm} |")
    else:
        lines.append("No normalized financial metrics were available.")

    lines += ["", "## Verified claim register", ""]
    if claims:
        for claim in claims:
            text = claim.get("text") or claim.get("claim") or claim.get("statement") or ""
            cid = claim.get("claim_id", "")
            refs = claim.get("evidence_ids") or claim.get("source_refs") or []
            provenance = f" · evidence: {', '.join(map(str, refs))}" if refs else ""
            lines.append(f"- {text} `{cid}`{provenance}")
    else:
        lines.append("No structured claims were available.")

    lines += [
        "",
        "## Evidence & limitations",
        "",
        f"- External evidence objects: {len(evidence)}",
        f"- Direct data provider: {ingestion.get('provider', 'not recorded')}",
    ]
    if warnings:
        for warning in warnings:
            lines.append(f"- Limitation: {warning}")
    elif not evidence:
        lines.append("- Limitation: no external web evidence was collected for this run.")

    lines += [
        "",
        "## Verification & run metadata",
        "",
        f"- Verification: {'PASS' if verification.passed else 'REVIEW REQUIRED'}",
        f"- Claims checked: {verification.checked_claims}",
        f"- Supported claims: {verification.supported_claims}",
        f"- Verification issues: {len(verification.issues)}",
        f"- Run ID: {run_id}",
        f"- End-to-end latency: {latency/1000:.2f}s",
        f"- Estimated API cost: {money(estimated_cost) if estimated_cost else 'N/A'}",
        "",
        "## Method",
        "",
        "Financial ratios are calculated deterministically from normalized source rows. "
        "Gemini is used for grounded interpretation, not for financial arithmetic. "
        "Structured claims are passed through the verification layer before inclusion in this report.",
    ]
    return "\n".join(lines)


def render_report(
    result: dict,
    company: str,
    run_id: str,
    latency: float,
    estimated_cost: float,
) -> None:
    analysis = result.get("analysis", {}) or {}
    verification = VerificationResult(**result["verification"])
    metrics = metrics_frame(result)
    insights = analysis.get("insights") or []
    claims = claims_from(result)
    evidence = result.get("evidence", []) or []
    ingestion = result.get("ingestion", {}) or {}

    st.markdown("### Auditable report")
    st.caption("A decision-ready view of verified financial facts, grounded AI interpretation, evidence coverage and run provenance.")

    # Report status header
    status_cls = "status-pass" if verification.passed else "status-fail"
    status_text = "Verified report" if verification.passed else "Review required"
    st.markdown(
        f'<div class="{status_cls}"><b>{status_text}</b> · '
        f'{verification.supported_claims}/{verification.checked_claims} claims supported · '
        f'Run <code>{html.escape(run_id)}</code></div>',
        unsafe_allow_html=True,
    )

    # Executive interpretation
    st.markdown("#### Executive analysis")
    executive = analysis.get("executive_analysis")
    if not executive:
        summary = analysis.get("executive_summary") or []
        executive = " ".join(map(str, summary)) if isinstance(summary, list) else str(summary or "")
    if executive:
        st.markdown(
            f'<div class="ai-box"><div class="eyebrow">Grounded Gemini interpretation</div>'
            f'<div style="margin-top:.45rem;line-height:1.65">{html.escape(str(executive))}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No higher-level AI interpretation was available for this run.")

    # Key findings, not a repeated dump of every metric.
    st.markdown("#### Key findings")
    if insights:
        cols = st.columns(2)
        for idx, insight in enumerate(insights):
            with cols[idx % 2]:
                category = str(insight.get("category", "other")).replace("_", " ").upper()
                title = str(insight.get("title", "Finding"))
                narrative = str(insight.get("narrative", ""))
                status = insight.get("evidence_status", "sufficient")
                ids = insight.get("claim_ids") or []
                badge = "INSUFFICIENT EVIDENCE" if status == "insufficient" else f"{len(ids)} VERIFIED CLAIM{'S' if len(ids) != 1 else ''}"
                st.markdown(
                    f"""<div class="agent-card">
                    <div class="eyebrow">{html.escape(category)} · {html.escape(badge)}</div>
                    <div class="agent-title" style="margin-top:.35rem">{html.escape(title)}</div>
                    <div style="opacity:.82;font-size:.91rem;line-height:1.55;margin-top:.35rem">{html.escape(narrative)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No structured higher-level findings were produced.")

    # Compact performance table is more report-like than 11 repetitive sentences.
    st.markdown("#### Financial performance")
    if not metrics.empty:
        display = metrics.copy()
        for col in ("revenue_growth", "gross_margin", "net_margin"):
            if col in display:
                display[col] = display[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
        display = display.rename(columns={
            "year": "Fiscal year",
            "revenue_growth": "Revenue growth",
            "gross_margin": "Gross margin",
            "net_margin": "Net margin",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No normalized financial metrics were available.")

    # Evidence and verification summary.
    st.markdown("#### Evidence & verification")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Claims checked", verification.checked_claims)
    e2.metric("Supported", verification.supported_claims)
    e3.metric("External evidence", len(evidence))
    e4.metric("Verification", "PASS" if verification.passed else "REVIEW")

    provider = ingestion.get("provider", "Not recorded")
    st.caption(f"Primary financial source · {provider}")

    _, optional_warnings = warning_summary(result)
    important_warnings, _ = warning_summary(result)
    all_limits = [*important_warnings, *optional_warnings]
    if all_limits:
        with st.expander("Limitations & unavailable evidence", expanded=True):
            for warning in all_limits:
                st.markdown(f"- {warning}")
    elif not evidence:
        with st.expander("Limitations & unavailable evidence"):
            st.markdown("- No external web evidence was collected for this run.")

    # Raw claims belong behind disclosure, not in the executive body.
    with st.expander(f"Verified claim register ({len(claims)})", expanded=False):
        if claims:
            for claim in claims:
                text = claim.get("text") or claim.get("claim") or claim.get("statement")
                if text:
                    st.markdown(f"- {text}")
                    cid = claim.get("claim_id")
                    if cid:
                        st.caption(f"Claim ID · {cid}")
        else:
            st.caption("No structured claims available.")

    st.markdown("#### Method & provenance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Run ID", run_id)
    m2.metric("End-to-end latency", f"{latency/1000:.2f}s")
    m3.metric("Reasoning layer", "Gemini" if result.get("llm_usage") else "Deterministic")

    st.caption(
        "Financial ratios are calculated deterministically from normalized source rows. "
        "Gemini provides grounded interpretation; it does not perform the financial arithmetic. "
        "Structured claims are verified before they enter the report."
    )

    report_md = build_auditable_report_markdown(
        result=result,
        company=company,
        run_id=run_id,
        latency=latency,
        estimated_cost=estimated_cost,
    )
    st.download_button(
        "Download auditable Markdown report",
        report_md,
        file_name=f"{company.lower().replace(' ', '_')}_auditable_financial_report.md",
        mime="text/markdown",
        use_container_width=True,
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ◈ AI Agent Finance")
    st.caption(f"{APP_VERSION} · Auditable financial intelligence")
    st.divider()

    source_mode = st.radio(
        "Data source",
        ["Automatic · Yahoo Finance", "Upload CSV"],
        help="Automatic mode ingests annual statements and market context. CSV is the controlled fallback.",
    )

    if source_mode.startswith("Automatic"):
        company_label = st.selectbox("Company", list(COMPANIES.keys()), index=0)
        selected = COMPANIES[company_label]
        company, ticker = selected["name"], selected["ticker"]
        st.caption(f"{ticker} · {selected['sector']}")
        years = st.slider("Annual periods", 2, 4, 4)
        suggested = selected["competitors"]
        competitor_tickers = st.multiselect(
            "Peer set",
            options=sorted({x for v in COMPANIES.values() for x in [v["ticker"], *v["competitors"]] if x != ticker}),
            default=suggested[:3],
            format_func=label_for_ticker,
        )
    else:
        company = st.text_input("Company / display name", "Uploaded Company")
        ticker, years, competitor_tickers = "", 4, []

    upload = st.file_uploader(
        "Annual financial CSV", type=["csv"],
        help="Required: year, revenue, gross_profit, net_income, currency",
        disabled=source_mode != "Upload CSV",
    )

    with st.expander("Analysis options"):
        include_market = st.checkbox("30-day market data", True, disabled=not source_mode.startswith("Automatic"))
        include_news = st.checkbox("Recent news", True, disabled=not source_mode.startswith("Automatic"))
        max_revisions = st.slider("Max verification revisions", 0, 3, 2)

    ready = bool(ticker.strip()) if source_mode.startswith("Automatic") else upload is not None
    analyze_clicked = st.button("Run AI analysis", type="primary", use_container_width=True, disabled=not ready)

    st.divider()
    st.markdown(
        "<div class='small-muted'><b>Architecture</b><br>"
        "Agent planning → tools → deterministic finance → Gemini reasoning → verification → report</div>",
        unsafe_allow_html=True,
    )


# ---------- Main ----------
st.markdown("<div class='eyebrow'>AI-powered financial research & verification</div>", unsafe_allow_html=True)
st.title("Financial Intelligence Workspace")
st.caption("A traceable workspace where agent decisions, deterministic calculations and LLM interpretation remain explicitly separated.")

if "last_run" not in st.session_state:
    st.session_state.last_run = None

if analyze_clicked:
    store = RunStore()
    run_id = store.start(company)
    started = time.perf_counter()

    try:
        with st.status("Executing agent workflow…", expanded=True) as status:
            st.write("01 · Resolving data requirements")
            ingestion_service = DataIngestionService()

            if source_mode.startswith("Automatic"):
                st.write("02 · Calling financial data tools")
                payload = ingestion_service.from_ticker(ticker, years, include_market, include_news)
                payload["competitor_financials"] = ingestion_service.ingest_competitors(competitor_tickers, years)
            else:
                st.write("02 · Normalizing uploaded financials")
                payload = ingestion_service.from_csv(pd.read_csv(upload))

            st.write("03 · Running research and analysis agents")
            result = build_graph().invoke({
                "rows": payload["rows"],
                "company": company,
                "ticker": ticker if source_mode.startswith("Automatic") else "",
                "competitors": competitor_tickers,
                "competitor_financials": payload.get("competitor_financials", {}),
                "max_revision_attempts": max_revisions,
                "market_data": payload["market_data"],
                "news": payload["news"],
                "ingestion": payload["ingestion"],
            })

            result.update({k: payload[k] for k in ("market_data", "news", "ingestion")})
            result["competitor_financials"] = payload.get("competitor_financials", {})

            st.write("04 · Verifying claim provenance and financial sanity")
            latency = (time.perf_counter() - started) * 1000
            llm_usage = result.get("llm_usage", [])
            estimated_cost = sum(float(x.get("estimated_cost_usd", 0) or 0) for x in llm_usage)

            store.event(run_id, "verification", result.get("verification", {}))
            store.event(run_id, "llm_usage", llm_usage)
            store.finish(
                run_id,
                "failed" if result.get("errors") else "completed",
                latency,
                result.get("errors"),
                result.get("warnings"),
                estimated_cost,
            )
            status.update(
                label="Agent workflow complete" if not result.get("errors") else "Workflow completed with errors",
                state="complete" if not result.get("errors") else "error",
                expanded=False,
            )

        st.session_state.last_run = {
            "result": result, "run_id": run_id, "latency": latency,
            "cost": estimated_cost, "company": company, "ticker": ticker,
        }

    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000
        store.finish(run_id, "failed", latency, [str(exc)], [])
        st.error(f"Workflow failed: {exc}")


run = st.session_state.last_run

if run:
    result = run["result"]
    run_id, latency, estimated_cost = run["run_id"], run["latency"], run["cost"]

    if result.get("errors"):
        st.error("; ".join(result["errors"]))

    claims = claims_from(result)
    verification = VerificationResult(**result["verification"])
    ev = evaluate(verification, claims, latency, estimated_cost)

    render_run_header(result, run["company"], run.get("ticker", ""), verification)

    # Only surface important warnings globally. Optional integration gaps live in Sources.
    important_warnings, _ = warning_summary(result)
    for warning in important_warnings:
        st.warning(warning)

    overview, financials, competitors_tab, ai_tab, sources_tab, report_tab, audit_tab = st.tabs(
        ["Overview", "Financials", "Competitors", "AI Analysis", "Sources", "Report", "Audit"]
    )

    with overview:
        # Agent activity is the primary product surface, so keep it full-width
        # and above the financial outputs.
        render_agent_execution(result, verification, compact=True)

        st.divider()

        # Financial results follow the agent trace in a clear vertical hierarchy:
        # Executive snapshot → Financial trends → Reliability.
        render_overview(result, ev, latency, estimated_cost, verification)

    with financials:
        render_financials(result)

    with competitors_tab:
        render_competitors(result)

    with ai_tab:
        render_ai_insight(result)
        st.divider()
        render_agent_execution(result, verification)

    with sources_tab:
        render_sources(result)

    with report_tab:
        render_report(result, run["company"], run_id, latency, estimated_cost)

    with audit_tab:
        render_audit(result, run_id, latency, estimated_cost, verification)

else:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Start a traceable analysis</div>
      <div class="hero-title">Select a company and run the agent workflow</div>
      <div class="hero-sub">The app will separate data ingestion, deterministic finance calculations, Gemini reasoning and verification so every result can be audited.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Ingest**  \nAnnual fundamentals, peers and optional market context.")
    c2.markdown("**2 · Reason**  \nResearch Agent plans tools; Gemini interprets grounded financial evidence.")
    c3.markdown("**3 · Verify**  \nEvery structured claim is checked before entering the final report.")

    sample = Path("data/samples/example_financials.csv")
    if sample.exists():
        with st.expander("CSV fallback format"):
            st.dataframe(pd.read_csv(sample), use_container_width=True, hide_index=True)
            st.download_button("Download sample CSV", sample.read_bytes(), file_name="example_financials.csv", mime="text/csv")

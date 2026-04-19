"""
Visual HyDE Analysis Dashboard
===============================
Streamlit app for exploring Phase 1 experiment results.

Usage:
    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual HyDE Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_BASE   = ROOT / "results"
IMG_DIR        = ROOT / "data" / "processed" / "chartqa"
GEN_CHART_DIR  = ROOT / "data" / "generated_charts" / "matplotlib"

RETRIEVER_META: dict[str, dict] = {
    "text_direct":             {"label": "Text-Direct",   "color": "#EF4444"},
    "tcd_hyde":                {"label": "TCD-HyDE",      "color": "#F59E0B"},
    "visual_hyde_matplotlib":  {"label": "Visual HyDE",   "color": "#10B981"},
    "hybrid_rrf_a0.5":         {"label": "Hybrid RRF",    "color": "#3B82F6"},
    "colpali":                 {"label": "ColPali",        "color": "#8B5CF6"},
}

QT_ICON = {"trend": "🔵", "comparison": "🟡", "numeric": "🔴"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def label(r: str) -> str:
    return RETRIEVER_META.get(r, {}).get("label", r)

def color(r: str) -> str:
    return RETRIEVER_META.get(r, {}).get("color", "#888888")

def corpus_id_to_path(corpus_id: str) -> Path:
    img_id = corpus_id.removeprefix("chartqa_")
    return IMG_DIR / f"{img_id}.png"

def gen_chart_path(query_id: str) -> Path:
    return GEN_CHART_DIR / f"{query_id}.png"

def load_img(path: Path) -> Image.Image | None:
    if path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None
    return None

def reciprocal_rank(results_list: list[dict], relevant_ids: set[str]) -> float:
    for res in results_list:
        if res["corpus_id"] in relevant_ids:
            return 1.0 / res["rank"]
    return 0.0

def badge(text: str, bg: str, fg: str = "white") -> str:
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 8px;"
        f"border-radius:4px;font-size:12px;font-weight:bold'>{text}</span>"
    )


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def discover_runs() -> list[Path]:
    base = RESULTS_BASE / "chartqa"
    if not base.exists():
        return []
    return sorted([p for p in base.glob("*/results.json")], reverse=True)


@st.cache_data
def load_results_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="📦 Loading ChartQA from cache...")
def load_chartqa(max_queries: int) -> tuple[list[dict], dict[str, Path]]:
    """
    Re-load ChartQA from local HuggingFace cache.
    Returns (query_list, corpus_path_map).
    """
    from visual_hyde.data.loaders import ChartQALoader
    from visual_hyde.config import get_settings

    settings = get_settings()
    loader = ChartQALoader()
    corpus_items, query_items = loader.load(
        split="test",
        max_queries=max_queries,
        cache_dir=settings.paths.raw_dir / "chartqa",
    )

    queries = [
        {
            "id": q.id,
            "text": q.text,
            "query_type": q.query_type.value,
            "relevant_ids": list(q.relevant_ids),
        }
        for q in query_items
    ]
    corpus_map: dict[str, Path] = {
        item.id: corpus_id_to_path(item.id) for item in corpus_items
    }
    return queries, corpus_map


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Visual HyDE")
    st.caption("Experiment Analysis Dashboard")

    runs = discover_runs()
    if not runs:
        st.error("results/chartqa/*/results.json が見つかりません。Phase 1 を実行してください。")
        st.stop()

    selected_run_path = st.selectbox(
        "実験ラン",
        options=runs,
        format_func=lambda p: p.parent.name,
    )

    results = load_results_json(str(selected_run_path))
    meta    = results.get("metadata", {})
    n_q     = meta.get("n_queries", 100)

    all_retrievers = list(results["metrics"].keys())
    active_retrievers = st.multiselect(
        "表示する手法",
        options=all_retrievers,
        default=[r for r in all_retrievers if r != "colpali"],
        format_func=label,
    )
    top_k = st.slider("Top-K 表示枚数", min_value=1, max_value=10, value=5)

    st.divider()
    st.markdown("**MRR@10 サマリー**")
    for r in active_retrievers:
        m = results["metrics"].get(r, {})
        mrr = m.get("mrr@10", 0.0)
        c   = color(r)
        st.markdown(
            f"<span style='color:{c}'>■</span> **{label(r)}** &nbsp; `{mrr:.4f}`",
            unsafe_allow_html=True,
        )


# ── Load query data ───────────────────────────────────────────────────────────

queries, corpus_map = load_chartqa(max_queries=n_q)
query_map: dict[str, dict] = {q["id"]: q for q in queries}


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["📊 Metrics Overview", "🔍 Query Explorer", "🔬 Error Analysis"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Metrics Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Metrics Overview")

    # ── Overall metrics table + bar chart ──────────────────────────────────
    rows = []
    for r in active_retrievers:
        m = results["metrics"].get(r, {})
        rows.append({
            "Retriever":  label(r),
            "MRR@10":     round(m.get("mrr@10",     0.0), 4),
            "Recall@5":   round(m.get("recall@5",   0.0), 4),
            "Recall@10":  round(m.get("recall@10",  0.0), 4),
            "NDCG@10":    round(m.get("ndcg@10",    0.0), 4),
        })
    df_metrics = pd.DataFrame(rows).set_index("Retriever")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("全指標一覧")
        st.dataframe(
            df_metrics.style
                .background_gradient(cmap="Blues", axis=0)
                .format("{:.4f}"),
            use_container_width=True,
            height=220,
        )
    with col_r:
        st.subheader("MRR@10 比較")
        st.bar_chart(df_metrics["MRR@10"], height=220)

    # ── Per-query-type heatmap ─────────────────────────────────────────────
    st.subheader("クエリタイプ別 MRR@10")
    qt_data   = results.get("per_query_type", {})
    qt_labels = ["trend", "comparison", "numeric"]

    qt_rows = []
    for r in active_retrievers:
        row = {"手法": label(r)}
        for qt in qt_labels:
            row[qt.capitalize()] = round(qt_data.get(r, {}).get(qt, {}).get("mrr@10", 0.0), 4)
        qt_rows.append(row)

    df_qt = pd.DataFrame(qt_rows).set_index("手法")
    st.dataframe(
        df_qt.style
            .background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=0.5)
            .format("{:.4f}"),
        use_container_width=True,
        height=200,
    )

    st.caption(
        "🔵 Trend: 傾向・増減クエリ  |  🟡 Comparison: 比較クエリ  |  🔴 Numeric: 数値クエリ"
    )

    # ── Metadata ───────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("クエリ数",   meta.get("n_queries", "—"))
    c2.metric("コーパス数", meta.get("n_corpus",  "—"))
    c3.metric("データセット", meta.get("dataset", "—").upper())
    c4.metric("実験ラン",   selected_run_path.parent.name)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Query Explorer
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Query Explorer")

    # ── Filters ────────────────────────────────────────────────────────────
    fc1, fc2 = st.columns([1, 4])
    with fc1:
        qt_filter = st.selectbox("タイプ絞り込み", ["All", "trend", "comparison", "numeric"])
    filtered_qs = queries if qt_filter == "All" else [q for q in queries if q["query_type"] == qt_filter]

    with fc2:
        sel_qid = st.selectbox(
            "クエリを選択",
            options=[q["id"] for q in filtered_qs],
            format_func=lambda qid: f"[{query_map[qid]['query_type']}] {query_map[qid]['text'][:90]}",
        )

    if not sel_qid:
        st.info("クエリを選択してください")
    else:
        q = query_map[sel_qid]
        relevant_set = set(q["relevant_ids"])
        qt_icon = QT_ICON.get(q["query_type"], "⚪")

        # ── Query info ─────────────────────────────────────────────────────
        st.markdown(
            f"<div style='background:#1E293B;padding:16px;border-radius:8px;margin-bottom:12px'>"
            f"<div style='font-size:18px;color:white;font-weight:bold'>{qt_icon} {q['text']}</div>"
            f"<div style='margin-top:6px'>"
            f"{badge(q['query_type'], '#3B82F6')} &nbsp;"
            f"<span style='color:#94A3B8;font-size:12px'>{sel_qid}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Ground truth + Generated chart ────────────────────────────────
        gt_col, gen_col = st.columns([1, 1])

        with gt_col:
            st.markdown("#### ✅ 正解チャート（Ground Truth）")
            for rel_id in q["relevant_ids"]:
                img = load_img(corpus_id_to_path(rel_id))
                if img:
                    st.image(img, caption=rel_id, use_container_width=True)
                else:
                    st.warning(f"画像なし: {rel_id}")

        with gen_col:
            st.markdown("#### 📊 Visual HyDE 生成チャート（仮説）")
            gen_img = load_img(gen_chart_path(sel_qid))
            if gen_img:
                st.image(gen_img, caption=f"生成チャート ({sel_qid}.png)", use_container_width=True)
            else:
                st.info("生成チャートが見つかりません\n（Visual HyDE を実行すると保存されます）")

        # ── Retriever results grid ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔎 各手法の検索結果 Top-K")

        if not active_retrievers:
            st.warning("サイドバーで手法を1つ以上選択してください")
        else:
            result_cols = st.columns(len(active_retrievers))

            for col, r in zip(result_cols, active_retrievers):
                c = color(r)
                lbl = label(r)

                # Get outputs for this query
                raw_outputs = results.get("raw_outputs", {}).get(r, [])
                qout = next((o for o in raw_outputs if o["query_id"] == sel_qid), None)

                with col:
                    # Header
                    st.markdown(
                        f"<div style='background:{c}22;border-left:4px solid {c};"
                        f"padding:6px 10px;border-radius:4px;margin-bottom:8px'>"
                        f"<strong style='color:{c}'>{lbl}</strong></div>",
                        unsafe_allow_html=True,
                    )

                    if qout is None:
                        st.caption("結果なし")
                        continue

                    rr = reciprocal_rank(qout["results"], relevant_set)
                    st.markdown(
                        f"<div style='text-align:center;color:{c};font-weight:bold;"
                        f"font-size:16px;margin-bottom:6px'>RR = {rr:.3f}</div>",
                        unsafe_allow_html=True,
                    )

                    for res in qout["results"][:top_k]:
                        cid   = res["corpus_id"]
                        score = res["score"]
                        rank  = res["rank"]
                        hit   = cid in relevant_set

                        border = f"3px solid #10B981" if hit else "1px solid #334155"
                        bg     = "#064E3B22" if hit else "transparent"

                        img = load_img(corpus_id_to_path(cid))

                        st.markdown(
                            f"<div style='border:{border};background:{bg};"
                            f"border-radius:6px;padding:4px;margin-bottom:6px'>",
                            unsafe_allow_html=True,
                        )
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.caption("🖼️ 画像なし")

                        hit_str = "✅ HIT" if hit else f"Rank {rank}"
                        st.caption(f"{hit_str} · {score:.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Error Analysis")
    st.caption("ある手法が成功し別の手法が失敗するクエリを探します")

    col_w, col_l = st.columns(2)
    with col_w:
        winner = st.selectbox(
            "✅ 成功している手法",
            options=active_retrievers,
            format_func=label,
            key="ea_winner",
        )
    with col_l:
        default_loser_idx = 1 if len(active_retrievers) > 1 else 0
        loser = st.selectbox(
            "❌ 失敗している手法",
            options=active_retrievers,
            format_func=label,
            key="ea_loser",
            index=default_loser_idx,
        )

    def get_rr(retriever: str, qid: str) -> float:
        raw = results.get("raw_outputs", {}).get(retriever, [])
        out = next((o for o in raw if o["query_id"] == qid), None)
        if out is None:
            return 0.0
        rel = set(query_map[qid]["relevant_ids"]) if qid in query_map else set()
        return reciprocal_rank(out["results"], rel)

    # Build comparison table
    ea_rows = []
    for q in queries:
        qid    = q["id"]
        rr_w   = get_rr(winner, qid)
        rr_l   = get_rr(loser,  qid)
        if rr_w > 0 and rr_l == 0:
            ea_rows.append({
                "_qid":   qid,
                "クエリ": q["text"][:70] + ("…" if len(q["text"]) > 70 else ""),
                "Type":   q["query_type"],
                f"RR ({label(winner)})": f"{rr_w:.3f}",
                f"RR ({label(loser)})":  "0.000",
            })

    if winner == loser:
        st.info("異なる手法を選択してください")
    elif not ea_rows:
        st.info(
            f"**{label(winner)}** が成功し **{label(loser)}** が失敗するクエリは見つかりませんでした"
        )
    else:
        st.markdown(
            f"**{len(ea_rows)} 件** のクエリで "
            f"**{label(winner)}** ✅ 成功 / **{label(loser)}** ❌ 失敗"
        )

        df_ea = pd.DataFrame(ea_rows).drop(columns=["_qid"])
        type_color_map = {"trend": "lightblue", "comparison": "lightyellow", "numeric": "lightpink"}
        st.dataframe(df_ea, use_container_width=True, height=260)

        # Query detail selector
        ea_qid = st.selectbox(
            "詳細を確認するクエリ",
            options=[r["_qid"] for r in ea_rows],
            format_func=lambda qid: query_map[qid]["text"][:80],
            key="ea_detail",
        )

        if ea_qid:
            eq = query_map[ea_qid]
            rel_set = set(eq["relevant_ids"])

            st.markdown(
                f"<div style='background:#1E293B;padding:14px;border-radius:8px;margin:10px 0'>"
                f"<div style='color:white;font-weight:bold'>"
                f"{QT_ICON.get(eq['query_type'],'⚪')} {eq['text']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            img_col, win_col, los_col = st.columns(3)

            with img_col:
                st.markdown("**✅ Ground Truth**")
                for rel_id in eq["relevant_ids"]:
                    img = load_img(corpus_id_to_path(rel_id))
                    if img:
                        st.image(img, caption=rel_id, use_container_width=True)

            with win_col:
                wc = color(winner)
                st.markdown(f"**{label(winner)} ✅ Top-3**")
                raw_w = results.get("raw_outputs", {}).get(winner, [])
                out_w = next((o for o in raw_w if o["query_id"] == ea_qid), None)
                if out_w:
                    for res in out_w["results"][:3]:
                        hit = res["corpus_id"] in rel_set
                        img = load_img(corpus_id_to_path(res["corpus_id"]))
                        border = f"3px solid {wc}" if hit else f"1px solid #334155"
                        st.markdown(
                            f"<div style='border:{border};border-radius:6px;padding:3px;margin-bottom:4px'>",
                            unsafe_allow_html=True,
                        )
                        if img:
                            st.image(img, use_container_width=True)
                        st.caption(f"{'✅' if hit else ''} Rank{res['rank']} · {res['score']:.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)

            with los_col:
                lc = color(loser)
                st.markdown(f"**{label(loser)} ❌ Top-3**")
                raw_l = results.get("raw_outputs", {}).get(loser, [])
                out_l = next((o for o in raw_l if o["query_id"] == ea_qid), None)
                if out_l:
                    for res in out_l["results"][:3]:
                        hit = res["corpus_id"] in rel_set
                        img = load_img(corpus_id_to_path(res["corpus_id"]))
                        border = f"3px solid #EF4444" if hit else f"1px solid #334155"
                        st.markdown(
                            f"<div style='border:{border};border-radius:6px;padding:3px;margin-bottom:4px'>",
                            unsafe_allow_html=True,
                        )
                        if img:
                            st.image(img, use_container_width=True)
                        st.caption(f"{'✅' if hit else ''} Rank{res['rank']} · {res['score']:.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)

            # Show generated chart for context
            gen = load_img(gen_chart_path(ea_qid))
            if gen:
                st.markdown("---")
                st.markdown("**📊 Visual HyDE が生成した仮説チャート**")
                genc = st.columns([1, 2, 1])
                with genc[1]:
                    st.image(gen, caption="仮説チャート", use_container_width=True)

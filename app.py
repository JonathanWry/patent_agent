from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.claim_analysis import ClaimLimitation, VerificationResult, extract_evidence_for_candidate
from src.data_loader import combine_patent_pools, load_hf_par4pc_patent_pool, load_par4pc_case, load_unique_patent_pool
from src.free_text_qa import gather_query_evidence, heuristic_rag_answer, verify_rag_answer_heuristic
from src.graph import run_graph
from src.llm_tools import answer_query_with_rag, openai_available, plan_turn_llm, verify_rag_answer_llm
from src.patent_rerank import rank_patent_pool_hybrid_coverage
from src.patent_rerank import rank_patent_pool_patent_specialized
from src.persistent_index import (
    index_exists,
    load_persistent_candidates,
    load_persistent_manifest,
    search_persistent_index,
)
from src.retrieval import (
    _sentence_transformer_model,
    rank_patent_pool_bm25,
    rank_patent_pool_cross_encoder,
    rank_patent_pool_local_embeddings,
)
from src.query_planner import classify_turn, enrich_query_with_context


DEFAULT_DATA_DIR = Path("../PANORAMA/data/benchmark/par4pc")
DEFAULT_INDEX_DIR = Path("data/indexes/par4pc_patentsberta_demo")
DEFAULT_EMBEDDING_MODEL = "AI-Growth-Lab/PatentSBERTa"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FREE_TEXT_PRIMARY_RETRIEVAL_OPTIONS = ["bm25", "local-embedding"]
BENCHMARK_PRIMARY_RETRIEVAL_OPTIONS = ["local-embedding", "patent-specialized", "bm25"]
FREE_TEXT_EXPERIMENTAL_OPTIONS = ["patent-specialized", "hybrid-coverage", "local-cross-encoder"]
BENCHMARK_EXPERIMENTAL_OPTIONS = ["patent-specialized", "hybrid-coverage", "local-cross-encoder", "openai-embedding", "llm-rerank"]
DEFAULT_FREE_TEXT = (
    "1. A method for leveraging social networks in physical gatherings, the method comprising: "
    "generating, by one or more computer processors, a profile for each participant of one or more "
    "participants at a physical gathering; receiving, by one or more computer processors, data from "
    "one or more computer systems associated with the one or more participants of the physical gathering, "
    "wherein each participant of the one or more participants is associated with a computer system; "
    "receiving, by one or more computer processors, a request for information from a computer system "
    "associated with a first participant of the one or more participants of the physical gathering; "
    "determining, by one or more computer processors, whether the first participant has access to the "
    "information requested based on the profile for the first participant; responsive to determining that "
    "the first participant has access to the information requested, analyzing, by one or more computer "
    "processors, the data received from the one or more computer systems associated with the one or more "
    "participants of the physical gathering to identify data to provide to the first participant to "
    "fulfill the request for information; and providing, by one or more computer processors, the "
    "identified data to the computer system associated with the first participant of the physical gathering."
)


@st.cache_data
def list_case_paths(data_dir: str) -> list[str]:
    return [str(path) for path in sorted(Path(data_dir).glob("par4pc_*.json"))]


@st.cache_data
def preview_case(case_path: str):
    return load_par4pc_case(case_path)


@st.cache_data
def load_pool(
    data_dir: str,
    pool_source: str,
    hub_rows_per_split: int,
):
    local_pool = load_unique_patent_pool(data_dir)
    if pool_source == "Local sample pool":
        return local_pool

    hf_limit = None if hub_rows_per_split <= 0 else hub_rows_per_split
    hub_pool = load_hf_par4pc_patent_pool(max_rows_per_split=hf_limit)
    if pool_source == "Hub PAR4PC pool":
        return hub_pool
    return combine_patent_pools(local_pool, hub_pool)


def warm_up_search_backend(
    data_dir: str,
    pool_source: str,
    hub_rows_per_split: int,
    retrieval_method: str,
    embedding_model: str,
    index_dir: str,
) -> tuple[int, str]:
    if pool_source == "Persistent local index":
        manifest = load_persistent_manifest(index_dir)
        model_name = ""
        if retrieval_method == "local-embedding":
            model_name = embedding_model or manifest.embedding_model
            _sentence_transformer_model(model_name)
        return manifest.patent_count, model_name

    pool = load_pool(data_dir, pool_source, hub_rows_per_split)
    model_name = ""
    if retrieval_method == "local-embedding":
        model_name = embedding_model or DEFAULT_EMBEDDING_MODEL
        _sentence_transformer_model(model_name)
    return len(pool), model_name


def ranked_table(ranked) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": index,
                "label": getattr(result, "letter", ""),
                "patent_id": result.patent_id,
                "score": round(result.score, 3),
                "title": result.title,
            }
            for index, result in enumerate(ranked, start=1)
        ]
    )


def limitation_table(limitations) -> pd.DataFrame:
    return pd.DataFrame(
        [{"label": item.label, "limitation": item.text} for item in limitations]
    )


def claim_chart_table(chart) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "limitation": row.limitation_label,
                "candidate": f"{row.candidate_letter} ({row.patent_id})",
                "source": row.source,
                "score": round(row.score, 3),
                "verification": row.verification,
                "reason": row.verification_reason,
                "evidence": row.evidence,
            }
            for row in chart
        ]
    )


def query_evidence_table(snippets) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patent_id": snippet.patent_id,
                "title": snippet.title,
                "source": snippet.source,
                "retrieval_score": round(snippet.retrieval_score, 3),
                "segment_score": round(snippet.segment_score, 3),
                "evidence": snippet.evidence,
            }
            for snippet in snippets
        ]
    )


def search_patents(
    query_text: str,
    pool,
    retrieval_method: str,
    embedding_model: str,
    reranker_model: str,
    top_k: int,
    pool_source: str,
    index_dir: str,
    force_subset: bool = False,
    llm_model: str = "",
    use_llm_retrieval_decompose: bool = False,
    use_llm_query_expansion: bool = False,
):
    if retrieval_method == "patent-specialized":
        if pool_source == "Persistent local index" and not force_subset:
            initial = search_persistent_index(
                query_text,
                index_dir=index_dir,
                top_k=max(top_k * 4, 12),
                embedding_model=embedding_model,
            )
            pool = [item.candidate for item in initial]
        elif pool_source == "Persistent local index":
            pool = load_persistent_candidates(index_dir)
        return rank_patent_pool_patent_specialized(
            query_text,
            pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
            use_query_expansion=True,
            use_llm_expansion=use_llm_query_expansion,
            use_llm_decompose=use_llm_retrieval_decompose,
            llm_model=llm_model,
        )
    if retrieval_method == "hybrid-coverage":
        if pool_source == "Persistent local index" and not force_subset:
            pool = load_persistent_candidates(index_dir)
        return rank_patent_pool_hybrid_coverage(
            query_text,
            pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        )
    if pool_source == "Persistent local index" and retrieval_method == "local-embedding" and not force_subset:
        return search_persistent_index(
            query_text,
            index_dir=index_dir,
            top_k=top_k,
            embedding_model=embedding_model,
        )
    if pool_source == "Persistent local index" and not force_subset:
        pool = load_persistent_candidates(index_dir)
    if retrieval_method == "local-embedding":
        return rank_patent_pool_local_embeddings(
            query_text,
            pool,
            top_k=top_k,
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        )
    if retrieval_method == "local-cross-encoder":
        return rank_patent_pool_cross_encoder(
            query_text,
            pool,
            top_k=top_k,
            reranker_model=reranker_model or DEFAULT_RERANKER_MODEL,
        )
    return rank_patent_pool_bm25(query_text, pool, top_k=top_k)


def free_text_summary(query_text: str, ranked) -> tuple[str, pd.DataFrame]:
    rows = []
    summary_lines = [
        "I searched the indexed patent pool and ranked the closest patents to your query.",
        "",
    ]
    for index, result in enumerate(ranked, start=1):
        limitation = ClaimLimitation(label="Q1", text=query_text)
        evidence = extract_evidence_for_candidate(limitation, result.candidate)
        summary_lines.append(
            f"{index}. `{result.patent_id}` {result.title} "
            f"(source: {evidence.source}, score={result.score:.3f})"
        )
        rows.append(
            {
                "rank": index,
                "patent_id": result.patent_id,
                "title": result.title,
                "score": round(result.score, 3),
                "source": evidence.source,
                "evidence": evidence.evidence,
            }
        )
    return "\n".join(summary_lines), pd.DataFrame(rows)


def generate_free_text_answer(
    query_text: str,
    ranked,
    llm_model: str,
    use_llm_answer: bool,
    plan=None,
):
    snippets = gather_query_evidence(query_text, ranked, snippets_per_patent=2)
    warnings: list[str] = []
    if use_llm_answer:
        if openai_available():
            response = answer_query_with_rag(query_text, snippets, model=llm_model)
            answer = response.answer
            if response.insufficiency_note:
                answer += f"\n\nNote: {response.insufficiency_note}"
            return answer, snippets, warnings
        warnings.append("OPENAI_API_KEY not set; used heuristic grounded answer.")
    return heuristic_rag_answer(query_text, ranked, snippets, plan=plan), snippets, warnings


def verify_free_text_answer(
    answer_text: str,
    snippets,
    llm_model: str,
    use_llm_answer_verification: bool,
) -> tuple[VerificationResult, list[str]]:
    warnings: list[str] = []
    if use_llm_answer_verification:
        if openai_available():
            return verify_rag_answer_llm(answer_text, snippets, model=llm_model), warnings
        warnings.append("OPENAI_API_KEY not set; used heuristic answer verification.")
    return verify_rag_answer_heuristic(answer_text, snippets), warnings


def render_benchmark_mode(
    data_dir: str,
    top_k: int,
    retrieval_method: str,
    use_llm_decompose: bool,
    use_llm_verify: bool,
    llm_model: str,
    embedding_model: str,
    reranker_model: str,
) -> None:
    st.info(
        "Benchmark Analysis is the labeled PAR4PC evaluation path. "
        "Use `local-embedding` as the stable default. Compare against `patent-specialized` as an experimental patent-aware reranker."
    )
    case_paths = list_case_paths(data_dir)
    if not case_paths:
        st.error("No PAR4PC JSON files found.")
        return

    case_path = st.selectbox(
        "Benchmark case",
        options=case_paths,
        format_func=lambda path: Path(path).name,
    )
    case = preview_case(case_path)

    st.chat_message("user").write(
        f"Analyze this claim against the benchmark candidate patents:\n\n{case.target_claim}"
    )

    run = st.button("Analyze Benchmark Case", type="primary")
    if not run:
        st.info("Choose a benchmark case and run the agent.")
        return

    with st.spinner("Running patent agent..."):
        result = run_graph(
            case_path=case_path,
            top_k=top_k,
            use_llm_decompose=use_llm_decompose,
            use_llm_verify=use_llm_verify,
            retrieval_method=retrieval_method,
            llm_model=llm_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )

    ranked = result["ranked"]
    limitations = result["limitations"]
    chart = result["claim_chart"]
    report = result["report"]
    predicted = [item.letter for item in ranked]
    gold = set(result["case"].gold_answers)
    recall = len(set(predicted) & gold) / len(gold) if gold else 0.0

    summary = (
        f"I ranked the candidate prior art for claim {case.claim_number}. "
        f"Top-{top_k}: {', '.join(predicted)}. "
        f"Gold: {', '.join(case.gold_answers) or 'N/A'}. "
        f"Hit@1={'yes' if set(predicted[:1]) & gold else 'no'}, "
        f"Hit@{top_k}={'yes' if set(predicted[:top_k]) & gold else 'no'}, "
        f"Recall@{top_k}={recall:.3f}."
    )
    st.chat_message("assistant").write(summary)

    tab_summary, tab_evidence, tab_report = st.tabs(["Summary", "Evidence", "Report"])
    with tab_summary:
        meta_cols = st.columns(4)
        meta_cols[0].metric("Application", case.application_number)
        meta_cols[1].metric("Claim", str(case.claim_number))
        meta_cols[2].metric("Gold", ", ".join(case.gold_answers) or "N/A")
        meta_cols[3].metric("Candidates", str(len(case.candidates)))
        st.dataframe(ranked_table(ranked), use_container_width=True, hide_index=True)
        st.dataframe(limitation_table(limitations), use_container_width=True, hide_index=True)
    with tab_evidence:
        st.dataframe(
            claim_chart_table(chart),
            use_container_width=True,
            hide_index=True,
            column_config={
                "evidence": st.column_config.TextColumn("evidence", width="large"),
                "reason": st.column_config.TextColumn("reason", width="medium"),
            },
        )
    with tab_report:
        st.markdown(report)
        st.download_button(
            "Download Markdown Report",
            data=report,
            file_name=f"{Path(case_path).stem}_report.md",
            mime="text/markdown",
        )

    if result.get("warnings"):
        st.warning("\n".join(result["warnings"]))


def render_free_text_mode(
    data_dir: str,
    top_k: int,
    retrieval_method: str,
    embedding_model: str,
    reranker_model: str,
    llm_model: str,
    use_llm_answer: bool,
    use_llm_planner: bool,
    use_llm_retrieval_decompose: bool,
    use_llm_query_expansion: bool,
    use_llm_answer_verification: bool,
    pool_source: str,
    hub_rows_per_split: int,
    index_dir: str,
) -> None:
    st.info(
        "Free-text Search is the exploratory patent QA mode. "
        "For larger patent pools, start with `bm25` for speed and stable recall. "
        "Treat `patent-specialized` here as a research path, not the default."
    )
    pool = None
    if pool_source == "Persistent local index":
        if not index_exists(index_dir):
            st.error(
                "Persistent index not found. Build it first with "
                f"`python -m src.build_patent_index --index-dir \"{index_dir}\"`."
            )
            return
        manifest = load_persistent_manifest(index_dir)
        st.caption(
            f"Persistent local index: {manifest.patent_count} patents | model={manifest.embedding_model}"
        )
    else:
        if pool_source != "Local sample pool":
            st.info("The first Hub-backed search downloads and caches PAR4PC parquet files locally.")
        pool = load_pool(data_dir, pool_source, hub_rows_per_split)
        st.caption(f"{pool_source}: {len(pool)} patents")

    if "free_text_messages" not in st.session_state:
        st.session_state.free_text_messages = [
            {
                "role": "assistant",
                "content": (
                    "Send patent-related text, a claim, or an invention description. "
                    "I will search the indexed patent pool, rank the closest patents, "
                    "and show supporting evidence snippets."
                ),
            }
        ]
    if "free_text_agent_state" not in st.session_state:
        st.session_state.free_text_agent_state = {
            "last_ranked": [],
            "last_snippets": [],
            "last_plan": None,
            "working_patents": [],
            "last_query": "",
        }

    for message in st.session_state.free_text_messages:
        st.chat_message(message["role"]).write(message["content"])

    if st.button("Use example query"):
        st.session_state.free_text_pending_query = DEFAULT_FREE_TEXT

    query = st.chat_input("Ask for related patents")
    if not query:
        query = st.session_state.pop("free_text_pending_query", "")
    if not query:
        return

    st.session_state.free_text_messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    agent_state = st.session_state.free_text_agent_state
    if use_llm_planner and openai_available():
        plan = plan_turn_llm(
            query_text=query,
            has_context=bool(agent_state["working_patents"]),
            previous_titles=[result.title for result in agent_state["last_ranked"][:5]],
            model=llm_model or None,
        )
    else:
        plan = classify_turn(query, has_context=bool(agent_state["working_patents"]))
    effective_query = enrich_query_with_context(query, plan, agent_state["last_ranked"])

    with st.spinner("Searching related patents..."):
        if plan.action == "retrieve_new":
            ranked = search_patents(
                query_text=effective_query,
                pool=pool,
                retrieval_method=retrieval_method,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
                top_k=top_k,
                pool_source=pool_source,
                index_dir=index_dir,
                llm_model=llm_model,
                use_llm_retrieval_decompose=use_llm_retrieval_decompose,
                use_llm_query_expansion=use_llm_query_expansion,
            )
            working_patents = [result.candidate for result in ranked]
        elif plan.action == "reuse_context":
            ranked = agent_state["last_ranked"]
            working_patents = agent_state["working_patents"]
        else:
            working_patents = agent_state["working_patents"]
            ranked = search_patents(
                query_text=query,
                pool=working_patents,
                retrieval_method=retrieval_method,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
                top_k=min(top_k, len(working_patents) or top_k),
                pool_source=pool_source,
                index_dir=index_dir,
                force_subset=True,
                llm_model=llm_model,
                use_llm_retrieval_decompose=use_llm_retrieval_decompose,
                use_llm_query_expansion=use_llm_query_expansion,
            )
        answer, snippets, warnings = generate_free_text_answer(
            query_text=query,
            ranked=ranked,
            llm_model=llm_model,
            use_llm_answer=use_llm_answer,
            plan=plan,
        )
        answer_verification, verification_warnings = verify_free_text_answer(
            answer_text=answer,
            snippets=snippets,
            llm_model=llm_model,
            use_llm_answer_verification=use_llm_answer_verification,
        )
        warnings.extend(verification_warnings)
        summary, evidence_df = free_text_summary(query, ranked)

    agent_state["last_ranked"] = ranked
    agent_state["last_snippets"] = snippets
    agent_state["last_plan"] = plan
    agent_state["working_patents"] = working_patents
    agent_state["last_query"] = query

    st.session_state.free_text_messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
    if warnings:
        st.warning("\n".join(warnings))

    with st.expander("Agent Decision"):
        st.write(f"Intent: `{plan.intent}`")
        st.write(f"Action: `{plan.action}`")
        st.write(f"Reason: {plan.reason}")
        if effective_query != query:
            st.write("Context-enriched query:")
            st.code(effective_query)
    with st.expander("Answer Verification"):
        st.write(f"Status: `{answer_verification.status}`")
        st.write(f"Reason: {answer_verification.reason}")

    with st.expander("Ranked Patents", expanded=True):
        st.dataframe(ranked_table(ranked), use_container_width=True, hide_index=True)

    with st.expander("Supporting Evidence", expanded=True):
        st.dataframe(
            query_evidence_table(snippets),
            use_container_width=True,
            hide_index=True,
            column_config={
                "evidence": st.column_config.TextColumn("evidence", width="large"),
                "title": st.column_config.TextColumn("title", width="medium"),
            },
        )
    with st.expander("Retrieval Summary"):
        st.dataframe(
            evidence_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "evidence": st.column_config.TextColumn("evidence", width="large"),
            },
        )
        st.markdown(summary)


def main() -> None:
    st.set_page_config(page_title="Patent Prior-Art Agent", layout="wide")
    st.title("Patent Prior-Art Agent")
    st.caption("Search related patents from text, or inspect benchmark claim analysis.")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", options=["Free-text Search", "Benchmark Analysis"], index=0)
        if mode == "Free-text Search":
            st.caption("Recommended retrieval: `bm25` on larger pools. Use experimental methods only for comparison.")
        else:
            st.caption("Recommended retrieval: `local-embedding` as the current stable benchmark default; compare `patent-specialized` experimentally.")
        data_dir = st.text_input("PAR4PC data directory", value=str(DEFAULT_DATA_DIR))
        top_k = st.slider("Top-k prior art", min_value=1, max_value=8, value=3)
        show_experimental = st.checkbox("Show experimental retrieval methods", value=False)
        retrieval_options = list(
            FREE_TEXT_PRIMARY_RETRIEVAL_OPTIONS if mode == "Free-text Search" else BENCHMARK_PRIMARY_RETRIEVAL_OPTIONS
        )
        experimental_options = (
            FREE_TEXT_EXPERIMENTAL_OPTIONS
            if mode == "Free-text Search"
            else BENCHMARK_EXPERIMENTAL_OPTIONS
        )
        if show_experimental:
            retrieval_options.extend(experimental_options)
        retrieval_method = st.selectbox(
            "Retrieval method",
            options=retrieval_options,
            index=0,
        )
        if retrieval_method in experimental_options:
            st.info("This retrieval path is kept for ablation. It is not the recommended demo default.")
        pool_source = st.selectbox(
            "Free-text patent pool",
            options=["Persistent local index", "Local sample pool", "Hub PAR4PC pool", "Combined"],
            index=0,
            help="Free-text mode searches this pool. Benchmark mode still uses the selected case's A-H candidates.",
        )
        index_dir = st.text_input("Persistent index directory", value=str(DEFAULT_INDEX_DIR))
        hub_rows_per_split = st.number_input(
            "Hub rows per split for free-text pool",
            min_value=0,
            max_value=54028,
            value=500,
            step=250,
            help="0 loads full train/validation/test splits. Smaller values start faster.",
        )

        with st.expander("Advanced Settings"):
            use_llm_answer = st.checkbox("Use LLM grounded answer", value=False)
            use_llm_planner = st.checkbox("Use LLM planner", value=False)
            use_llm_retrieval_decompose = st.checkbox("Use LLM retrieval decomposition", value=False)
            use_llm_query_expansion = st.checkbox("Use LLM query expansion", value=False)
            use_llm_answer_verification = st.checkbox("Use LLM answer verification", value=False)
            use_llm_decompose = st.checkbox("Use LLM claim decomposition")
            use_llm_verify = st.checkbox("Use LLM evidence verification")
            llm_model = st.text_input("LLM model", value="gpt-4o-mini")
            embedding_model = st.text_input(
                "Embedding model override",
                value="",
                help="Leave blank for the method default.",
            )
            reranker_model = st.text_input("Local reranker model", value=DEFAULT_RERANKER_MODEL)

        if mode == "Free-text Search" and st.button("Preload search backend"):
            with st.spinner("Loading search pool and model cache..."):
                pool_size, model_name = warm_up_search_backend(
                    data_dir=data_dir,
                    pool_source=pool_source,
                    hub_rows_per_split=int(hub_rows_per_split),
                    retrieval_method=retrieval_method,
                    embedding_model=embedding_model,
                    index_dir=index_dir,
                )
            detail = f"Loaded {pool_size} patents"
            if model_name:
                detail += f" and warmed {model_name}"
            st.success(detail)

        needs_openai = (
            use_llm_answer
            or use_llm_planner
            or use_llm_retrieval_decompose
            or use_llm_query_expansion
            or use_llm_answer_verification
            or use_llm_decompose
            or use_llm_verify
            or retrieval_method in {"openai-embedding", "llm-rerank"}
        )
        if needs_openai and not openai_available():
            st.warning("OPENAI_API_KEY is not set. OpenAI-dependent paths will fall back.")

    if mode == "Benchmark Analysis":
        render_benchmark_mode(
            data_dir=data_dir,
            top_k=top_k,
            retrieval_method=retrieval_method,
            use_llm_decompose=use_llm_decompose,
            use_llm_verify=use_llm_verify,
            llm_model=llm_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )
    else:
        render_free_text_mode(
            data_dir=data_dir,
            top_k=top_k,
            retrieval_method=retrieval_method,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            llm_model=llm_model,
            use_llm_answer=use_llm_answer,
            use_llm_planner=use_llm_planner,
            use_llm_retrieval_decompose=use_llm_retrieval_decompose,
            use_llm_query_expansion=use_llm_query_expansion,
            use_llm_answer_verification=use_llm_answer_verification,
            pool_source=pool_source,
            hub_rows_per_split=int(hub_rows_per_split),
            index_dir=index_dir,
        )


if __name__ == "__main__":
    main()

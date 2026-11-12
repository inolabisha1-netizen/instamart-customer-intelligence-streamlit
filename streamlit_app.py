
from pathlib import Path

import pandas as pd
import streamlit as st

from src.policy_rag import PolicyRAG

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Instamart AI Portfolio",
    page_icon="🛒",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS = BASE_DIR / "outputs"


# ---------------------------------------
# DATA LOADING
# ---------------------------------------

@st.cache_data
def load_segments():
    return pd.read_csv(OUTPUTS / "customer_segments.csv")


@st.cache_data
def load_retention():
    return pd.read_csv(OUTPUTS / "retention_list.csv")


@st.cache_data
def load_tickets():
    return pd.read_csv(OUTPUTS / "triage_results.csv")


segments = load_segments()
retention = load_retention()
tickets = load_tickets()

@st.cache_resource
def load_rag():
    return PolicyRAG()

rag = load_rag()


# ---------------------------------------
# APPLICATION HEADER
# ---------------------------------------

st.title("🛒 Instamart Customer Intelligence & Support Copilot")

st.caption(
    "AI/ML Portfolio | Customer Segmentation • Retention Analytics "
    "• AI Support Automation"
)

st.info(
    "Interactive portfolio demonstration using saved pipeline outputs. "
    "The dashboard does not retrain models or make live AI predictions."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Customer Intelligence",
    "📈 Retention Analytics",
    "🎧 Support Copilot",
    "🤖 AI Policy Assistant"
])


# ---------------------------------------
# TAB 1 — CUSTOMER INTELLIGENCE
# ---------------------------------------

with tab1:

    st.header("Customer Segmentation Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", f"{len(segments):,}")
    col2.metric(
        "Premium Customers",
        f"{int(segments['is_premium'].sum()):,}"
    )
    col3.metric(
        "Customer Segments",
        segments["segment_name"].nunique()
    )

    st.subheader("Customer Distribution by Segment")

    segment_counts = segments["segment_name"].value_counts()
    st.bar_chart(segment_counts)

    st.subheader("Explore Customer Segments")

    segment_options = sorted(
        segments["segment_name"].unique()
    )

    selected_segments = st.multiselect(
        "Select customer segments",
        options=segment_options,
        default=segment_options
    )

    filtered = segments[
        segments["segment_name"].isin(selected_segments)
    ]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Segment Characteristics")

    if not filtered.empty:
        profile = (
            filtered.groupby("segment_name")
            .agg(
                customers=("customer_id", "count"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean")
            )
            .round(2)
        )

        st.dataframe(
            profile,
            use_container_width=True
        )
    else:
        st.info(
            "Select at least one customer segment "
            "to see its characteristics."
        )

    st.caption(
        "Customer segmentation uses saved RFM-based "
        "clustering results from the capstone pipeline."
    )


# ---------------------------------------
# TAB 2 — RETENTION ANALYTICS
# ---------------------------------------

with tab2:

    st.header("Customer Retention Analytics")

    st.caption(
        "Model-generated retention shortlist showing "
        "estimated customer lapse probabilities."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers Shortlisted",
        len(retention)
    )

    col2.metric(
        "Average Lapse Probability",
        f"{retention['lapse_probability'].mean():.1%}"
    )

    col3.metric(
        "Highest Lapse Probability",
        f"{retention['lapse_probability'].max():.1%}"
    )

    st.subheader("Lapse Probability by Customer")

    risk_chart = retention.set_index(
        "customer_id"
    )["lapse_probability"]

    st.bar_chart(risk_chart)

    st.subheader("Retention Priority List")

    sorted_retention = retention.sort_values(
        "lapse_probability",
        ascending=False
    )

    st.dataframe(
        sorted_retention,
        use_container_width=True,
        hide_index=True,
        column_config={
            "lapse_probability": st.column_config.ProgressColumn(
                "Estimated Lapse Probability",
                min_value=0.0,
                max_value=1.0,
                format="percent"
            )
        }
    )

    st.info(
        "These 10 customers form a saved retention shortlist. "
        "The displayed probabilities are model estimates, "
        "not guaranteed future outcomes."
    )


# ---------------------------------------
# TAB 3 — SUPPORT COPILOT
# ---------------------------------------

with tab3:

    st.header("AI Support Ticket Analytics")

    classified = tickets["status"].isin([
        "classified_json",
        "classified_keyword"
    ]).sum()

    human_review = tickets[
        "route"
    ].eq("human_review").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Tickets Processed",
        len(tickets)
    )

    col2.metric(
        "Classified Tickets",
        int(classified)
    )

    col3.metric(
        "Human Review Tickets",
        int(human_review)
    )

    st.subheader("Ticket Routing Distribution")

    st.bar_chart(
        tickets["route"].value_counts()
    )

    st.subheader("Ticket Priority Distribution")

    st.bar_chart(
        tickets["priority"].value_counts()
    )

    st.subheader("Explore Support Tickets")

    route_options = sorted(
        tickets["route"].unique()
    )

    selected_routes = st.multiselect(
        "Select ticket routes",
        options=route_options,
        default=route_options
    )

    filtered_tickets = tickets[
        tickets["route"].isin(selected_routes)
    ].copy()

    filtered_tickets["category"] = (
        filtered_tickets["category"].fillna(
            "Unclassified"
        )
    )

    st.dataframe(
        filtered_tickets,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Support tickets were processed by the saved "
        "AI triage pipeline. The dashboard displays "
        "recorded results rather than generating "
        "new predictions."
    )

# ---------------------------------------
# TAB 4 — AI POLICY ASSISTANT
# ---------------------------------------

with tab4:

    st.header("AI Policy Assistant")

    st.caption(
        "Ask questions about Instamart policies. "
        "The assistant uses TF-IDF retrieval and a simulated LLM."
    )

    st.info(
        "Portfolio demonstration using local policy documents "
        "and a mock LLM. No external AI API is called."
    )


    with st.form("policy_question_form"):
        question = st.text_input(
            "Ask a policy question",
            placeholder="What is the refund policy?"
        )

        submitted = st.form_submit_button(
            "Get Policy Answer",
            type="primary"
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a policy question.")
        else:
            with st.spinner("Searching policy documents..."):
                result = rag.answer(question.strip())

            st.subheader("Assistant Answer")

            answer_text = str(result.get("answer") or "").strip()

            if answer_text:
                st.info(answer_text)
            else:
                st.warning(
                    "I could not find that information "
                    "in the policy documents."
                )

            st.subheader("Retrieved Policy Sources")

            for hit in result["retrieved"]:
                with st.expander(
                    f"{hit['source']} | "
                    f"Similarity: {hit['score']:.3f}"
                ):
                    st.write(hit["text"])

            st.caption(
                "Similarity scores indicate lexical relevance, "
                "not confidence that the answer is correct."
            )

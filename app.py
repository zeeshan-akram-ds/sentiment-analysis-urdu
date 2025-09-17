import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
import plotly.express as px

# ----------------------
# Streamlit UI Settings
# ----------------------
st.set_page_config(
    page_title="Urdu & Roman Urdu Sentiment Classifier",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #4B0082;'>Urdu + Roman Urdu Sentiment Analysis</h1>
    <p style='text-align: center; font-size:18px; color:#555;'>Enter text in Urdu or Roman Urdu and get sentiment predictions from a fine-tuned transformer model.</p>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Load Model & Tokenizer
# ----------------------
@st.cache_resource
def load_model():
    model_path = "./final_model"  
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return clf

clf = load_model()

# ----------------------
# Helper Functions
# ----------------------
def decide_sentiment(sorted_res, threshold=0.15):
    """Return final sentiment, adding 'Mixed' if top two are close."""
    top, second = sorted_res[0], sorted_res[1]
    if abs(top['score'] - second['score']) < threshold:
        return "Mixed"
    return top['label'].capitalize()

def split_sentences(text, max_words=20):
    """
    Split Urdu/Roman Urdu text into smaller sentences.
    - Uses punctuation (. ? ! ۔ , ،)
    - Splits on connectors (اور, لیکن, مگر, or, but)
    - If still too long, chunk into max_words pieces
    """
    parts = re.split(r'[.!?،,۔]| اور | لیکن | مگر | or | but ', text)
    parts = [p.strip() for p in parts if p.strip()]

    final_parts = []
    for p in parts:
        words = p.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                final_parts.append(" ".join(words[i:i+max_words]))
        else:
            final_parts.append(p)
    return final_parts

def plot_scores_plotly(results, title="Sentiment Distribution"):
    """Compact bar chart using Plotly with inside labels."""
    labels = [r["label"].capitalize() for r in results]
    scores = [round(r["score"], 4) for r in results]

    fig = px.bar(
        x=labels,
        y=scores,
        text=scores,
        title=title,
        color=labels,
        color_discrete_sequence=["#1f77b4", "#2ca02c", "#d62728"]
    )
    fig.update_traces(textposition="inside", textfont_size=12)
    fig.update_layout(
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis_title="Sentiment",
        title_font=dict(size=14),
        margin=dict(l=10, r=10, t=30, b=10),
        height=250,
        showlegend=False
    )
    return fig

# ----------------------
# User Input Section
# ----------------------
user_text = st.text_area(
    "Write your text here:", 
    height=150, 
    placeholder="Daraz ki packaging weak thi"
)

force_split = st.checkbox("Analyze sentence by sentence (even for short reviews)", value=False)

if st.button("Analyze Sentiment", type="primary"):
    if user_text.strip():
        words = user_text.split()
        # automatic long review condition
        long_text = len(words) > 25  

        if long_text or force_split:
            st.info("Sentence-level analysis mode enabled...")

            sentences = split_sentences(user_text)
            all_labels = []

            for idx, sent in enumerate(sentences, 1):
                with st.spinner(f"Analyzing sentence {idx}/{len(sentences)}..."):
                    results = clf(sent)
                    sorted_res = sorted(results[0], key=lambda x: x['score'], reverse=True)

                # left: text + results, right: chart
                cols = st.columns([2, 1])  

                with cols[0]:
                    st.markdown(f"### Sentence {idx}")
                    st.markdown(f"**Text:** {sent}")
                    for r in sorted_res:
                        st.markdown(f"- **{r['label'].capitalize()}** → {r['score']:.4f}")
                    final_sent = decide_sentiment(sorted_res)
                    st.success(f"Final Sentiment: {final_sent}")
                    all_labels.append(final_sent)

                with cols[1]:
                    fig = plot_scores_plotly(sorted_res, title=f"Sentence {idx}")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

            # Aggregate final result
            if all_labels:
                majority = max(set(all_labels), key=all_labels.count)
                st.subheader(f"Overall Review Sentiment: **{majority}**")

                # Show simple distribution text instead of chart
                dist = {lab: all_labels.count(lab) for lab in set(all_labels)}
                st.markdown("### Sentiment Breakdown")
                for label, count in dist.items():
                    st.markdown(f"- **{label}** → {count} sentences")

        else:
            with st.spinner("Analyzing..."):
                results = clf(user_text)

            sorted_res = sorted(results[0], key=lambda x: x['score'], reverse=True)

            cols = st.columns([2, 1])

            with cols[0]:
                st.subheader("Prediction Scores")
                for r in sorted_res:
                    st.markdown(f"- **{r['label'].capitalize()}** → {r['score']:.4f}")
                final_sentiment = decide_sentiment(sorted_res)
                if final_sentiment == "Mixed":
                    st.warning(f"**Final Sentiment: Mixed (conflicting opinions detected)**")
                else:
                    st.success(f"**Final Sentiment: {final_sentiment}**")

            with cols[1]:
                fig = plot_scores_plotly(sorted_res, title="Sentiment Proportion")
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Please enter some text first.")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size:14px; color:#888;'>Built with Hugging Face 🤗 Transformers & Streamlit 🚀</p>", 
    unsafe_allow_html=True
)

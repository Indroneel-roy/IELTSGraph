import streamlit as st
from main import run_ielts_scorer

st.set_page_config(page_title="IELTS Essay Scorer", page_icon="📝")

st.title("🧠 IELTS Essay Scorer (AI-powered)")
st.write("Paste your essay below and get instant band-like feedback!")

essay = st.text_area("✍️ Enter your IELTS essay here:", height=250)

if st.button("Evaluate Essay"):
    if essay.strip():
        with st.spinner("Evaluating your essay..."):
            result = run_ielts_scorer(essay)
        
        st.subheader("📊 Scores")
        scores = result["individual_scores"]
        st.write(f"**Average Score:** {result['avg_score']:.2f}/9")

        st.write("---")
        st.write(f"**Language:** {scores[0]}/9")
        st.info(result["language_feedback"])
        st.write(f"**Coherence:** {scores[1]}/9")
        st.info(result["coherence_feedback"])
        st.write(f"**Lexical:** {scores[2]}/9")
        st.info(result["lexical_feedback"])
        st.write(f"**Grammar:** {scores[3]}/9")
        st.info(result["grammar_feedback"])
        
        st.write("---")
        st.subheader("🏁 Overall Feedback")
        st.success(result["overall_feedback"])
    else:
        st.warning("Please enter an essay before evaluating.")

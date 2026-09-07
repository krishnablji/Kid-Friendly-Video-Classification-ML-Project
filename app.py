"""
app.py
Streamlit Web UI Application for Kid-Friendly Video Classification.
- Allows user to upload any video.
- Automatically sends the video to Gemini API to generate vast scene descriptions.
- Passes Gemini output into our custom text classifier.
- Performs sentence-by-sentence evaluation with IMMEDIATE EARLY EXIT:
  Stops execution the moment one unsafe statement is detected without reading the rest of the text.
- Outputs purely SAFE or UNSAFE.
"""

import os
import sys
import time
import tempfile
import streamlit as st

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sentence_scanner import SentenceSafetyScanner, split_into_sentences
from src.gemini_service import GeminiVideoService, PRESET_DEMO_TEXTS

# Page Configuration
st.set_page_config(
    page_title="Kid-Friendly Video Safety Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Badges and Sentence Inspector
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1e293b;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .safe-badge {
        background-color: #dcfce7;
        color: #166534;
        border: 2px solid #22c55e;
        padding: 16px 28px;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        margin: 15px 0;
    }
    .unsafe-badge {
        background-color: #fee2e2;
        color: #991b1b;
        border: 2px solid #ef4444;
        padding: 16px 28px;
        border-radius: 12px;
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        margin: 15px 0;
    }
    .early-exit-alert {
        background-color: #fff1f2;
        border-left: 5px solid #f43f5e;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 12px 0;
        font-size: 0.95rem;
    }
    .sentence-card-safe {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 8px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .sentence-card-unsafe {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 8px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .sentence-card-skipped {
        background-color: #f8fafc;
        border-left: 4px solid #cbd5e1;
        padding: 8px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        color: #94a3b8;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_scanner():
    """Caches the trained model scanner in memory."""
    return SentenceSafetyScanner()


def main():
    st.markdown('<div class="main-header">🛡️ Kid-Friendly Video Safety Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Pipeline: 10s Video &rarr; Gemini Vision AI &rarr; '
        'Sentence-by-Sentence Early-Exit Classifier &rarr; SAFE / UNSAFE Verdict</div>',
        unsafe_allow_html=True
    )

    # Initialize model scanner
    try:
        scanner = get_scanner()
    except Exception as e:
        st.error(f"Error loading trained model: {e}")
        st.stop()

    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Required for live Gemini video analysis. If left empty, you can use the Demo Scenarios."
        )

        st.markdown("---")
        st.markdown("### 🧠 Model Architecture")
        st.write("- **Model**: Classical Logistic Regression")
        st.write("- **NLP Features**: TF-IDF (Unigrams + Bigrams)")
        st.write("- **Dataset**: SafeWatch-Bench-200K (25,000 samples)")
        st.write("- **Test Accuracy**: **91.52%**")
        st.write("- **Strategy**: Sentence-by-sentence early-exit")

        st.markdown("---")
        st.caption("Developed with classical NLP & machine learning.")

    # Workspace Layout
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("📹 Step 1: Select or Upload Video")
        mode = st.radio(
            "Choose Input Source:",
            ["Upload Custom Video", "Use Benchmark Demo Scenario"],
            horizontal=True
        )

        video_path = None
        gemini_summary_text = None

        if mode == "Upload Custom Video":
            uploaded_file = st.file_uploader(
                "Upload a video file (MP4, MOV, WEBM, AVI)",
                type=["mp4", "mov", "webm", "avi"]
            )
            if uploaded_file is not None:
                # Save temporarily for processing
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.flush()
                tfile.close()
                video_path = tfile.name

                st.video(video_path)
                st.caption(f"Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
        else:
            scenario = st.selectbox(
                "Select a Benchmark Scenario:",
                list(PRESET_DEMO_TEXTS.keys())
            )
            gemini_summary_text = PRESET_DEMO_TEXTS[scenario]
            st.info(f"Loaded scenario: **{scenario}**")

        analyze_btn = st.button("🚀 Analyze Video Safety", type="primary", use_container_width=True)

    with col_right:
        st.subheader("📊 Step 2: Automated Analysis & Safety Verdict")

        if analyze_btn:
            if mode == "Upload Custom Video" and not video_path:
                st.warning("Please upload a video file first!")
                st.stop()

            # Stage 1: Gemini Video Processing
            with st.spinner("🤖 Sending video to Gemini API for multimodal scene description..."):
                if mode == "Upload Custom Video":
                    if not api_key_input:
                        st.error("Please provide your Gemini API Key in the sidebar to process custom videos.")
                        st.stop()
                    try:
                        gemini_service = GeminiVideoService(api_key=api_key_input)
                        gemini_summary_text = gemini_service.summarize_video(video_path)
                    except Exception as e:
                        st.error(f"Gemini API Error: {e}")
                        st.stop()

            # Display Gemini's vast text description
            with st.expander("📝 Gemini's Generated Video Summary (Full Vast Text)", expanded=False):
                st.write(gemini_summary_text)

            # Stage 2: Sentence-by-Sentence Early-Exit Safety Scanning
            with st.spinner("🔍 Running Sentence-by-Sentence Early-Exit Classifier..."):
                t0 = time.time()
                scan_result = scanner.scan_text(gemini_summary_text)
                scan_duration = time.time() - t0

            # Final Verdict Banner
            if scan_result["final_decision"] == "SAFE":
                st.markdown(
                    '<div class="safe-badge">✅ SAFE (Kid-Friendly)</div>',
                    unsafe_allow_html=True
                )
                st.success(
                    f"All {scan_result['total_sentences']} sentences evaluated. "
                    f"Zero unsafe statements detected. Approved for children!"
                )
            else:
                st.markdown(
                    '<div class="unsafe-badge">❌ UNSAFE (Not Kid-Friendly)</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="early-exit-alert">'
                    f'<strong>⚠️ EARLY-EXIT TRIGGERED!</strong><br>'
                    f'An unsafe statement was detected at <strong>Sentence #{scan_result["trigger_index"]}</strong>. '
                    f'Execution was stopped immediately without reading the remaining '
                    f'{scan_result["total_sentences"] - scan_result["sentences_scanned_count"]} sentence(s)!'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Total Sentences in Text", scan_result["total_sentences"])
            with m_col2:
                st.metric("Sentences Scanned Before Stop", scan_result["sentences_scanned_count"])
            with m_col3:
                st.metric("Model Scan Time", f"{scan_duration*1000:.1f} ms")

            # Sentence-by-Sentence Execution Breakdown
            st.markdown("### 🔎 Sentence-by-Sentence Audit Log")

            all_sentences = split_into_sentences(gemini_summary_text)
            scanned_records = scan_result["history"]

            for i, sent in enumerate(all_sentences, start=1):
                if i <= len(scanned_records):
                    rec = scanned_records[i - 1]
                    if rec["prediction"] == "SAFE":
                        st.markdown(
                            f'<div class="sentence-card-safe">'
                            f'<strong>Sentence {i} &mdash; <span style="color:#16a34a;">[ SAFE ]</span> '
                            f'(Safe Prob: {rec["safe_probability"]*100:.1f}%)</strong><br>'
                            f'"{sent}"'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="sentence-card-unsafe">'
                            f'<strong>Sentence {i} &mdash; <span style="color:#dc2626;">[ UNSAFE ]</span> '
                            f'(Unsafe Prob: {rec["unsafe_probability"]*100:.1f}%) &mdash; ⚡ EXECUTION HALTED HERE</strong><br>'
                            f'"{sent}"'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    # Skipped sentences due to early-exit!
                    st.markdown(
                        f'<div class="sentence-card-skipped">'
                        f'<strong>Sentence {i} &mdash; [ SKIPPED &mdash; NOT EVALUATED ]</strong><br>'
                        f'"{sent}"'
                        f'</div>',
                        unsafe_allow_html=True
                    )


if __name__ == "__main__":
    main()

"""Public EduSense AI walkthrough using synthetic data only."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edusense.demo import build_demo_session  # noqa: E402
from edusense.reporting import make_pdf_report  # noqa: E402


st.set_page_config(
    page_title="EduSense AI · Live Demo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #F7F8FC; color: #1D2340; }
    [data-testid="stSidebar"] { background: #EEF0FA; border-right: 1px solid #DEE2F2; }
    .brand { padding: .45rem 0 1rem; }
    .brand-mark { display:inline-block; width:2rem; height:2rem; border-radius:.8rem;
      background:linear-gradient(135deg,#5367F5,#8794FF); color:white; text-align:center;
      line-height:2rem; font-weight:800; margin-right:.45rem; }
    .brand-name { font-size:1.12rem; font-weight:800; vertical-align:middle; color:#20284B; }
    .eyebrow { color:#5967CC; font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .hero { border-radius:1.15rem; padding:1.45rem 1.65rem; margin:.4rem 0 1rem;
      background:linear-gradient(110deg,#E8EBFF,#F5F3FF); border:1px solid #DCE0FA; }
    .hero h1 { margin:.2rem 0 .35rem; color:#20284B; font-size:2rem; }
    .hero p { margin:0; color:#4D5578; }
    div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #E0E4F2;
      border-radius:1rem; padding:1rem 1.05rem; box-shadow:0 3px 12px #303A6610; }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_dashboard() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Interactive presentation demo</div>
          <h1>EduSense AI</h1>
          <p>Explore classroom-level signals and teacher prompts with synthetic sample data.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This hosted walkthrough uses generated sample data only. It has no camera, microphone, "
        "or media-upload controls."
    )

    choice, size = st.columns([1.5, 1])
    with choice:
        scenario = st.selectbox(
            "Choose a sample scenario",
            ["Group discussion", "Quiet reading", "Room transition"],
        )
    with size:
        class_size = st.slider("Assumed class size", min_value=15, max_value=35, value=24)

    result = build_demo_session(scenario, class_size)
    summary = result["summary"]
    metrics = result["metrics"]
    attendance = result["attendance"]

    st.caption(result["scenario_description"])
    st.caption("All values below are fabricated for demonstration; no real classroom was analyzed.")

    cards = st.columns(4)
    cards[0].metric("Activity signal", f"{summary['mean_activity_signal']:.1f}/100")
    cards[1].metric("Present profiles", f"{summary['present_profiles']}/{summary['tracked_profiles']}")
    cards[2].metric("Roster coverage", f"{summary['roster_coverage']:.1f}%")
    cards[3].metric("Room audio", f"{summary['mean_audio_dbfs']:.1f} dBFS")

    st.caption(
        "The activity signal averages motion with speech-like audio. It is a classroom-level "
        "heuristic—not a measure of learning, attention, or individual performance."
    )
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.subheader("Sample session timeline")
        st.line_chart(metrics.set_index("elapsed_seconds")[["activity_signal", "motion_signal"]], height=320)
        st.dataframe(
            metrics[[
                "elapsed_seconds", "detected_faces", "present_profiles", "motion_signal",
                "audio_activity", "activity_signal",
            ]],
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Teaching prompts")
        st.caption("Rule-based ideas for a teacher to review in context.")
        for item in result["suggestions"]:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.write(item["action"])
                st.caption(item["reason"])
        st.subheader("Anonymous sample profiles")
        st.dataframe(pd.DataFrame(attendance), width="stretch", hide_index=True)

    csv_data = metrics.to_csv(index=False).encode("utf-8")
    pdf_data = make_pdf_report(
        str(result["started_at"]),
        str(result["source_label"]),
        summary,
        metrics,
    )
    csv_col, pdf_col, _ = st.columns([1, 1, 2])
    csv_col.download_button(
        "Download sample CSV",
        data=csv_data,
        file_name="edusense-synthetic-demo.csv",
        mime="text/csv",
        width="stretch",
    )
    pdf_col.download_button(
        "Download sample PDF",
        data=pdf_data,
        file_name="edusense-synthetic-demo.pdf",
        mime="application/pdf",
        width="stretch",
    )


def show_about() -> None:
    st.markdown("## How to read this demo")
    st.markdown(
        """
        - **Motion signal** is a synthetic room-level movement pattern.
        - **Speech-like audio** is a generated probability pattern, not recorded sound.
        - **Present profiles** are anonymous sample counts, not named students.
        - **Roster coverage** is an illustrative count divided by the class-size control.
        - **Teaching prompts** are rules-based suggestions; a teacher decides what fits.
        """
    )
    st.warning(
        "Do not use activity signals for grading, discipline, diagnosis, or student ranking. "
        "The real video/audio workflow is a local app in the source repository and should only "
        "be used with appropriate notice and consent."
    )
    st.link_button("View full source on GitHub", "https://github.com/kartikk0211/edusense-ai")


with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">E</span><span class="brand-name">EduSense AI</span></div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Demo", ["Dashboard", "About & limits"], label_visibility="collapsed")
    st.caption("Public demo · synthetic data only")

if page == "Dashboard":
    show_dashboard()
else:
    show_about()

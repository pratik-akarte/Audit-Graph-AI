import streamlit as st
import requests
import time

# Base URL pointing to your backend server.py
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Video Compliance QA Dashboard",
    page_icon="📹",
    layout="wide"
)

st.title("📹 Video Compliance & QA Agent")

# Sidebar - Health & Server Check
with st.sidebar:
    st.header("System Status")
    if st.button("Check Backend Connection"):
        try:
            res = requests.get(f"{BACKEND_URL}/health", timeout=3)
            if res.status_code == 200:
                st.success("Backend API Online")
            else:
                st.warning(f"Backend status: {res.status_code}")
        except Exception as e:
            st.error("Backend Server Unreachable")

# Main Page Layout (Two Columns)
col1, col2 = st.columns([1, 1])

# --- Column 1: Video Input ---
with col1:
    st.subheader("1. Video Input")
    video_url = st.text_input(
        "Enter Video Storage/Blob URL",
        placeholder="https://yourstorage.blob.core.windows.net/container/sample.mp4"
    )

    if video_url:
        st.video(video_url)

    start_audit = st.button("Run Compliance Audit", type="primary", use_container_width=True)

# --- Column 2: Results & Reporting ---
with col2:
    st.subheader("2. Audit Report")

    if start_audit:
        if not video_url:
            st.warning("Please enter a valid video URL first.")
        else:
            with st.status("Running Compliance Workflow...", expanded=True) as status:
                st.write("📤 Sending video URL to backend pipeline...")
                
                try:
                    payload = {"video_url": video_url}
                    response = requests.post(f"{BACKEND_URL}/audit", json=payload)

                    if response.status_code == 200:
                        st.write("🔍 RAG Engine analyzing knowledge base & transcript...")
                        data = response.json()
                        status.update(label="Audit Completed!", state="complete", expanded=False)

                        # Parse results
                        verdict = data.get("status", "UNKNOWN")
                        report = data.get("final_report", "No report text available.")

                        # Show Verdict Badge
                        if verdict == "PASS":
                            st.success(f"### Verdict: {verdict}")
                        else:
                            st.error(f"### Verdict: {verdict}")

                        # Show Report Output
                        st.markdown("#### Detailed Findings")
                        st.text_area("Audit Details", value=report, height=350)
                    else:
                        status.update(label="Audit Failed", state="error")
                        st.error(f"Server error ({response.status_code}): {response.text}")

                except Exception as err:
                    status.update(label="Connection Error", state="error")
                    st.error(f"Failed to communicate with backend: {err}")
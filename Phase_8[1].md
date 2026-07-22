# Phase 8: Frontend / UI Development

## Objectives
- Build a simple, intuitive interface for users to upload media and view results.

## Tech Options
| Option | When to Use |
|---|---|
| **Streamlit** | Fastest to build, great for MVP/demo/researchers |
| **React.js** | Best for production, custom UX, mobile-responsive |

## Streamlit MVP Example
```python
import streamlit as st
import requests

st.title("🕵️ Deepfake Detection System")
uploaded_file = st.file_uploader("Upload a video or image", type=["mp4", "jpg", "png"])

if uploaded_file and st.button("Analyze"):
    with st.spinner("Analyzing..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/api/v1/detect/video", files=files)
        result = response.json()

    st.subheader(f"Result: {result['result']}")
    st.metric("Confidence", f"{result['confidence']*100:.2f}%")
    if result['result'] == "FAKE":
        st.error("⚠️ This media appears to be manipulated.")
    else:
        st.success("✅ This media appears authentic.")
```

## React.js Key Components (for production version)
- `UploadPanel.jsx` — drag & drop file upload
- `ResultCard.jsx` — shows label, confidence gauge, heatmap overlay
- `HistoryTable.jsx` — past scans for logged-in users
- `Dashboard.jsx` — analytics charts (see Phase 9)

## UX Considerations
- Show progress/loading state during video processing (can take several seconds).
- Display confidence as a percentage **and** a visual gauge — avoid pure binary framing since deepfake detection is probabilistic.
- Always show a disclaimer: "AI detection is not 100% certain — use as one signal among several."

## Deliverables
- Working Streamlit MVP (`src/dashboard/app.py`) or React frontend (`frontend/`)
- Responsive upload + result display flow
- Heatmap visualization component

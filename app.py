
import json
import urllib.request
import urllib.error

import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Agentic RAG Research Assistant")
st.caption("Ask questions about your security research documents.")

question = st.text_area(
    "Enter your question",
    placeholder="How can directory traversal be detected?",
    height=100,
)

if st.button("Ask Question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            payload = json.dumps(
                {"question": question.strip()}
            ).encode("utf-8")

            request = urllib.request.Request(
                API_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with st.spinner("Searching documents and generating an answer..."):
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))

            st.subheader("Answer")
            st.write(result.get("answer", "No answer returned."))

            st.subheader("Source citations")
            citations = result.get("citations", [])

            if citations:
                for citation in citations:
                    st.write(f"- `{citation}`")
            else:
                st.info("No citations returned.")

            with st.expander("Request details"):
                st.write("Status:", result.get("status"))
                st.write("Trace ID:", result.get("trace_id"))
                st.write("Latency:", f"{result.get('latency', 0):.3f} seconds")
                st.write("Cost:", f"${result.get('cost', 0):.4f}")

        except urllib.error.URLError:
            st.error(
                "Cannot connect to the API. "
                "Make sure the FastAPI server is running."
            )
        except Exception as error:
            st.error(f"Something went wrong: {error}")

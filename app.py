import streamlit as st
from PIL import Image

from image_utils import analyze_image
from llm_chain import get_chain

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Image Chatbot",
    page_icon="🖼️",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.chat-container {
    max-width: 900px;
    margin: auto;
}
.stChatMessage {
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_context" not in st.session_state:
    st.session_state.image_context = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🖼️ Image Upload")
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_container_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                st.session_state.image_context = analyze_image(image)

            st.success("Image analyzed successfully!")

    if st.session_state.image_context:
        st.subheader("📌 Image Summary")
        st.write(st.session_state.image_context)

    st.divider()
    st.caption("⚡ Powered by Free Open-Source Models")

# ---------------- Main Chat UI ----------------
st.title("💬 Image Analysis Chatbot")

st.markdown(
    "Upload an image from the sidebar and chat with the assistant about it."
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask something about the image...")

if user_input:
    if not st.session_state.image_context:
        st.warning("Please upload and analyze an image first.")
    else:
        # User message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # Assistant response
        llm, prompt = get_chain()
        full_prompt = prompt.format(
            image_context=st.session_state.image_context,
            question=user_input
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = llm(full_prompt)
                st.write(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import base64
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Image Analyzer Chatbot")
st.title("Image Analyzer Chatbot")

# Load API Key from .env
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Please set your GOOGLE_API_KEY in the .env file.")
    st.stop()

# Initialize session state for chat history and image tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "image_base64" not in st.session_state:
    st.session_state.image_base64 = None

# Sidebar for image upload
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.sidebar.image(uploaded_file)
    
    # Process new image upload
    if st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.image_data = uploaded_file.getvalue()
        st.session_state.image_base64 = base64.b64encode(st.session_state.image_data).decode("utf-8")
        
        # Clear chat history when a new image is uploaded (optional, but usually desired)
        st.session_state.chat_history = []
        st.sidebar.success("New image uploaded. Chat history cleared.")

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=api_key,
)

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        st.write(item["text"])
            else:
                st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        st.write(item["text"])
            else:
                st.write(msg.content)

# Chat Input
user_query = st.chat_input("Ask a question about the image...")

if user_query:
    if not st.session_state.image_base64:
        st.error("Please upload an image first.")
        st.stop()

    # Display user's question immediately
    with st.chat_message("user"):
        st.write(user_query)

    content = [
        {"type": "text", "text": user_query},
        {
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.image_base64}"}
        }
    ]

    # Create the HumanMessage and append to history
    new_msg = HumanMessage(content=content)
    st.session_state.chat_history.append(new_msg)

    # Get response from the model
    with st.chat_message("assistant"):
        try:
            # Request the stream from the model
            stream = llm.stream(st.session_state.chat_history)
            
            # Generator to safely extract text from the chunks as they arrive
            def stream_parser(chunk_stream):
                for chunk in chunk_stream:
                    if isinstance(chunk.content, list):
                        for item in chunk.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                yield item["text"]
                    else:
                        yield chunk.content
                        
            # st.write_stream automatically handles the typewriter effect and returns the concatenated string
            full_response_text = st.write_stream(stream_parser(stream))
            
            # Append only the cleaned, full text as an AIMessage to history
            st.session_state.chat_history.append(AIMessage(content=full_response_text))
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.chat_history.pop()
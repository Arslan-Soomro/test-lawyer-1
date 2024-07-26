import streamlit as st

chat_history = [
    {
        "role": "ai",
        "message": "Hi, how can I help you?"
    }
];

with st.chat_message("ai"):
    st.write("Hello 👋")
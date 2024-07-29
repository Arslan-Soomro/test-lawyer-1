import streamlit as st
from app_utils import ask_legal_assistant
from prompts import default_qa_prompt


if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "ai",
        "content": "הי איך אני יכול לעזור לך?"
    }]

st.markdown("""
<style>
body, html {
    direction: RTL;
    unicode-bidi: bidi-override;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

with st.expander("Custom Prompt"):
    custom_prompt = st.text_area(
        "Custom QA Prompt",
        default_qa_prompt
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("הקלד את השאלה שלך"):
    # Display user message in chat message container
    with st.chat_message("human"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "human", "content": user_input})
    # print(st.session_state.messages[0:-1])
    # print(st.session_state.messages[-1])
    response = ask_legal_assistant(st.session_state.messages[-1]["content"], st.session_state.messages[0:-1], custom_prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("ai"):
        st.markdown(response["answer"])

    with st.expander("See Response JSON"):
        st.json(response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "ai", "content": response["answer"]})

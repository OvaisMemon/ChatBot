import streamlit as st
from langchain_aws import ChatBedrock
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def demo_chatbot():
    llm = ChatBedrock(
        credentials_profile_name='default',
        model_id = 'us.meta.llama3-1-70b-instruct-v1:0',
        model_kwargs={
            "temperature": 0.5,
            "top_p": 0.9, 
            "max_tokens": 512
        }
    )

    return llm

st.title('Hi, I am a chatbot. How can I help you? :sunglasses:')

# Manage session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = InMemoryChatMessageHistory()

# Display previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input = st.chat_input('Type your message here ...')

if input:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": input})
    
    with st.chat_message("user"):
        st.markdown(input)

    # Retrieve chatbot model
    chatbot = demo_chatbot()

    # Runnable chain with message history
    chain = RunnableWithMessageHistory(chatbot, lambda _: st.session_state["chat_history"])

    # Get response from model
    response = chain.invoke(input, config={"configurable": {"session_id": "1"}})

    # Add chatbot response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": response.content})

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(response.content)
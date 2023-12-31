
import streamlit as st
from streamlit_chat import message
from backend.core import run_llm

st.set_page_config(page_title="SBDC", layout="wide")

st.header("SBDC - ChatBot!")

if 'input_value' not in st.session_state:
    st.session_state.input_value = 'input_value'

prompt = st.text_input("Prompt", placeholder="질문을 상세히 입력해주세요..", value="", key="input")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"

    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    
    return sources_string


if prompt:
    with st.spinner("Generationg response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        # print(generated_response)
        
        # sources = set(
        #     [doc.metadata["source"] for doc in generated_response["source_documents"]]
        # )

        # formatted_response = (
        #     f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        # )
        
        # sources = set(
        #     [doc.metadata["source"] for doc in generated_response["source_documents"]]
        # )

        formatted_response = (
            f"{generated_response['answer']}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))
        st.session_state["input_value"] = ""
        # st.experimental_rerun()

# if st.session_state["chat_answers_history"]:
if 'chat_answers_history' in st.session_state:
    for i, (user_query, generated_response) in enumerate(zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"])):
        message(user_query, is_user=True, key=f"user_message_{i}", seed=1)
        message(generated_response, key=f"bot_message_{i}", seed=2)
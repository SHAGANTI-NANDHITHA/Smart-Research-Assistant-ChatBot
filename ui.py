from chatbot import reply_llm
import streamlit as st
from streamlit_chat import message

st.title('Smart Research Assistant')
st.header('The Role of AI Chat-Bots in Academic Research')


prompt = st.text_input('Prompt', placeholder="Enter your prompt here..")

if "user_prompt_history" not in st.session_state:
    st.session_state['user_prompt_history'] = []

if "chat_answer_history" not in st.session_state:
    st.session_state['chat_answer_history'] = []

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [] 


def create_sources_string(source_urls) -> str:
    if not source_urls:
        return ""
    source_list = list(source_urls)
    source_list.sort()
    source_string = "sources: \n"
    for i, source in enumerate(source_list):
        source_string += f'{i+1}, {source}\n'
    return source_string


if prompt:
    with st.spinner('Generating response..'):
        
        generated_response = reply_llm(prompt)
        st.session_state['user_prompt_history'].append(prompt)
        st.session_state['chat_answer_history'].append(generated_response)
        st.session_state['chat_history'].append((prompt, generated_response)) 

if st.session_state['chat_answer_history']:
    for generated_response, user_query in zip(st.session_state['chat_answer_history'],st.session_state['user_prompt_history']):
        message(user_query,is_user=True)
        message(generated_response)
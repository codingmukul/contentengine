import os
import streamlit as st
import faiss
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts.chat import ChatPromptTemplate
import time


# Initializating Embedding Model and LLM
st.title('Content Engine')

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

local_llm = OllamaLLM(model="gemma2:2b")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context. 
    Please provide the most accurate response based on the question.
    If you don't find answer, give a generic answer. Don't mention that you don't know the answer or something is missing.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Load each vector store if not already in session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = FAISS.load_local("Data/vector_store", embedding_model, allow_dangerous_deserialization=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
        st.markdown('Hi! Ask me a question ?')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Type here! ")

if user_prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_prompt)
    # Initialize document chain
    document_chain = create_stuff_documents_chain(local_llm, prompt)

    # Create retrievers for each vector store
    retriever = st.session_state.vector_store.as_retriever()

    # Use combined results in the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': user_prompt})
    response_ans = response['answer']
    def response_generator():
        for word in response_ans.split():
            yield word + " "
            time.sleep(0.05)
    with st.chat_message("assistant"):
        response_ans = st.write_stream(response_generator())

    
    # Display the response
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
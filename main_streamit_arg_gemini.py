import streamlit as st
import tiktoken

import os   #gemini

from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI   #gemini

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import unstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import faiss

#from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
        page_title = "DirChat",
        page_icon = ":books:")
    
    st.title("_Private Data :red[QA Chat]_ :books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf','docx','pptx'], accept_multiple_files=True)
        gemini_api_key = st.text_input("Gemini API KEY", key = "chatbot_api_key", type = "password")     #openai_api 는 gemini_api 로 변경
        process = st.button("Process")
        
    if process:
        if not gemini_api_key:
            st.info("Please add your Gemini API Key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)    #vectorestore 일치여부 확인필요
        
        st.session_state.conversation = get_conversation_chain(vetorestore, gemini_api_key)
        
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content" : "안년하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어보세요"}]
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    history = StreamlitChatMessageHistory(key = "chat_messages")
    
    # Chat logic
    if query := st.chat_input("질문을 입력해 주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            
            chain = st.session_state.conversation
            
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:    #확인 필요
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']
                
                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
    # st.write(source_documents)
                
# add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:    # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
            
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documnets = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = unstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
            
        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 900,
        chunk_overlap = 100,
        length_function = tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name = "jhgan/ko-sroberta-multitask",
                                        model_kwargs = {'device': 'cpu'},
                                        encode_kwargs = {'normalize_embeddings': True}
                                        )
    vectordb = faiss.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, gemini_api_key):
    # llm = ChatOpenAI(openai_api_key = openai_api_key, model_name = 'gpt-3.5-turbo', temperature = 0)
    
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDk0_7GU4FojcOW2Ko_Y2hh9OIK_XbE7-Y"   #gemini
    llm = ChatGoogleGenerativeAI(model = "gemini-pro")   #gemini
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        chain_type = "stuff",
        retriever = vetorestore.as_retriever(search_type = 'mmr', vervose = True),
        memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True, output_key='answer'),
        get_chat_history = lambda h: h,
        return_source_documents = True,
        verbose = True
    )
    
    return conversation_chain

if __name__ == '_main_':
    main()
import os
import tempfile
import streamlit as st
import pandas as pd

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableMap

import hashlib
import shutil

# ✅ 파일 해시 생성
def get_file_hash(uploaded_file):
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(file_content).hexdigest()

# ✅ pysqlite3 패치
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

st.header("(주말을 순삭시킨)유저 응답 기반 Q&A 챗봇 💬")

option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-4.1-nano"))

# ✅ CSV 로딩 → 유저 단위로 문서 생성
@st.cache_resource
def load_csv_and_create_docs(file_path: str):
    df = pd.read_csv(file_path)

    if 'user_id' not in df.columns or 'answer' not in df.columns:
        st.error("CSV 파일은 'user_id' 와 'answer' 컬럼을 포함해야 합니다.")
        return []

    docs = []
    for user_id, group in df.groupby('user_id'):
        content = "\n".join(group['answer'].astype(str).tolist())
        metadata = {"source": f"user_{user_id}"}
        docs.append(Document(page_content=content, metadata=metadata))

    return docs

# ✅ 벡터스토어 생성
@st.cache_resource
def create_vector_store(file_path: str):
    docs = load_csv_and_create_docs(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    file_hash = os.path.splitext(os.path.basename(file_path))[0]
    persist_dir = f"./chroma_db_user/{file_hash}"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=persist_dir
    )
    return vectorstore

# ✅ RAG 체인 초기화
@st.cache_resource
def initialize_components(file_path: str, selected_model: str):
    vectorstore = create_vector_store(file_path)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "이전 대화 내용을 반영해 현재 질문을 독립형 질문으로 바꿔줘."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서 내용을 참고하여 질문에 무조건 한국어로 답변해줘. 문서와 유사한 내용이 없으면 무조건 '관련된 내용이 없습니다'라고 말해줘. 꼭 이모지 써줘! 참고 문서는 아래에 보여줄 거야.\n\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = RunnableMap({
        "context": lambda x: x["context"],  # context 그대로 pass
        "answer": create_stuff_documents_chain(llm, qa_prompt)
    })

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# ✅ 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요 (user_id, answer 포함)", type=["csv"])

if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{file_hash}.csv")

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    rag_chain = initialize_components(temp_path, option)
    chat_history = StreamlitChatMessageHistory(key="chat_messages_user")

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )

    if len(chat_history.messages) == 0:
        chat_history.add_ai_message("업로드된 유저 응답 기반으로 무엇이든 물어보세요! 🤗")

    for msg in chat_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt_message := st.chat_input("질문을 입력하세요"):
        st.chat_message("human").write(prompt_message)
        with st.chat_message("ai"):
            with st.spinner("생각 중입니다..."):
                config = {"configurable": {"session_id": "user_session"}}
                response = conversational_rag_chain.invoke(
                    {"input": prompt_message},
                    config,
                )
                answer = response['answer']
                st.write(answer)

                if "관련된 내용이 없습니다" not in answer and response.get("context"):
                    with st.expander("참고 문서 확인"):
                        for doc in response['context']:
                            source = doc.metadata.get('source', '알 수 없음')
                            source_filename = os.path.basename(source)
                            st.markdown(f"👤 {source_filename}")
                            st.markdown(doc.page_content)

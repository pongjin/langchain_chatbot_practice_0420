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

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']


st.header("유저 응답 기반 Q&A 챗봇 💬")

option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-4.1-nano"))

# ✅ CSV 로딩 후 유저 단위로 청크 만들기
@st.cache_resource
def load_csv_and_create_docs(_file):
    df = pd.read_csv(_file)

    if 'user_id' not in df.columns or 'answer' not in df.columns:
        st.error("CSV 파일은 'user_id' 와 'answer' 컬럼을 포함해야 합니다.")
        return []

    docs = []
    for user_id, group in df.groupby('user_id'):
        content = "\n".join(group['answer'].astype(str).tolist())
        metadata = {"source": f"user_{user_id}"}
        docs.append(Document(page_content=content, metadata=metadata))

    return docs

# ✅ Vectorstore 생성
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)

    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory="./chroma_db_user"
    )
    return vectorstore

# ✅ RAG 체인 초기화
@st.cache_resource
def initialize_components(_docs, selected_model):
    vectorstore = create_vector_store(_docs)
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "이전 대화 내용을 반영해 현재 질문을 독립형 질문으로 바꿔줘."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서 내용을 참고하여 질문에 무조건 한국어로 답변해줘. 문서와 유사한 내용이 없으면 무조건 '관련된 내용이 없습나다'라고 말해줘. 꼭 이모지 써줘! 참고 문서는 아래에 보여줄 거야.\n\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# ✅ CSV 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요 (user_id, answer 포함)", type=["csv"])

if uploaded_file:
    docs = load_csv_and_create_docs(uploaded_file)
    rag_chain = initialize_components(docs, option)
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
                st.write(response['answer'])
                
                # ✅ "관련된 내용이 없습니다" 같은 문구가 있으면 참고 문서 숨김
                if "관련된 내용이 없습니다" not in answer and response.get("context"):
                    with st.expander("참고 문서 확인"):
                        for doc in response['context']:
                            source = doc.metadata.get('source', '알 수 없음')

                            # ✅ 경로 정제: 임시 경로 제거
                            if "/var/" in source:
                                continue  # 이 줄을 쓰면 임시 파일은 아예 안 보여줌

                            # 또는 아래처럼 파일 이름만 추출해서 보여줄 수도 있음
                            source_filename = os.path.basename(source)

                            # 📄 파일명과 페이지 번호 바로 뒤에 page_content 추가
                            st.markdown(f"👤 {source_filename}")
                            st.markdown(doc.page_content)  # 페이지 내용 바로 출력

import streamlit as st
from dotenv import load_dotenv
import os, json
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, get_response_synthesizer, ServiceContext, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from typing import List
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import PyPDF2
from io import BytesIO
import pandas as pd

load_dotenv('.env')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
dir_path = './index'

# openAI
def set_llms(embeddings_llm: str, llm: str):
    embed_model = OpenAIEmbedding(model=embeddings_llm) #??
    llm = OpenAI(model=llm_model,
        temperature=0.5,
        system_prompt="You are an expert on the answering questions about the documents/context provided to you. Assume that all questions are related to the documents/context provided to you. Keep your answers technical and based on facts – do not hallucinate features."
        )
    Settings.llm = llm
    return llm

def load_file(file_path: str) -> List:
    return SimpleDirectoryReader(input_files=[file_path]).load_data()

def transform_data(documents: List, chunk_size: int, chunk_overlap: int, seperator: str = " ") -> List:
    pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(separator=seperator, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3),
        ]
    )
    return pipeline.run(documents=documents)

def create_vector_store(nodes: List, service_context, save: bool=True):
    index = VectorStoreIndex(nodes, service_context)
    if save:
        index.storage_context.persist(persist_dir=dir_path)
    return index

def load_vector_store():
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=dir_path))

def retriever(input_text: str, index, top_k: int):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )
    response = query_engine.query(input_text)
    return response

def process_pdf(file_buffer):
    print(f'file type: {type(file_buffer)}')
    pdf_reader = PyPDF2.PdfReader(file_buffer)
    document_texts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
    documents = [Document(text=text) for text in document_texts if text]
    print(f'Documents type: {type(documents)}')
    return documents

def process_csv(file):
    df = pd.read_csv(file)
    return [Document(text=str(row)) for index, row in df.iterrows()]

def process_xlsx(file):
    df = pd.read_excel(file, engine='openpyxl')
    return [Document(text=str(row)) for index, row in df.iterrows()]

def process_txt(file):
    content = file.read().decode('utf-8')
    return [Document(text=content)]

def load_documents(uploaded_file):
    file_buffer = BytesIO(uploaded_file.getvalue())
    if uploaded_file.type == "application/pdf":
        documents = process_pdf(file_buffer)
    elif uploaded_file.type == "text/csv":
        documents = process_csv(file_buffer)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        documents = process_xlsx(file_buffer)
    elif uploaded_file.type == "text/plain":
        documents = process_txt(file_buffer)
    else:
        st.error("Unsupported file type!")
    return documents

@st.cache_resource(show_spinner=False) # what does this actually do?
def load_data(documents, chunk_size: int = 516, chunk_overlap: int = 128, seperator: str = " "):
    llm = set_llms(embeddings_model, llm_model)
    service_context = ServiceContext.from_defaults(llm=llm)
    nodes = transform_data(documents, chunk_size, chunk_overlap, separator)
    index = create_vector_store(nodes, service_context)
    # add retrieve
    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True) # what is this? it shouldn't be here
    st.session_state.chat_engine = chat_engine


#''' FRONT-END '''

st.title('🦜🔗 Quickstart App')
with st.sidebar:

    embeddings_model = st.selectbox(
        "Select embedding model",
        ("text-embedding-3-small", "text-embedding-3-large"),
        key='embeddings_model')
    llm_model = st.selectbox(
        "Select LLM model",
        ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"),
        key='llm_model')

    chunk_size = st.number_input('Chunk size', 512)
    chunk_overlap = st.number_input('Chunk overlap', 128)
    separator = st.text_input('Separator', ' ')
    top_k = st.slider('top-k', 0, 15, 1)

    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'csv', 'xlsx', 'txt'])
    if uploaded_file:
        st.session_state['uploaded_file'] = uploaded_file

    if st.button('Create embeddings'):
        if 'uploaded_files' in st.session_state:
            documents = load_documents(st.session_state['uploaded_file'])
            st.write("filename:", uploaded_file.name)
            st.success("Done!")
            load_data(documents, chunk_size, chunk_overlap, separator)
            st.write('Indexing completed for:', uploaded_file.name)
        else:
            st.error('Please upload at least one PDF file before clicking this button.')


if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the article"} ## ??
    ]

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #response = chat_engine.chat(prompt)
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

# TO DO: Handle different file formats, better code structure, dockerfile, requirements, readme, add front-end spice
# Improvements: from vector store selection, improve whole RAG process, add usage of local models, evaluation
# Git help: https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches


# Deliverable:
    # upload to sidebar --> load file / use old ones or just chat normally
    # Upload file button: https://github.com/gabrielchua/RAGxplorer
    # Option to save vector store and load previously saved
    # Add evaluation
    # Add embedding and llm parameters
    # Improve front-end
    # Model selection (local embeddings & llm with openAI)
    # Upload to github
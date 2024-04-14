import streamlit as st
from dotenv import load_dotenv
import os, json
from huggingface_hub import InferenceClient
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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from tempfile import NamedTemporaryFile
import shutil
import PyPDF2
from io import BytesIO
# import pandas as pd
# import PyPDF2
# import openpyxl

load_dotenv('.env')
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
dir_path = './index'
file_name = './4 Ways to Quantify Fat Tails with Python _ by Shaw Talebi _ Towards Data Science.pdf'

## Hugigng face 
#Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#Settings.llm = None
#llm = 'mistralai/Mistral-7B-Instruct-v0.2'#'mistralai/Mistral-7B-Instruct-v0.1'

# openAI
def set_llms(embeddings_llm: str, llm: str):
    embed_model = OpenAIEmbedding(model=embeddings_llm)
    llm = OpenAI(temperature=0.2, model=llm)
    Settings.llm = llm


st.title('🦜🔗 Quickstart App')

def decode_bytes_string(response):
    try:
        # Decode and load JSON in one line
        json_data = json.loads(response.decode('utf-8'))
    except UnicodeDecodeError as e:
        json_data = f"Decoding error: {str(e)}"
    except json.JSONDecodeError as e:
        json_data = f"JSON decoding error: {str(e)}"
    return json_data

def load_file(file_path: str) -> List:
    return SimpleDirectoryReader(input_files=[file_path]).load_data()

def transform_data(documents: List, chunk_size: int, chunk_overlap: int, seperator: str = " ") -> List:
    pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(separator=seperator, chunk_size=chunk_size, chunk_overlap=chunk_overlap), # chunking
        #TitleExtractor(nodes=5), # metadata
        #QuestionsAnsweredExtractor(questions=3), # metadata
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

def prompt_template(input_text: str, context: str) -> str:
    # Prompt templates
    prompt_template_w_context = lambda context, comment: f"""[INST]Answer the users question.

    {context}
    Please respond to the following user question. Use the context above if it is helpful.

    {comment}
    [/INST]
    """
    return prompt_template_w_context(context, input_text)

def local_llm_generate_response(input_text: str, model: str) -> str:
    inference = InferenceClient(model=model, token=HUGGING_FACE_TOKEN)
    response = inference.post(json={"inputs": input_text}, model=model)
    st.info(decode_bytes_string(response)[0]['generated_text'])

def process_pdf(file_buffer):
    print(f'file type: {type(file_buffer)}')
    pdf_reader = PyPDF2.PdfReader(file_buffer)
    document_texts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
    documents = [Document(text=text) for text in document_texts if text]
    print(f'Documents type: {type(documents)}')
    return documents

# def process_csv(file):
#     df = pd.read_csv(file)
#     return [Document(text=str(row)) for index, row in df.iterrows()]

# def process_xlsx(file):
#     df = pd.read_excel(file, engine='openpyxl')
#     return [Document(text=str(row)) for index, row in df.iterrows()]

# def process_txt(file):
#     content = file.read().decode('utf-8')
#     return [Document(text=content)]

#if os.path.exists(dir_path):
#    index = load_vector_store()
#else:
#documents = load_file(file_name)

# @st.cache_resource(show_spinner=False)
# def load_data(_documents: List, chunk_size: int = 512, chunk_overlap: int = 128, llm: str = 'gpt-35-turbo', seperator: str = ' '):
#     service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm, temperature=0.5, system_prompt="You are an expert on the measuring fat tailness and your job is to answer technical questions. Assume that all questions are related to the fat tailness. Keep your answers technical and based on facts – do not hallucinate features."))
#     with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
#         nodes = transform_data(_documents, chunk_size, chunk_overlap, seperator)
#         index = create_vector_store(nodes, service_context)
#         return index

def load_data(documents, chunk_size, chunk_overlap, llm, separator):
    service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm, temperature=0.5, system_prompt="You are an expert on the measuring fat tailness and your job is to answer technical questions. Assume that all questions are related to the fat tailness. Keep your answers technical and based on facts – do not hallucinate features."))
    nodes = transform_data(documents, chunk_size, chunk_overlap, separator)
    index = create_vector_store(nodes, service_context)
    return index.as_chat_engine(chat_mode="condense_question", verbose=True) #index

''' FRONT-END '''
# TO DO: get selections passed to functions (sessions?), save vector_store, load previously saved vector_store's

with st.sidebar:

    embeddings_model = st.selectbox(
         "Select embedding model",
         ("text-embedding-3-small", "text-embedding-3-large"),
         key='embeddings_model')
    llm_model = st.selectbox(
         "Select LLM model",
         ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"),
         key='llm_model')
    st.write(f"Selected LLM model: {llm_model}")

    chunk_size = st.number_input('Chunk size', 512)
    chunk_overlap = st.number_input('Chunk overlap', 128)
    separator = st.text_input('Separator', ' ')
    top_k = st.slider('top-k', 0, 15, 1)

    # select index

    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
    if uploaded_file:
        st.session_state['uploaded_files'] = uploaded_file

    if st.button('Create embeddings'):
        if 'uploaded_files' in st.session_state:
            file_buffer = BytesIO(uploaded_file.getvalue())

            # Extract text from PDF
            if uploaded_file.type == "application/pdf":
                documents = process_pdf(file_buffer)

            st.write("filename:", uploaded_file.name)
            st.success("Done!")
            set_llms(embeddings_model, llm_model)
            service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm_model, temperature=0.5, system_prompt="You are an expert on the measuring fat tailness and your job is to answer technical questions. Assume that all questions are related to the fat tailness. Keep your answers technical and based on facts – do not hallucinate features."))
            nodes = transform_data(documents, chunk_size, chunk_overlap, separator)
            index = create_vector_store(nodes, service_context)
            chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
            st.session_state.chat_engine = chat_engine
            #st.session_state.chat_engine = load_data(documents, chunk_size, chunk_overlap, separator, llm_model)
            st.write('Indexing completed for:', uploaded_file.name)
        else:
            st.error('Please upload at least one PDF file before clicking this button.')


if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the article"}
    ]

# add LLM select, temperature

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

# TO DO: Handle different file formats, better code structure
# Improvements: from vector store selection



# Deliverable:
    # upload to sidebar --> load file / use old ones or just chat normally
    # Upload file button: https://github.com/gabrielchua/RAGxplorer
    # Option to save vector store and load previously saved
    # Add evaluation
    # Add embedding and llm parameters
    # Improve front-end
    # Model selection (local embeddings & llm with openAI)
    # Upload to github


# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#     submitted = st.form_submit_button('Submit')
#     if submitted:
#         if os.path.exists(dir_path):
#             index = load_vector_store()
#         else:
#             documents = load_file(file_name)
#             nodes = transform_data(documents)
#             index = create_vector_store(nodes)
#         retriever(text, index, 5)


# resources: https://blog.streamlit.io/build-a-real-time-rag-chatbot-google-drive-sharepoint/
    # https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/

        #prompt = prompt_template(text, context)
        #generate_response_openai(text, model=llm)

# 1. Check each part what is imputed and the result (see what is the issue?, see how it works, what is retreieves/ answers?)
# 2. Add evaluation
# 3. Improve each part of the RAG (retrieval, llm, embeddings, prompting, transformation, agents, semantic taggging, vectore store...)
# 4. Create seperate index creation + index update mechanism. In the app all we do is call that index
    # Add index cache
# 5. Add conversation history

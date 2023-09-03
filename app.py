import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from llama_index import LLMPredictor
#from transformers import HuggingFaceHub
from langchain import HuggingFaceHub
from streamlit.components.v1 import html
from pathlib import Path
from time import sleep
import random
import string

import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Open AI Doc-Chat Assistant", layout="wide")
st.subheader("Open AI Doc-Chat Assistant: Life Enhancing with AI!")

css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

documents=[]

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))  
random_string = generate_random_string(20)
directory_path=random_string

with st.sidebar:
    st.subheader("Upload your Documents Here: ")
    pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=False)
    if pdf_files is not None:
        os.makedirs(directory_path)
        file_path = os.path.join(directory_path, pdf_files.name)
        with open(file_path, 'wb') as f:
            f.write(pdf_files.read())
        st.success('File uploaded & saved successfully.')
        documents = SimpleDirectoryReader(directory_path).load_data()

# Load documents from a directory
#documents = SimpleDirectoryReader('data').load_data()

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

llm_predictor = LLMPredictor(HuggingFaceHub(repo_id="HuggingFaceH4/starchat-beta", model_kwargs={"min_length":100, "max_new_tokens":1024, "do_sample":True, "temperature":0.2,"top_k":50, "top_p":0.95, "eos_token_id":49155}))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

new_index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
)

new_index.storage_context.persist("directory_path")

storage_context = StorageContext.from_defaults(persist_dir="directory_path")

loadedindex = load_index_from_storage(storage_context=storage_context, service_context=service_context)

query_engine = loadedindex.as_query_engine()

while True:
    try:
        question = st.text_input("Enter your query here:")
        print("Your query:\n"+question)
        if question.strip().isspace():
            st.write("Query Empty. Please enter valid query first.")
            break
        elif question == "":
#            st.write("Query Empty. Please enter valid query first.")
            break
        elif question.strip() == "":
            st.write("Query Empty. Please enter valid query first.")
            break
        elif question.isspace():
            st.write("Query Empty. Please enter valid query first.")
            break
        elif question=="exit":
            break
        elif question!="":
            initial_response = query_engine.query(question)
            temp_ai_response=str(initial_response)
            final_ai_response=temp_ai_response.partition('<|end|>')[0] 
            print("AI Response:\n"+final_ai_response)
            st.write("AI Response:\n\n"+final_ai_response)
    except Exception as e:
        st.stop()

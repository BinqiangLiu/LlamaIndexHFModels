import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader
#llama_index handles pdf file processing under the hood, even for encryption related files

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

st.set_page_config(page_title="PDF AI Chat Assistant - Open Source Version", layout="wide")
st.subheader("Welcome to PDF AI Chat Assistant - Life Enhancing with AI.")
st.write("Important notice: This Open PDF AI Chat Assistant is offered for information and study purpose only and by no means for any other use. Any user should never interact with the AI Assistant in any way that is against any related promulgated regulations. The user is the only entity responsible for interactions taken between the user and the AI Chat Assistant.")

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

print(f"定义处理多余的Context文本的函数")
def remove_context(text):
    # 检查 'Context:' 是否存在
    if 'Context:' in text:
        # 找到第一个 '\n\n' 的位置
        end_of_context = text.find('\n\n')
        # 删除 'Context:' 到第一个 '\n\n' 之间的部分
        return text[end_of_context + 2:]  # '+2' 是为了跳过两个换行符
    else:
        # 如果 'Context:' 不存在，返回原始文本
        return text
print(f"处理多余的Context文本函数定义结束")    

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

llm_predictor = LLMPredictor(HuggingFaceHub(repo_id="HuggingFaceH4/starchat-beta", model_kwargs={"min_length":512, "max_new_tokens":1024, "do_sample":True, "temperature":0.2,"top_k":50, "top_p":0.95, "eos_token_id":49155}))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)

if "tf_switch" not in st.session_state:
    st.session_state.tf_switch = True     
    
if "query_engine" not in st.session_state:
    st.session_state.query_engine =  ""

with st.sidebar:
    st.subheader("Upload your Documents Here: ")
    pdf_files = st.file_uploader("Choose your PDF Files and Press OK", type=['pdf'], accept_multiple_files=True)
    if pdf_files:
        os.makedirs(directory_path)
        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file.name)
            with open(file_path, 'wb') as f:
                f.write(pdf_file.read())
            st.success(f"File '{pdf_file.name}' saved successfully.")
            
    if st.button('Process to AI Chat'):            
        with st.spinner("Processing your PDF file..."):
            try:
                documents = SimpleDirectoryReader(directory_path).load_data()
                # Load documents from a directory named 'data'
                #documents = SimpleDirectoryReader('data').load_data()                
                new_index = VectorStoreIndex.from_documents(
                    documents,
                    service_context=service_context,
                )
                new_index.storage_context.persist("directory_path")
                storage_context = StorageContext.from_defaults(persist_dir="directory_path")
                loadedindex = load_index_from_storage(storage_context=storage_context, service_context=service_context)
                st.session_state.query_engine =  loadedindex.as_query_engine()
                #query_engine = loadedindex.as_query_engine()  
                st.write("File processed. Now you can proceed to query your PDF file!")
                st.session_state.tf_switch=False
            except Exception as e:
                print("waiting for path creation.")  
                st.write("File processing failed. Please try again.")  

st.session_state.user_question = st.text_input("Enter your question & query your PDF file:", disabled=st.session_state.tf_switch)    
if st.session_state.user_question !="" and not st.session_state.user_question.strip().isspace() and not st.session_state.user_question == "" and not st.session_state.user_question.strip() == "" and not st.session_state.user_question.isspace():
    with st.spinner("AI Working...Please wait a while to Cheers!"):  
        print("Your query:\n"+st.session_state.user_question)
        try:        
            initial_response = st.session_state.query_engine.query(st.session_state.user_question)
            temp_ai_response=str(initial_response)
            cleaned_initial_ai_response = remove_context(temp_ai_response)
            #final_ai_response=temp_ai_response.partition('<|end|>')[0] 
            final_ai_response = cleaned_initial_ai_response.split('<|end|>')[0].strip().replace('\n\n', '\n').replace('<|end|>', '').replace('<|user|>', '').replace('<|system|>', '').replace('<|assistant|>', '')
            new_final_ai_response = final_ai_response.split('Unhelpful Answer:')[0].strip()
            new_final_ai_response = new_final_ai_response.split('Note:')[0].strip()
            new_final_ai_response = new_final_ai_response.split('Please provide feedback on how to improve the chatbot.')[0].strip()            
            
            print("AI Response:\n"+new_final_ai_response)
            st.write("AI Response:\n\n"+new_final_ai_response)
            
        except Exception as e:
            st.write("Unknown error. Please try again.")

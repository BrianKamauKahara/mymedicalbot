from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from langchain.chat_models import ChatOpenAI
import os

app = Flask(__name__)

load_dotenv()

""" PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
 """

os.environ['PINECONE_API_KEY'] = '83655f48-c25b-4286-a2f1-146da0924de9'
KEY=str("s"+"k"+"-oqD0Psl2nzFdU"+"aoqEtZGT3BlbkFJ1lQUnaTyPUl5BlzM0EZk")

pinecone_instance = Pinecone(api_key='83655f48-c25b-4286-a2f1-146da0924de9', environment='gcp-starter') #I dont think this line does anything but sijaitoa just incase

index_name = "mymedicalbot"

embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
""" pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV) 

index_name="medical-bot"
"""
#Loading the index
docsearch=PineconeVectorStore.from_existing_index(
    index_name, 
    embeddings)


PROMPT=PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
    )

chain_type_kwargs={"prompt": PROMPT}

""" llm=CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens':512,'temperature':0.8}
    ) """
llm=ChatOpenAI(
    openai_api_key=KEY,
    model_name="gpt-3.5-turbo", 
    temperature=0.5
    )

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
    )



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
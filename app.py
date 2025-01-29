from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from pinecone import Pinecone, ServerlessSpec
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
#PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone


# Create an instance of Pinecone with your API key
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

myindex="medicochatbot-1"
# Now proceed with creating or using indexes
if 'myindex' not in pc.list_indexes().names():
    pc.create_index(
        name='myindex',
        dimension=384,  # Use the correct dimension for your embeddings
        metric='cosine',  # You can change the metric to 'cosine', 'dot-product', etc.
        spec=ServerlessSpec(
            cloud='aws',  # Specify the cloud provider (e.g., 'aws', 'gcp', 'azure')
            region='us-east-1'  # Specify the region'
        )
    )





#Loading the index
docsearch=Pinecone.Index(myindex, embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



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


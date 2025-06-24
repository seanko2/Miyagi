import os
import lancedb
import json
import speech_to_text as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

#Chat Prompt Template offers better conversation flow capabilities and have roles which is applicable to this use case
from langchain_core.prompts import ChatPromptTemplate


#README: This is for user prompting and retrieiving relevant documents from RAG to feed into LLM
#This is the engine of Miyagi

#aws credentials
config_data = json.load(open("config.json"))

aws_access_key = config_data["aws_access_key_id"]
aws_secret_key = config_data["aws_secret_access_key"]

db = lancedb.connect(
    uri="s3://lance.vector.db/",
    storage_options={
    "aws_access_key_id": aws_access_key,
    "aws_secret_access_key": aws_secret_key,
    "region": "us-east-1"
    })

load_dotenv()
#HuggingFace token 
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name= embedding_model, model_kwargs={'device': 'cpu'})

# Function to retrieve relevant documents from RAG
def retrieve(query):
    table = "embeddings_tbl"

    print(db)
    print(embeddings)


    #this is an easy way that abstracts the need to embed query yourself and retrieve docuemnts
    vector_store = LanceDB(connection=db, table_name=table, embedding= embeddings)
    print("Row count:", vector_store._table.to_arrow().num_rows)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("retriever", retriever)
    #getting relevant documents here
    retrieved_docs = retriever.invoke(query)
    
    return retrieved_docs 



#this section is copied from huggingface model card --> deploy --> inference provider
client = InferenceClient(
    provider="hf-inference",
    api_key= hf_token,
)

#I want to do states because sometimes the model needs more tokens to give feedback (state == 1)
#but if I only want the model to ask a question (state == -1), i want to limit those tokens to a way smaller amount
state = -1
curr_question = ""
for _ in range(2):
    user_input = st.listen_n_transcribe()

    relevant_documents = retrieve(user_input)

    if isinstance(relevant_documents, list):
        relevant_documents = " ".join([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in relevant_documents])
    elif hasattr(relevant_documents, "page_content"): 
        relevant_documents = relevant_documents.page_content
    if state == -1:
        max_resp = 80
        message = [
        {
                "role": "system",
                "content": (
                    "You are a job interview coach. Generate a *single*, concise behavioral interview question. "
                    "Avoid adding extra context or explanation."
                )
            },
            {
                "role": "user",
                "content": "The following is relevant documents" + relevant_documents + "This is user input" + user_input
            },
            
        ]
    elif state == 1:
        max_resp = 300
        message = [
        {
                "role": "system",
                "content": "You are a conversational assistant that helps with behaviorial interview questions. Generate a single, concise, and common behaviorial interview question."
            },
            {
                "role": "user",
                "content": "The following is relevant documents" + relevant_documents + "This is user input" + user_input
            },
            {
                "role": "assistant",
                "content": "The question is: " + curr_question
            },
            
        ]

    state *= -1
    # Send the request to Hugging Face inference API
    completion = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages= message,
        max_tokens = max_resp,
        temperature= .5,
        
    )

    print(completion.choices[0].message.content)
    curr_question = completion.choices[0].message.content



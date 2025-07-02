from flask import Flask, jsonify, render_template, session
import os
import lancedb
import speech_to_text as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = os.urandom(24)


#README: This is for user prompting and retrieiving relevant documents from RAG to feed into LLM
#This is the engine of Miyagi

load_dotenv()
#aws credentials
aws_access_key = os.getenv("aws_access_key_id")
aws_secret_key = os.getenv("aws_secret_access_key")
db = lancedb.connect(
    uri="s3://lance.vector.db/",
    storage_options={
    "aws_access_key_id": aws_access_key,
    "aws_secret_access_key": aws_secret_key,
    "region": "us-east-1"
    })


#HuggingFace token 
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
os.environ["TOKENIZERS_PARALLELISM"] = "false"

embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name= embedding_model, model_kwargs={'device': 'cpu'})

#this section is copied from huggingface model card --> deploy --> inference provider
client = InferenceClient(
    provider="hf-inference",
    api_key= hf_token,
)


def retrieve(query):
    """Function to retrieve relevant documents from RAG"""
    table = "embeddings_tbl"

    #this is an easy way that abstracts the need to embed query yourself and retrieve docuemnts
    vector_store = LanceDB(connection=db, table_name=table, embedding= embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    #getting relevant documents here
    retrieved_docs = retriever.invoke(query)
    
    return retrieved_docs 


def process_documents(relevant_documents):
    """Process retrieved documents into a string format"""
    if isinstance(relevant_documents, list):
        return " ".join([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in relevant_documents])
    elif hasattr(relevant_documents, "page_content"): 
        return relevant_documents.page_content
    return str(relevant_documents)

#I want to have two functions in this chatbot, the Interviewing and the Constructively Critical
def generate_question(user_input, relevant_documents):
    """Generate a behavioral interview question"""
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
            "content": f"The following is relevant documents: {relevant_documents}. This is user input: {user_input}"
        }
    ]
    
    completion = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=message,
        max_tokens=80,
        temperature=0.5,
    )
    
    return completion.choices[0].message.content


def provide_feedback(user_input, relevant_documents, question):
    """Provide feedback on the user's response"""
    message = [
        {
            "role": "system",
            "content": (
                "You are a job interview coach. Provide constructive criticism to the response user gave using common best practices you know. "
                "Make sure to reference pieces of the user input while giving feedback and provide good examples. When responding, make sure to talk directly at user using second-person terms "
            )
        },
        {
            "role": "user",
            "content": f"The following is relevant documents: {relevant_documents}. This is user input: {user_input}"
        },
        {
            "role": "assistant",
            "content": f"The question is: {question}"
        }
    ]
    
    completion = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=message,
        max_tokens=600,
        temperature=0.5,
    )
    
    return completion.choices[0].message.content

@app.route('/')
def home():
    """Home page"""
    if 'state' not in session:
        session['state'] = -1  # -1 for question mode, 1 for feedback mode
        session['current_question'] = ""
    
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_interview():
    """Start the interview process"""
    session['state'] = -1
    session['current_question'] = ""
    
    return jsonify({
        'message': "Hi my name is Miyagi, your interviewing coach. How would you like me to help today?",
        'status': 'ready'
    })

@app.route('/voice_input', methods=['POST'])
def voice_input():
    """Process voice input using speech-to-text"""
    try:
        # Use your existing speech_to_text function
        user_input = st.listen_n_transcribe()
        
        # Process the transcribed input
        relevant_documents = retrieve(user_input)
        processed_docs = process_documents(relevant_documents)
        
        if session.get('state', -1) == -1:
            # Question mode
            question = generate_question(user_input, processed_docs)
            session['current_question'] = question
            session['state'] = 1
            
            return jsonify({
                'transcription': user_input,
                'response': question,
                'type': 'question',
                'state': session['state']
            })
            
        elif session.get('state', -1) == 1:
            # Feedback mode
            feedback = provide_feedback(user_input, processed_docs, session.get('current_question', ''))
            session['state'] = -1
            
            return jsonify({
                'transcription': user_input,
                'response': feedback,
                'type': 'feedback',
                'state': session['state']
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_session():
    """Reset the interview session"""
    session.clear()
    session['state'] = -1
    session['current_question'] = ""
    
    return jsonify({'message': 'Session reset successfully'})

@app.route('/status')
def get_status():
    """Get current session status"""
    return jsonify({
        'state': session.get('state', -1),
        'current_question': session.get('current_question', ''),
        'mode': 'question' if session.get('state', -1) == -1 else 'feedback'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
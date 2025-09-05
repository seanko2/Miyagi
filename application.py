from flask import Flask, jsonify, render_template, session, request
import os
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

application = Flask(__name__)
application.secret_key = os.urandom(24)


#README: This is for user prompting and retrieiving relevant documents from RAG to feed into LLM
#This is the engine of Miyagi

#so that hugging face has permission for my cache direcotires 
cache_dir = "/app/.cache/huggingface"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

# Your existing embedding model code
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model, 
    model_kwargs={'device': 'cpu'},
    cache_folder=cache_dir  # Explicitly set cache folder
)

# Load .env only in development
if os.environ.get('FLASK_ENV') != 'production':
    load_dotenv()

# Use environment variables (set in EB console)
aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

# Add validation
if not all([aws_access_key, aws_secret_key, hf_token]):
    raise ValueError("Missing required environment variables")

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
    provider="auto",
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

#I want to have two functions in this chatbot, the Interviewing and the Constructive critical coach
def generate_question(user_input, relevant_documents):
    """Generate a behavioral interview question"""
    message = [
        {
            "role": "system",
            "content": (
                "You are a job interview coach. Generate using relevant documents and ask the user using second person a *single*, *concise* sample behavioral interview question. "
                "provide *only* the question itself and nothing else except the question"
            )
        },
        {
            "role": "user",
            "content": f"Here are relevant documents: {relevant_documents} and this is the user's prompt: {user_input}. Generate a new question each time."
        }
    ]
    
    try:
        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=message,
            max_tokens=80,
            temperature=0.5,
        )
        
        # Add debugging and error handling
        print(f"Completion response: {completion}")
        
        if completion and hasattr(completion, 'choices') and completion.choices:
            return completion.choices[0].message.content
        else:
            print("No choices in completion response")
            return "Can you tell me about a time when you had to overcome a significant challenge at work?"
            
    except Exception as e:
        print(f"Error in generate_question: {e}")
        # Return a fallback question if API fails
        return "Can you tell me about a time when you had to overcome a significant challenge at work?"

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
            "content": f"The following is relevant documents: {relevant_documents}. This is user's answer to the question: {user_input}"
        },
        {
            "role": "assistant",
            "content": f"The question is: {question}"
        }
    ]
    
    try:
        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=message,
            max_tokens=600,
            temperature=0.55,
        )
        
        print(f"Feedback completion response: {completion}")
        
        if completion and hasattr(completion, 'choices') and completion.choices:
            return completion.choices[0].message.content
        else:
            print("No choices in feedback completion response")
            return "That's a good start! Try to be more specific about the actions you took and the results you achieved. Use the STAR method (Situation, Task, Action, Result) to structure your response."
            
    except Exception as e:
        print(f"Error in provide_feedback: {e}")
        return "That's a good start! Try to be more specific about the actions you took and the results you achieved. Use the STAR method (Situation, Task, Action, Result) to structure your response."

@application.route('/')
def home():
    """Home page"""
    if 'state' not in session:
        session['state'] = -1  # -1 for question mode, 1 for feedback mode
        session['current_question'] = ""
    
    return render_template('index.html')

@application.route('/start', methods=['POST'])
def start_interview():
    """Start the interview process and immediately generate first question"""
    try:
        session['state'] = 1  # Set to feedback mode since we're about to ask a question
        session['current_question'] = ""
        
        # Generate first question immediately
        default_input = "Hey Miyagi, can you ask me a sample behavioral interview question I would typically get in an interview"
        
        # Add error handling for document retrieval
        try:
            relevant_documents = retrieve(default_input)
            processed_docs = process_documents(relevant_documents)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            processed_docs = "No documents available"
        
        question = generate_question(default_input, processed_docs)
        session['current_question'] = question
        
        return jsonify({
            'message': question,
            'status': 'question_ready',
            'type': 'question',
            'state': session['state']
        })
        
    except Exception as e:
        print(f"Error in start_interview: {e}")
        return jsonify({
            'error': 'Failed to start interview',
            'message': str(e)
        }), 500

@application.route('/next_question', methods=['POST'])
def next_question():
    """Generate next question without user input"""
    try:
        default_input = "Hey Miyagi, can you ask me another sample behavioral interview question I would typically get in an interview"
        
        try:
            relevant_documents = retrieve(default_input)
            processed_docs = process_documents(relevant_documents)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            processed_docs = "No documents available"
        
        question = generate_question(default_input, processed_docs)
        
        session['current_question'] = question
        session['state'] = 1  # Set to feedback mode to expect voice input next
        
        return jsonify({
            'response': question,
            'type': 'question',
            'state': session['state']
        })
        
    except Exception as e:
        print(f"Error in next_question: {e}")
        return jsonify({
            'error': 'Failed to generate next question',
            'message': str(e)
        }), 500

@application.route('/process_speech_text', methods=['POST'])
def process_speech_text():
    """Process transcribed text from browser speech recognition"""
    try:
        # Check if we're in the correct state for input
        if session.get('state', -1) != 1:
            return jsonify({'error': 'Speech input not available in current state'}), 400
        
        data = request.get_json()
        user_input = data.get('text', '')
        
        if not user_input:
            return jsonify({'error': 'No text provided'}), 400
        
        # Process the transcribed input (same as before)
        try:
            relevant_documents = retrieve(user_input)
            processed_docs = process_documents(relevant_documents)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            processed_docs = "No documents available"
        
        feedback = provide_feedback(user_input, processed_docs, session.get('current_question', ''))
        session['state'] = -1  # Set to question mode after providing feedback
        
        return jsonify({
            'response': feedback,
            'type': 'feedback',
            'state': session['state']
        })
    
    except Exception as e:
        print(f"Error in process_speech_text: {e}")
        return jsonify({'error': str(e)}), 500

@application.route('/reset', methods=['POST'])
def reset_session():
    """Reset the interview session"""
    session.clear()
    session['state'] = -1
    session['current_question'] = ""
    
    return jsonify({'message': 'Session reset successfully'})

@application.route('/status')
def get_status():
    """Get current session status"""
    return jsonify({
        'state': session.get('state', -1),
        'current_question': session.get('current_question', ''),
        'mode': 'question' if session.get('state', -1) == -1 else 'feedback'
    })


@application.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    application.run(debug=True, host='127.0.0.1', port=8888)
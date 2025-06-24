from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
#HuggingFace token 
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

login(hf_token)
print(bool(hf_token))
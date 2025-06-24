import os
import json
import boto3
import lancedb
from langchain_community.vectorstores import LanceDB
import pyarrow as pa

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#README: this file is store the vector embeddings of the data I scraped and send to S3 bucket


#HuggingFace token 
hf_token = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name= embedding_model, model_kwargs={'device': 'cpu'})


#FOR TRAINING DATA
def process_file_to_vector(local_path):
    with open(local_path, "r", encoding="utf=8") as file:
        try:
            text = file.read()
        except UnicodeDecodeError as e:
            print(f"Error decoding file: {local_path}")
            print(f"Error details: {e}")
        
        
        except Exception as e:
            print(f"An unexpected error occurred with file: {local_path}")
            print(f"Error details: {e}")
        

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 100)
    chunks = text_splitter.split_text(text =text)

    vectors = embeddings.embed_documents(chunks)
    chunked_data = [(chunk, vector) for chunk, vector in zip(chunks, vectors)]

    return chunked_data

#aws credentials
config_data = json.load(open("config.json"))

aws_access_key = config_data["aws_access_key_id"]
aws_secret_key = config_data["aws_secret_access_key"]




s3_client = boto3.client(
"s3",
aws_access_key_id = aws_access_key,
aws_secret_access_key = aws_secret_key
)

db = lancedb.connect(
    uri="s3://lance.vector.db/",
    storage_options={
    "aws_access_key_id": aws_access_key,
    "aws_secret_access_key": aws_secret_key,
    "region": "us-east-1"
    }
)

table_name = "embeddings_tbl"

# Create a table in LanceDB
my_schema = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),       
    ("vector", pa.list_(pa.float32(), 384)), 
    ("metadata", pa.struct([("file_name", pa.string())]))
])

#if table_name not in db.table_names():
    #db.create_table(table_name, schema= my_schema, mode = "overwrite")

#using this for now because of debugging 
if table_name in db.table_names():
    db.drop_table(table_name)  # Removes the existing table

db.create_table(table_name, schema=my_schema, mode="overwrite")


table = db.open_table(table_name)

with open("buckets.txt", 'r') as file:
    buckets = file.read().splitlines()

# List objects in the S3 bucket
for bucket in buckets:
    bucket_name = bucket
    response = s3_client.list_objects_v2(Bucket= bucket_name)

    if 'Contents' in response:
        for obj in response['Contents']:
            s3_file_path = obj['Key']
            local_file_path = os.path.join("/tmp", s3_file_path)  # Temporary local storage

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            try:
                #downloading temporarily on the operating system
                s3_client.download_file(bucket_name, s3_file_path, local_file_path)
            except Exception as e:
                print(f"Error downloading {s3_file_path}: {e}")
            

            chunks_vectors = process_file_to_vector(local_file_path)  # Returns chunked_data : TEXT, VECTOR 
            metadata = {"file_name": str(s3_file_path)}

            for idx, (chunk, vector) in enumerate(chunks_vectors):  
                clean_id = f"{obj['ETag'].strip('\"')}_{idx}"
                
                # Insert into LanceDB with both text and vector
                table.add([{
                    "id": clean_id,
                    "text": chunk,  # Store the actual text chunk
                    "vector": vector,  # Store the embedding
                    "metadata": metadata
                }])

            # Clean up local file
            os.remove(local_file_path)
    else:
        print(f"The bucket '{bucket_name}' is empty or does not contain any files.")





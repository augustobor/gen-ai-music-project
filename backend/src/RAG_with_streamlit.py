import os
import json
import hashlib
import logging

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from langchain_postgres.vectorstores import PGVector
from langchain_community.utilities.redis import get_client
from langchain_community.storage import RedisStore

from .utils import (
    load_music_data,
    summarize_text_and_tables,
    store_docs_in_retriever,
    initialize_retriever,
)
from database import COLLECTION_NAME, CONNECTION_STRING

load_dotenv()

client = get_client(
    url=os.environ.get("REDIS_URL"),
    encoding="utf-8",
    decode_responses=True,
)

def _get_file_path(file_upload):

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists

    if isinstance(file_upload, str):
        file_path = file_upload  # Already a string path
    else:
        file_path = os.path.join(temp_dir, file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
        return file_path


# Generate a unique hash for a PDF file
def generate_music_hash(music_path):
    """Generate a SHA-256 hash of the music file content."""
    with open(music_path, "rb") as f:
        music_bytes = f.read()
    return hashlib.sha256(music_bytes).hexdigest()


# Process uploaded music file
def process_music(file_upload):
    print('Processing music hash info...')
    
    file_path =  _get_file_path(file_upload)
    music_hash = generate_music_hash(file_path)

    load_retriever = initialize_retriever()
    existing = client.exists(f"music:{music_hash}")
    print(f"Checking Redis for hash {music_hash}: {'Exists' if existing else 'Not found'}")

    if existing:
        print(f"Music already exists with hash {music_hash}. Skipping upload.")
        return load_retriever

    print(f"New music detected. Processing... {music_hash}")
 
    3# If we have a new music file, we need to process it. Load, extract text, generate hash, and store in Redis.
    # Create a temporary directory to store the music file
    music_elements = load_music_data([file_path]) #file_path
    
    # Extract text and tables from the music elements
    text = [element.page_content for element in music_elements if 
            'Document' in str(type(element))]
   
    summaries = summarize_text_and_tables(text, []) #tables
    retriever = store_docs_in_retriever(text, summaries['text'], [],  summaries['table'], load_retriever)
    
    # Store the music hash in Redis
    client.set(f"music:{music_hash}", json.dumps({"text": "music processed"}))  

    # Debug: Check if Redis stored the key
    stored = client.exists(f"music:{music_hash}")
    # #remove temp directory
    # shutil.rmtree("dir")
    print(f"Stored music hash in Redis: {'Success' if stored else 'Failed'}")
    return retriever
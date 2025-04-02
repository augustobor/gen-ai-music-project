import os, json, time, uuid, logging, hashlib, tempfile, shutil  
from pathlib import Path  
from base64 import b64decode  
import torch, redis, streamlit as st  
import librosa
import librosa.display
import numpy as np

from dotenv import load_dotenv; load_dotenv()  

from IPython.display import display, HTML  
from database import COLLECTION_NAME, CONNECTION_STRING  
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
from langchain_core.messages import SystemMessage, HumanMessage  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser  
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda  
from langchain.schema.document import Document  
from langchain.retrievers.multi_vector import MultiVectorRetriever  
from langchain_postgres.vectorstores import PGVector  
from langchain_community.utilities.redis import get_client  
from langchain_community.storage import RedisStore  

def load_music_data(file_paths):
    """
    Loads music data from the given file paths, extracts MFCC features,
    and returns a list of Document objects.
    Path_files -> list of Documents to store.
    """

    documents = []
    for file_path in file_paths:
        logging.info(f"Loading music data from {file_path}")
        try:
            # Load the audio file using librosa
            song_sr, sr = librosa.load(file_path)

            # Extract features (example: MFCCs)
            mfccs = librosa.feature.mfcc(y=song_sr, sr=sr, n_mfcc=40)

            # Create Document object to store it
            metadata = {"source": file_path}
            page_content = f"MFCCs: {mfccs.tolist()}"  # Convert to list for serialization
            doc = Document(page_content=page_content, metadata=metadata)

            documents.append(doc)  # Add the Document to the list
            logging.info(f"Music data from {file_path} loaded and features extracted.")

        except Exception as e:
            logging.error(f"Error loading music data from {file_path}: {e}")

    loaded_files = [doc.metadata['source'] for doc in documents]
    print(f"Loaded music files: {loaded_files}")

    return documents  # Return the list of Document objects


# Summarize extracted text and tables using LLM
def summarize_text_and_tables(text, tables):
    """
    Summarizes the given text and tables using a Language Model.
    """
    logging.info("Ready to summarize data with LLM")
    prompt_text = """You are an assistant tasked with summarizing text and tables. \
    
                    You are to give a concise summary of the table or text and do nothing else. 
                    Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini") # Replace gpt-4o-mini with music LLM

    # model = MusicGen.get_pretrained("medium")
    # model.set_generation_params(duration=8)  # generate 8 seconds.

    summarize_chain = {"element": RunnablePassthrough()}| prompt | model | StrOutputParser()
    logging.info(f"{model} done with summarization")
    return {
        "text": summarize_chain.batch(text, {"max_concurrency": 5}),
        "table": summarize_chain.batch(tables, {"max_concurrency": 5})
    }


def initialize_retriever():
    """
    Initializes and returns a MultiVectorRetriever.
    """

    store = RedisStore(client=client)
    id_key = "doc_id"
    vectorstore = PGVector(
            embeddings=OpenAIEmbeddings(),
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
            )
    retrieval_loader = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    return retrieval_loader


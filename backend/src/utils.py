#Data Loading
import logging
import librosa
import numpy as np
import os
import hashlib
import json
import shutil
import uuid

from langchain.schema.document import Document
from langchain_community.utilities.redis import get_client
from database import COLLECTION_NAME, CONNECTION_STRING
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_community.storage import RedisStore

client = get_client(
    url=os.environ.get("REDIS_URL"),
    encoding="utf-8",
    decode_responses=True,
)

def store_docs_in_retriever(text, text_summary, table, table_summary, retriever):
    """Store text and table documents along with their summaries in the retriever."""

    def add_documents_to_retriever(documents, summaries, retriever, id_key = "doc_id"):
        """Helper function to add documents and their summaries to the retriever."""
        if not summaries:
            return None, []

        doc_ids = [str(uuid.uuid4()) for _ in documents]
        summary_docs = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]

        retriever.vectorstore.add_documents(summary_docs, ids=doc_ids)
        retriever.docstore.mset(list(zip(doc_ids, documents)))     

# Add text, table, and image summaries to the retriever
    add_documents_to_retriever(text, text_summary, retriever)
    add_documents_to_retriever(table, table_summary, retriever)
    return retriever


# Parse the retriever output
def parse_retriver_output(data):
    parsed_elements = []
    for element in data:
        # Decode bytes to string if necessary
        if isinstance(element, bytes):
            element = element.decode("utf-8")
        
        parsed_elements.append(element)
    
    return parsed_elements
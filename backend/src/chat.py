import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

from .utils import parse_retriver_output

def chat_with_llm(retriever):

    logging.info(f"Context ready to send to LLM ")
    prompt_text = """
        You are a music playlist curator AI. You will receive a list of song descriptions
        and a request for a playlist with certain characteristics. Your job is to create
        a playlist that fits the request, using the provided song descriptions.

        Song Descriptions:
        {context}

        Playlist Request:
        {question}

        Provide the playlist as a list of song titles, each on a new line.
        """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini") # Replace gpt-4o-mini with music LLM
    #model = MusicGen.get_pretrained("medium")
 
    rag_chain = ({
       "context": retriever | RunnableLambda(parse_retriver_output), "question": RunnablePassthrough(),
        } 
        | prompt 
        | model 
        | StrOutputParser()
        )
        
    logging.info(f"Completed! ")

    return rag_chain
from typing import Dict
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import BaseRetriever
from .config import secrets, models, params
from .prompts import RECIPE_PROMPT, SYSTEM_PROMPT, PREFERENCE_PROMPT


def get_gemini_llm():
    """
    Initialize Gemini-based Chat LLM for LangChain.
    """
    if not secrets.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set. Please set it in your environment or .env file."
        )
    genai.configure(api_key=secrets.GEMINI_API_KEY)
    return ChatGoogleGenerativeAI(
        model=models.gemini_model_name,
        temperature=params.generator_temperature,
        google_api_key=secrets.GEMINI_API_KEY,
        max_output_tokens=params.generator_max_tokens,
        convert_system_message_to_human=True,
    )


def create_recipe_chain():
    """
    Create an LLMChain for recipe synthesis using the global prompt.
    """
    llm = get_gemini_llm()
    return LLMChain(llm=llm, prompt=RECIPE_PROMPT)


def create_retrieval_qa_chain(retriever: BaseRetriever):
    """
    Optional: Standard RetrievalQA chain (unused in recipe synthesis but handy for Q&A).
    """
    llm = get_gemini_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": RECIPE_PROMPT},
        return_source_documents=True,
    )


def create_preference_parsing_chain():
    """
    Create an LLMChain for parsing user preferences into structured JSON.
    """
    llm = get_gemini_llm()

    return LLMChain(llm=llm, prompt=PREFERENCE_PROMPT)

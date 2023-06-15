from urllib.parse import urlparse
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel
from langchain.vectorstores import Chroma
import logging
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader, UnstructuredFileLoader
from langchain import LLMChain, OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from typing import List
import warnings

warnings.filterwarnings("ignore")


load_dotenv()
openai_api_key = os.environ.get("OPEN_API_KEY")
ai_model = os.environ.get("AI_MODEL")
ai_prompt_template = os.environ.get("AI_PROMPT_TEMPLATE")
system_prompt_template = os.environ.get("SYSTEM_PROMPT_TEMPLATE")

logger = logging.getLogger(__name__)


class RetrivalBaseQaResponse(BaseModel):
    """pydantic model for API response"""

    question: str
    answer: str
    status_code: int


class WebLink(BaseModel):
    """pydantic model for web link"""

    link: str = None


def text_split(docs: List[Document]) -> List[Document]:
    """
    Split the input documents into smaller texts.

    Args:
        docs (list[str]): The input documents.

    Returns:
        list[str]: The split texts.
    """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    return texts


def remove_file(file_path) -> None:
    """remove file from directory"""
    try:
        os.remove(file_path)
    except FileNotFoundError:
        logger.error("File does not exist", exc_info=True)


def get_real_path(file_name: str) -> str:
    """
    Get the real (absolute) path of a file based on the given file name.

    Args:
        file_name (str): The name of the file.

    Returns:
        str: The real (absolute) path of the file.
    """
    real_path = os.path.realpath(os.path.join(os.getcwd(), file_name))
    return real_path


def is_link(string: str) -> bool:
    """
    Check if a string represents a valid URL link.

    Args:
        string (str): The string to be checked.

    Returns:
        bool: True if the string represents a valid URL link, False otherwise.
    """
    parsed = urlparse(string)
    return bool(parsed.scheme)


def load_pdf_file_as_documents(file_path: str) -> List[Document]:
    """
    Load a PDF file and split it into separate documents.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list[str]: The loaded documents.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load_and_split()
    return documents


def load_web_based_link_as_document(link: str) -> List[Document]:
    """
    Load web-based content from a given link.

    Args:
        link (str): The web link.

    Returns:
        list[str]: The loaded documents.
    """
    loader = WebBaseLoader(link)
    documents = loader.load()
    return documents


def get_additional_info_from_document(question: str, documents: List[Document]) -> str:
    """
    Get the content of a document that is relevant to the given question.

    Args:
        question (str): The question.
        texts (list[str]): The texts from the document.

    Returns:
        answer (str): The relevant content.
    """
    texts = text_split(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    model = ChatOpenAI(model_name=ai_model, openai_api_key=openai_api_key)
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    result = qa(dict(question=question, chat_history=[]))
    answer = result.get("answer")
    return answer


def load_unstructured_file_as_documents(file_path: str) -> List[Document]:
    """
    Load an unstructured file and extract its content.

    Args:
        file_path (str): The path to the unstructured file.

    Returns:
        list[Document]: The loaded documents.
    """
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()
    return documents


def is_pdf(file_path: str) -> bool:
    """
    Check if a file has a PDF extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file is a PDF, False otherwise.
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower() == ".pdf"


def run_chain(prompt) -> str:
    """
    Run the language model chain to generate a response.

    Args:
        prompt (PromptTemplate): The prompt template for the language model.
        question (str): The question.
        additional_info (str): Additional information related to the topic.

    Returns:
        str: The generated response.
    """
    model = OpenAI(model_name=ai_model, openai_api_key=openai_api_key)
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run({})
    return response


def create_prompt(additional_info: str) -> ChatPromptTemplate:
    """
    Create the prompt for the language model chain.

    Args:
        additional_info (str): Additional information related to the topic.

    Returns:
        ChatPromptTemplate: The chat prompt template.
    """

    system_message_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(template=system_prompt_template, input_variables=[])
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(template=additional_info, input_variables=[])
    )
    ai_message_prompt = AIMessagePromptTemplate(
        prompt=PromptTemplate(template=ai_prompt_template, input_variables=[])
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt, ai_message_prompt]
    )
    return chat_prompt

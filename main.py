from fastapi import FastAPI, UploadFile, HTTPException
from urllib.parse import urlparse
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.image import UnstructuredImageLoader
from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.schema import Document
import logging
from langchain.document_loaders import UnstructuredURLLoader
import os
from dotenv import load_dotenv
from typing import List
from langchain.document_loaders import TextLoader

load_dotenv()
openai_api_key = os.environ.get("OPEN_API_KEY")
ai_model = os.environ.get("AI_MODEL")
logger = logging.getLogger(__name__)

app = FastAPI()


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


class InputTextRequestData(BaseModel):
    text: str


class RetrivalBaseQaResponse(BaseModel):
    question: str
    answer: str
    status_code: int


def retrieval_based_qa(question: str, documents: List[Document]) -> str:
    """
    Perform retrieval-based question answering.

    Args:
        question (str): The question to be answered.
        documents (List[Document]): A list of documents for retrieval.

    Returns:
        str: The answer to the question.
    """

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa.run(question)
    return response


@app.post("/upload_file")
async def upload_file(input_file: UploadFile, question: str):
    """
    Upload a file and perform retrieval-based question answering.

    Args:
        input_file (UploadFile): The uploaded file.
        question (str): The question to be answered.

    Returns:
        dict: A dictionary containing the question, answer, and status code.
    """

    contents = await input_file.read()
    with open(input_file.filename, "wb") as file:
        file.write(contents)

    file_path = get_real_path(file.name)
    loader = UnstructuredImageLoader(file_path)
    documents = loader.load()
    os.remove(file_path)
    try:
        response = retrieval_based_qa(question, documents)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return HTTPException(status_code=500, detail="Internal Server Error")

    return RetrivalBaseQaResponse(
        question=question, answer=response, status_code=200
    ).dict()


@app.post("/upload_text")
async def upload_file(text: InputTextRequestData, question: str):
    """
    Upload a text or a URL and perform retrieval-based question answering.

    Args:
        text (InputTextRequestData): The input text or URL.
        question (str): The question to be answered.

    Returns:
        dict: A dictionary containing the question, answer, and status code.
    """

    if is_link(text.text):
        loader = UnstructuredURLLoader(urls=[text.text])
        documents = loader.load()

    else:
        file_path = "output.txt"
        with open(file_path, "w") as file:
            file.write(text.text)

        file_path = get_real_path(file.name)
        loader = TextLoader(file_path, encoding="utf8")
        documents = loader.load()
        os.remove(file_path)
    try:
        response = retrieval_based_qa(question, documents)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return HTTPException(status_code=500, detail="Internal Server Error")
    return RetrivalBaseQaResponse(
        question=question, answer=response, status_code=200
    ).dict()

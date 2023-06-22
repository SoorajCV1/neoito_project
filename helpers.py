from urllib.parse import urlparse
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel
from langchain.vectorstores import Chroma
import logging
import os
from dotenv import load_dotenv
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
import pinecone
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import SimpleSequentialChain
from langchain.document_loaders import UnstructuredPDFLoader



warnings.filterwarnings("ignore")


load_dotenv()
openai_api_key = os.environ.get("OPEN_API_KEY")
ai_model = os.environ.get("AI_MODEL")
ai_prompt_template = os.environ.get("AI_PROMPT_TEMPLATE")
system_prompt_template = os.environ.get("SYSTEM_PROMPT_TEMPLATE")
pine_cone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")
logger = logging.getLogger(__name__)

pinecone.init(
    api_key=pine_cone_api_key, environment=pinecone_env
)

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_documents(docs)
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
    

    loader = UnstructuredPDFLoader(file_path) 
    documents = loader.load()
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

def get_additional_info_from_document_using_pine_cone(question: str, documents: List[Document], upload_doc_in_pine_cone = True) -> str:
    """
    Retrieve additional information from documents using Pinecone vector indexes.

    Args:
        question (str): The question.
        documents (List[Document]): The documents to search.
        upload_doc_in_pine_cone (bool): Whether to upload the documents to the Pinecone index.

    Returns:
        str: The retrieved information.
    """
    index_name = "langchain1"
    if not index_name in pinecone.list_indexes():
        pinecone.create_index("langchain1", dimension=1536)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    texts = text_split(documents)
    if upload_doc_in_pine_cone:
        page_content = []
        metadatas= []
        for t in texts:
            page_content.append(t.page_content)
            metadatas.append({"doc_id":"abc_doc"})
        docsearch = Pinecone.from_texts(page_content,embeddings, index_name="langchain1", metadatas=metadatas)
    else:
        docsearch = Pinecone.from_existing_index(embedding=embeddings, index_name="langchain1")
    docs = docsearch.similarity_search(query=question, include_metadata=True,filter={"doc_id":"abc_doc"})
    return docs
def create_chain_one():
    """
    Create the first chain for getting additional info of user question from document or link.

    Returns:
        LLMChain: The created chain.
    """
  
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def create_chain_two():
    """
    Create the second chain for getting anser of user question from chat gpt.

    Returns:
        LLMChain: The created chain.
    """
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
  
    system_message_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(template=system_prompt_template, input_variables=[])
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(template="{query}", input_variables=["query"])
    )
    ai_message_prompt = AIMessagePromptTemplate(
        prompt=PromptTemplate(template=ai_prompt_template, input_variables=[])
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt, ai_message_prompt]
    )
    chain =LLMChain(llm=llm, prompt=chat_prompt)
    return chain
    
    

def get_additional_info_from_document_using_chroma(question: str, documents: List[Document]) -> str:
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
    # docs=db.similarity_search(question)
    # return docs[0].page_content
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


def run_chain(adittional_docs, question) -> str:
    """
    Run the language model chain to generate a response.

    Args:
        prompt (PromptTemplate): The prompt template for the language model.
        question (str): The question.
        additional_info (str): Additional information related to the topic.

    Returns:
        str: The generated response.
    """

    chain_one = create_chain_one()
    chain_two = create_chain_two()

    overall_chain = SimpleSequentialChain(chains=[chain_one,chain_two])
    answer = overall_chain.run(input={"input_documents":adittional_docs, "question":question})
    return answer


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








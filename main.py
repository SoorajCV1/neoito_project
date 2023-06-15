from fastapi import FastAPI, UploadFile, HTTPException
import logging
from helpers import (
    WebLink,
    load_pdf_file_as_documents,
    remove_file,
    load_unstructured_file_as_documents,
    get_real_path,
    is_link,
    is_pdf,
    load_web_based_link_as_document,
    text_split,
    get_additional_info_from_document,
    create_prompt,
    run_chain,
    RetrivalBaseQaResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/upload_link")
async def upload_link(question: str, input_link: WebLink):
    """
    Upload a link and perform retrieval-based question answering.

    Args:
        question (str): The question to be answered.
        input_link (WebLink): The input web link.

    Returns:
        dict: A dictionary containing the question, answer, and status code.
    """
    link = input_link.link
    if not is_link(link):
        raise HTTPException(status_code=500, detail="Invalid link")
    
    documents = load_web_based_link_as_document(link)
    additional_info = get_additional_info_from_document(question, documents)

    try:
        prompt = create_prompt(additional_info)
        response = run_chain(prompt)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return HTTPException(status_code=500, detail="Internal Server Error")

    return RetrivalBaseQaResponse(
        question=question, answer=response, status_code=200
    ).dict()


@app.post("/upload_file/")
async def upload_file(question: str, input_file: UploadFile):
    """
    Upload a file and perform retrieval-based question answering.

    Args:
        question (str): The question to be answered.
        input_file (UploadFile): The uploaded file.

    Returns:
        dict: A dictionary containing the question, answer, and status code.
    """
    contents = await input_file.read()
    with open(input_file.filename, "wb") as file:
        file.write(contents)

    file_path = get_real_path(file.name)

    try:
        if is_pdf(file_path):
            documents = load_pdf_file_as_documents(file_path)
        else:
            documents = load_unstructured_file_as_documents(file_path)
        additional_info = get_additional_info_from_document(question, documents)
        remove_file(file_path)
        prompt = create_prompt(additional_info)
        response = run_chain(prompt)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return HTTPException(status_code=500, detail="Internal Server Error")

    return RetrivalBaseQaResponse(
        question=question, answer=response, status_code=200
    ).dict()

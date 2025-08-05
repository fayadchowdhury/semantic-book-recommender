from langchain_core.documents.base import Document
from langchain_core.embeddings.embeddings import Embeddings
from langchain_chroma import Chroma
import pandas as pd
import logging
import os

logger = logging.getLogger("etl")

def write_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Writes a Pandas DataFrame to a CSV file, creating directories if they don't exist.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to write to the CSV file.
    file_path (str): The path to the CSV file.
    
    Returns:
    None
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.debug(f"Saved df to {file_path}")

def write_data_to_vector_db(documents: list[Document], persist_directory: str, embedding_function: Embeddings) -> Chroma:
    """
    Writes a list of Document objects into a vector database after embedding with an embedding function
    and stores the database in a directory

    Parameters:
    documents (list[Document]): The list of Document objects to embed and store
    persist_directory (str): The directory to store the database in
    embedding_function (langchain_core.embeddings.embeddings.Embedding): The embedding function to use

    Returns:
    Chroma: A Chroma database object
    """
    db = Chroma.from_documents(documents=documents, embedding=embedding_function, persist_directory=persist_directory)
    return db
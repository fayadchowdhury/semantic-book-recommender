from langchain_core.embeddings.embeddings import Embeddings
from langchain_chroma import Chroma
import pandas as pd
import logging

logger = logging.getLogger("etl")

def read_data(file_path: str) -> pd.DataFrame:
    """
    Reads data from a CSV file into a Pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(file_path)
    logger.debug(f"Read CSV at {file_path}")
    return df

def read_data_from_vector_database(persist_directory: str, embedding_function: Embeddings) -> Chroma:
    """
    Reads from a vector database in a specified directory
    and de-embebds with the specified embedding_function
    
    Parameters:
    persist_directory (str): The directory to read the database from
    embedding_function (langchain_core.embeddings.embeddings.Embedding): The embedding function to use

    Returns:
    Chroma: A Chroma database object
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    return db
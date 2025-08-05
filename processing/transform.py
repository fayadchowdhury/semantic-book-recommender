import pandas as pd
import numpy as np
from transformers import pipeline
from transformers.pipelines.base import Pipeline
from langchain_core.documents.base import Document
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters.base import TextSplitter
from tqdm import tqdm
import logging

logger = logging.getLogger("etl")

def remove_null_rows(df: pd.DataFrame, columns_to_check: list[str]) -> pd.DataFrame:
    """
    Removes rows with null values in the columns specified in columns_to_check.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    columns_to_check (list[str]): The list of columns to check for null values.

    Returns:
    pd.DataFrame: The cleaned DataFrame with null rows removed.
    """
    df = df.dropna(subset=columns_to_check).reset_index(drop=True)
    logger.debug(f"Dropped null rows from {columns_to_check} columns")
    return df

def remove_short_descriptions(df: pd.DataFrame, cutoff_length: int = 25) -> pd.DataFrame:
    """
    Removes rows where the 'description' column has fewer than a specified number of words.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    cutoff_length (int): The minimum number of words required in the 'description'.

    Returns:
    pd.DataFrame: The filtered DataFrame with descriptions longer than the cutoff length.
    """
    df = df[df['description'].str.split().str.len() >= cutoff_length].reset_index(drop=True)
    logger.debug(f"Removed descriptions shorter than {cutoff_length} words")
    return df

def join_title_and_subtitle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the 'title' and 'subtitle' columns into a single 'title_join_subtitle' column.
    If 'subtitle' is NaN, it will only use the 'title'.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing 'title' and 'subtitle' columns.

    Returns:
    pd.DataFrame: The DataFrame with an additional 'title_join_subtitle' column.
    """
    df['title_join_subtitle'] = np.where(
        df["subtitle"].isna(), df["title"],
        df[["title", "subtitle"]].astype(str).agg(": ".join, axis = 1)
    )
    df = df.reset_index(drop=True)
    logger.debug(f"Combined title and subtitle")
    return df


def fix_thumbnail_urls(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Fixes the thumbnail column to add &fife=w800 for complete URLs
    and populate with cover-not-found photo asset for missing URLs.

    Parameters:
    df (pd.DataFrame): The DataFrame whose thumbanil column is being fixed

    Returns:
    pd.DataFrame: The DataFrame with the thumbnail column updated
    '''
    df["thumbnail"] = df["thumbnail"].apply(
        lambda x: x + "&fife=w800" if pd.notna(x) else "assets/cover-not-found.jpg"
    )
    logger.debug(f"Fixed thumbnails")
    return df

def drop_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    """
    Drops specified columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop columns.
    columns_to_drop (list[str]): The list of columns to drop.

    Returns:
    pd.DataFrame: The DataFrame with specified columns dropped.
    """
    df = df.drop(columns=columns_to_drop, axis=1).reset_index(drop=True)
    logger.debug(f"Drop {columns_to_drop} columns from df")
    return df

def map_simple_categories(df: pd.DataFrame, category_map: dict) -> pd.DataFrame:
    """
    Maps the 'categories' column in the DataFrame to a new set of categories based on a provided mapping.
    For simplicity, we go with fiction and non-fiction initially.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'categories' column.
    category_map (dict): A dictionary mapping old categories to new categories.
    
    Returns:
    pd.DataFrame: The DataFrame with the 'categories' column updated.
    """
    logger.debug(f"Category Mapping: {category_map}")
    df['simple_categories'] = df['categories'].map(category_map)
    logger.debug(f"Mapped categories")
    return df.reset_index(drop=True)

def generate_simple_category(text: str, category_classifier_pipeline: Pipeline, categories: list[str]) -> str:
    """
    Takes a text string and a list of labels and returns the most likely label
    using a transformer pipeline.

    Parameters:
    text (str): The text to classify.
    category_classifier_pipeline (transformers.base.pipelines.Pipeline): The category classifier pipeline to be used
    categories (list[str]): The categories to be used as labels for classification

    Returns:
    str: The predicted label for the text.
    """
    output = category_classifier_pipeline(text, categories)
    return output["labels"][np.argmax(output["scores"])]

def generate_simple_categories(df: pd.DataFrame, category_classifier_pipeline: Pipeline, categories: list[str]) -> pd.DataFrame:
    """
    Generate simple_categories (fiction/non-fiction) based on the 'description'
    using the generate_simple_category() function.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'description' column.
    category_classifier_pipeline (transformers.base.pipelines.Pipeline): The category classifier pipeline to be used
    categories (list[str]): The categories to be used as labels for classification

    Returns:
    pd.DataFrame: The DataFrame with the 'simple_categories' column updated.
    """
    for i in tqdm(range(len(df))):
        if pd.isna(df.loc[i, "simple_categories"]):
            df.loc[i, "simple_categories"] = generate_simple_category(df.loc[i, "description"], category_classifier_pipeline, categories)

    logger.debug(f"Generated simple categories")

def generate_emotions_(text: str, emotion_classifier_pipeline: Pipeline) -> str:
    """
    Takes a text string and a list of labels and returns the most likely label
    using a transformer pipeline.

    Parameters:
    text (str): The text to classify.
    emotion_classifier_pipeline (transformers.base.pipelines.Pipeline): The emotion classifier pipeline to be used

    Returns:
    str: The predicted label for the text.
    """
    output = emotion_classifier_pipeline(text)
    return output[0]

def generate_emotions(df: pd.DataFrame, emotion_classifier_pipeline: Pipeline) -> pd.DataFrame:
    """
    Generates scores for the emotions fear, neutral, sadness, surprise, disgust, joy and anger based on the 'description'
    using the generate_emotions_() function.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'description' column.
    emotion_classifier_pipeline (transformers.base.pipelines.Pipeline): The emotion classifier pipeline to be used
    
    Returns:
    pd.DataFrame: The DataFrame with the emotions columns added.
    """
    for i in tqdm(range(len(df))):
        text = df.loc[i, "description"]
        emotions = generate_emotions_(text, emotion_classifier_pipeline)
        for emotion in emotions:
            df.loc[i, emotion["label"]]=emotion["score"]

    logger.debug(f"Generated emotions")

def get_chunks(df: pd.DataFrame, columns_for_metadata: list[str], data_column: str, splitter: TextSplitter) -> list[Document]:
    """
    Takes a DataFrame and combines the values in the columns specified
    and splits based on specified TextSplitter and then returns a
    list of Document objects with all columns present in the original
    DataFrame as metadata

    Parameters:
    df (pd.DataFrame): The DataFrame to load
    columns (list[str]): The columns whose values are to be combines
    splitter: The kind of TextSplitter to be used

    Returns:
    list[Document]: The list of Document objects as split into chunks
    """
    df = df[columns_for_metadata]
    docs = DataFrameLoader(df, page_content_column=data_column).load()
    logger.debug(f"Loaded data from dataframe using DataFrameLoader")
    chunks = splitter.split_documents(docs)
    logger.debug(f"Split into chunks using TextSplitter")

    return chunks
from functools import partial
from processing.extract import read_data, read_data_from_vector_database
from processing.load import write_data, write_data_to_vector_db
from processing.transform import remove_null_rows, remove_short_descriptions, join_title_and_subtitle, fix_thumbnail_urls, drop_columns, map_simple_categories, generate_simple_categories, generate_emotions, get_chunks
from processing.models import generate_category_zero_shot_classifer, generate_emotion_classifier
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import torch
import logging
import os
from config.logger import setup_logging
from utils.ui_functions import recommendation_function
import gradio as gr

# Setup
setup_logging()
embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")

# Constants
raw_input_data_path = "data/input/books.csv"
cleaned_data_path = "data/output/cleaned_books.csv"
categorized_data_path = "data/output/categorized_books.csv"
categorized_emotion_added_data_path = "data/output/categorized_emotion_added_books.csv"

columns_to_check = ['num_pages', 'description', 'average_rating', 'published_year']
columns_to_drop = ["title", "subtitle"]

books_categories_map = {
    "Fiction": "fiction",
	"Juvenile Fiction": "fiction",
    "Biography & Autobiography": "non-fiction",
    "History": "non-fiction",
    "Literary Criticism": "non-fiction",
    "Philosophy": "non-fiction",
    "Religion": "non-fiction",
    "Comics & Graphic Novels": "fiction",
    "Drama": "fiction",
    "Juvenile Nonfiction": "non-fiction",
    "Science": "non-fiction",
    "Poetry": "fiction",
    "Literary Collections":	"fiction"
}

categories = ["fiction", "non-fiction"]
category_classifier_model = "facebook/bart-large-mnli"

sentiment_classifier_model = "j-hartmann/emotion-english-distilroberta-base"

vector_db_metadata_columns = ["isbn10", "description", "title_join_subtitle", "authors", "simple_categories", "thumbnail", "published_year", "average_rating", "num_pages", "ratings_count", "fear", "neutral", "sadness", "surprise", "disgust", "joy", "anger"]
vector_db_dir = "data/output/vector_db"


if __name__ == "__main__":
    logger = logging.getLogger("app")

    # Flow control flags
    vector_db_found = False
    categorized_emotion_added_data_found = False
    cleaned_data_found = False
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.debug(f"Device in use for pytorch models: {device}")
    
    # Checks
    logger.debug(f"Checking to see if vector database exists")
    if os.path.isdir(vector_db_dir):
        logger.debug(f"Found vector database, no ETL necessary")
        logger.debug(f"Reading from vector database directory: {vector_db_dir}")
        db = read_data_from_vector_database(vector_db_dir, embeddings)
    else:
        logger.debug(f"Vector database not found")
        logger.debug(f"Checking to see if categorized and emotion added data exists")
        if os.path.isdir(categorized_emotion_added_data_path):
            logger.debug(f"Found categorized and emotion added data")
            df = read_data(categorized_emotion_added_data_path)
        else:
            logger.debug(f"Categorized and emotion added data not found")
            logger.debug(f"Checking to see if cleaned data exists")
            if os.path.isdir(cleaned_data_path):
                logger.debug(f"Found cleaned data")
                df = read_data(cleaned_data_path)
            else:
                logger.debug(f"Cleaned data not found")
                logger.debug(f"Starting basic ETL on data")
                df = read_data(raw_input_data_path)
                df = remove_null_rows(df, columns_to_check)
                df = remove_short_descriptions(df)
                df = join_title_and_subtitle(df)
                df = fix_thumbnail_urls(df)
                df = drop_columns(df, columns_to_drop)
                df = map_simple_categories(df, books_categories_map)
                logger.debug(f"Finished basic ETL on data")
                write_data(df, cleaned_data_path)
                
            logger.debug(f"Starting to generate simple categories")
            category_classifier_pipeline = generate_category_zero_shot_classifer(category_classifier_model, device)
            generate_simple_categories(df, category_classifier_pipeline, categories)
            logger.debug(f"Finished generating simple categories")
            logger.debug(f"Dataframe after generating simple categories:\n{df}")
            logger.debug(f"Starting to generate emotions")
            emotion_classifier_pipeline = generate_emotion_classifier(sentiment_classifier_model, device)
            generate_emotions(df, emotion_classifier_pipeline)
            logger.debug(f"Finished generating emotions")
            logger.debug(f"Dataframe after generating emotions:\n{df}")
            write_data(df, categorized_emotion_added_data_path)

        logger.debug(f"Starting to generate chunks")
        chunks = get_chunks(df, columns_for_metadata=vector_db_metadata_columns, data_column="description", splitter=text_splitter)
        logger.debug(f"Finished generating chunks")
        logger.debug(f"Embedding in vector database")
        db = write_data_to_vector_db(chunks, vector_db_dir, embeddings)
        logger.debug(f"Finished embedding in vector database")

    logger.debug(F"Starting Gradio UI")
    recommendation_function_with_db = partial(recommendation_function, db)
    with gr.Blocks(theme=gr.themes.Default()) as dashboard:
        gr.Markdown("# Book Recommendation System")
        gr.Markdown("### Find your next favorite book based on your mood and preferences!")

        with gr.Row():
            query_input = gr.Textbox(label="What are you looking for?", placeholder="e.g. A zombie thriller where the protagonist can only be awake for 20 minutes at a time")
            category_input = gr.Dropdown(
                label="Fiction, non-fiction, both?",
                choices=["All"] + categories,
                value="All"
            )
            tone_input = gr.Dropdown(
                label="What mood are you in?",
                choices=["All", "Happy", "Surprising", "Angry", "Suspensful", "Sad"],
                value="All"
            )
        initial_k_input = gr.Slider(
            label="Initial recommendations k (must be higher than final recommendations k)",
            minimum=10,
            maximum=100,
            value=50,
            step=1
        )
        final_k_input = gr.Slider(
            label="Final recommendations k",
            minimum=1,
            maximum=20,
            value=16,
            step=1
        )
        submit_button = gr.Button("Show me some books that fit the brief!")

    
        results_output = gr.Gallery(
            label="Top picks",
            columns=8,
            rows=2
        )

        submit_button.click(
            fn=recommendation_function_with_db,
            inputs=[
                query_input,
                category_input,
                tone_input,
                initial_k_input,
                final_k_input
            ],
            outputs=results_output
        )

    dashboard.launch(server_name="0.0.0.0", server_port=7860)
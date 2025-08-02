import pandas as pd
import numpy as np

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr

# Load env variables
load_dotenv()

# Data file path
data_file = "data/cleaned_categorized_emotion_scored_books.csv"
isbn_description_file = "data/isbn_description.txt"

# Read books data
books_df = pd.read_csv(data_file)

# Handle thumbnails
books_df["thumbnail"] = books_df["thumbnail"].apply(
    lambda x: x + "&fife=w800" if pd.notna(x) else "assets/cover-not-found.jpg"
)

# Load documents with TextSplitter, split into chunks with CharacterTextSplitter, embed with OpenAIEmbeddings and store in Chroma
raw_documents = TextLoader(isbn_description_file).load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
chunks = text_splitter.split_documents(raw_documents)
chroma_db_books = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())

# Function to retrieve semantic recommendations
def get_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_k: int = 50,
        final_k: int = 16
) -> pd.DataFrame:
    '''
    Performs similarity search in the Chroma database based on the query and retrieves initial_k results
    Filters retrieved results by category and tone if specified and selects final_k results
    '''

    initial_recommendations = chroma_db_books.similarity_search_with_score(query, k=initial_k)
    isbns = [rec[0].page_content.strip('"').split(" ")[0] for rec in initial_recommendations]
    books_df_filtered = books_df[books_df["isbn10"].isin(isbns)]

    if category != "All":
        books_df_filtered = books_df_filtered[books_df_filtered["simple_categories"] == category]
    
    tone_mapping = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspensful": "fear",
        "Sad": "sadness",
        "All": None
    }

    if tone:
        mapped_tone = tone_mapping.get(tone)
        if mapped_tone:
            books_df_filtered = books_df_filtered.sort_values(by=mapped_tone, ascending=False)

    return books_df_filtered.head(final_k)

# Gradio UI function for recommendations
def recommendation_function(
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_k: int = 50,
        final_k: int = 16
):
    recs = get_semantic_recommendations(
        query=query,
        category=category,
        tone=tone,
        initial_k=initial_k,
        final_k=final_k
    )

    results = []

    for _, row in recs.iterrows():
        title_text = row["title_join_subtitle"]
        desc_text = row["description"].split()[:30] # Split to get first 30 words
        desc_text = " ".join(desc_text) + "..." if len(desc_text) == 30 else " ".join(desc_text)

        authors = row["authors"].split(";")
        authors_text = ""
        if len(authors) == 2:
            authors_text = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            authors_text = f"{', '.join(authors[:-1])} and {authors[-1]}"
        else:
            authors_text = authors[0] if authors else "Unknown"

        caption = title_text + " by " + authors_text + ": " + desc_text

        results.append((row["thumbnail"], caption))

    return results

# Generate Gradio interface
def main():
    with gr.Blocks(theme=gr.themes.Default()) as dashboard:
        gr.Markdown("# Book Recommendation System")
        gr.Markdown("### Find your next favorite book based on your mood and preferences!")

        with gr.Row():
            query_input = gr.Textbox(label="What are you looking for?", placeholder="e.g. A zombie thriller where the protagonist can only be awake for 20 minutes at a time")
            category_input = gr.Dropdown(
                label="Fiction, non-fiction, both?",
                choices=["All"] + sorted(books_df["simple_categories"].unique().tolist()),
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
            fn=recommendation_function,
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

if __name__ == "__main__":
    main()
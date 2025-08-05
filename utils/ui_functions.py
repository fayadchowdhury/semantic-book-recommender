import pandas as pd
from langchain_chroma import Chroma
from utils.docs_to_df import convert_docs_to_dataframe


def get_semantic_recommendations(chroma_db: Chroma, query: str, category: str = "All", tone: str = None, initial_k: int = 50, final_k: int = 16) -> pd.DataFrame:
    """
    Performs similarity search in the Chroma database based on the query and retrieves initial_k results.
    Filters retrieved results by category and tone if specified and selects final_k results.

    Parameters:
    chroma_db (langchain_chroma.Chroma): The Chroma vector database to query
    query (str): The query to run against the database
    category (str): The category to filter recommendations by (fiction/non-fiction/All for now)
    tone (str): The tone to rank recommendations by (joy/surprise/anger/fear/sadness/None)
    initial_k (int): The number of documents to retrieve initially from the database
    final_k (int): The number of documents to show after applying all filters

    Returns:
    pd.DataFrame: The final filtered list of recommendations
    """

    initial_recommendations = chroma_db.similarity_search_with_score(query, k=initial_k)
    books_df_filtered = convert_docs_to_dataframe(initial_recommendations)

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

def recommendation_function(chroma_db: Chroma, query: str, category: str = "All", tone: str = "All", initial_k: int = 50, final_k: int = 16) -> list[tuple[str, str]]:
    """
    Calls the get_semantic_recommendations() function and then extracts the thumbnail and caption to display

    Parameters:
    chroma_db (langchain_chroma.Chroma): The Chroma vector database to query
    query (str): The query to run against the database
    category (str): The category to filter recommendations by (fiction/non-fiction/All for now)
    tone (str): The tone to rank recommendations by (joy/surprise/anger/fear/sadness/None)
    initial_k (int): The number of documents to retrieve initially from the database
    final_k (int): The number of documents to show after applying all filters

    Returns:
    list[tuple[str, str]]: The final list of tuples of (thumbnail_url, caption)
    """
    recs = get_semantic_recommendations(chroma_db=chroma_db, query=query, category=category, tone=tone, initial_k=initial_k, final_k=final_k)

    results = []

    for _, row in recs.iterrows():
        title_text = row["title_join_subtitle"]
        desc_text = row["content"].split()[:30] # Split to get first 30 words
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
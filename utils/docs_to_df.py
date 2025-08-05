import pandas as pd
from langchain_core.documents.base import Document

def convert_docs_to_dataframe(scored_docs: list[tuple[Document, float]]) -> pd.DataFrame:
    """
    Takes a list of scored retrieved documents and converts them into a Pandas DataFrame

    Parameters:
    scored_docs (list[tuple[Document, float]]): The list of scored retrieved documents

    Returns:
    pd.DataFrame: The resultant Pandas DataFrame
    """
    data = []
    for doc, score in scored_docs:
        data.append(
            {
                "content": doc.page_content,
                **doc.metadata,
                "score": score
            }
        )
    
    return pd.DataFrame(data)
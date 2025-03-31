import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

df = pd.read_csv("comics/crawl/stories_updated.csv")

def create_vector_store(db_path: str) -> Chroma:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = []
    for _, row in df.iterrows():
        title = row["tiêu đề"]
        description = row["mô tả"]
        content = f"Tên truyện: {title}. Mô tả: {description}"
        
        documents.append(Document(page_content=content))

    db = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=db_path)
    return db

def retrieve_context(db: Chroma, query: str) -> List[Document]:
    """
    Retrieves relevant document chunks from the Chroma vector store based on a query.

    Parameters:
    db (Chroma): The Chroma vector store containing embedded documents.
    query (str): The query string to search for relevant document chunks.

    Returns:
    List[Document]: A list of retrieved relevant document chunks.
    """

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    print("Relevant chunks are retrieved...\n")
    relevant_chunks = retriever.invoke(query)

    return relevant_chunks

def build_context(relevant_chunks: List[Document]) -> str:
    """
    Builds a context string from retrieved relevant document chunks.

    Parameters:
    relevant_chunks (List[Document]): A list of retrieved relevant document chunks.

    Returns:
    str: A concatenated string containing the content of the relevant chunks.
    """

    print("Context is built from relevant chunks")
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    return context

def get_context(inputs: Dict[str, str]) -> Dict[str, str]:
    query, db_path  = inputs['query'], inputs['db_path']

    # Create new vector store if it does not exist
    if not os.path.exists(db_path):
        print("Creating a new vector store...\n")
        db = create_vector_store(db_path)

    # Load the existing vector store
    else:
        print("Loading the existing vector store\n")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    relevant_chunks = retrieve_context(db, query)
    context = build_context(relevant_chunks)

    return {'context': context, 'query': query}

def main(query):
    current_dir = "content/rag"
    persistent_directory = os.path.join(current_dir, "db", "chroma_db_pdf")
    template = """ Bạn là một mô hình AI được huấn luyện để tìm tên của bộ truyện dựa trên mô tả hoặc gần giống tiêu đề.
    Question : {query}
    \n
    Context : {context}
    \n
    Nếu không tìm thấy bộ truyện nào giống mô tả hoặc giống tên bộ truyện, hãy trả lời: Dữ liệu của tôi chưa có bộ truyện giống mô tả của bạn hoặc đây không phải là một bộ truyện.
    """

    rag_prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model='llama3.1')

    str_parser = StrOutputParser()

    rag_chain = (
        RunnableLambda(get_context)
        | rag_prompt
        | llm
        | str_parser
    )
    
    answer = rag_chain.invoke({'query':query, 'db_path':persistent_directory})
    return answer
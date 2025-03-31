import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain.schema import Document
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from typing import List, Dict
import traceback
from langchain_community.retrievers import BM25Retriever 
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

try:
    df = pd.read_csv("comics/crawl/stories_updated.csv")
    df['tiêu đề'] = df['tiêu đề'].fillna('')
    df['mô tả'] = df['mô tả'].fillna('')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file comics/crawl/stories_updated.csv")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file CSV: {e}")
    exit()

def load_documents_from_dataframe(dataframe: pd.DataFrame) -> List[Document]:
    """Tạo danh sách các đối tượng Document từ DataFrame."""
    documents = []
    for _, row in dataframe.iterrows():
        title = row["tiêu đề"]
        description = row["mô tả"]
        if title or description:
            content = f"Tên truyện: {title}. Mô tả: {description}"
            documents.append(Document(page_content=content))
        else:
            print(f"Cảnh báo: Bỏ qua dòng với tiêu đề và mô tả rỗng: {row.to_dict()}")
    return documents

def create_vector_store(db_path: str, documents: List[Document]) -> Chroma:
    """Tạo và lưu trữ Chroma DB."""
    print("Đang tạo Chroma vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not documents:
         raise ValueError("Không có documents nào để tạo vector store.")
    db = Chroma.from_documents(documents=documents, embedding=embedding_model, persist_directory=db_path)
    print("Tạo Chroma vector store thành công.")
    return db

def build_context(relevant_chunks: List[Document]) -> str:
    """Builds a context string from retrieved relevant document chunks."""
    print("Context is built from relevant chunks")
    if not relevant_chunks:
        return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    return context

def get_combined_context(query: str, db_path: str) -> Dict[str, str]:
    """
    Lấy ngữ cảnh bằng cách kết hợp kết quả từ Chroma (vector) và BM25 (keyword).

    Args:
        query: Chuỗi câu truy vấn của người dùng.
        db_path: Đường dẫn tệp đến thư mục cơ sở dữ liệu vector Chroma.

    Returns:
        Một dictionary chứa 'context' đã được truy xuất và 'query' gốc.
    """
    #Tạo danh sách Document (cần cho cả Chroma và BM25)
    print("Đang tải/chuẩn bị documents từ CSV...")
    try:
        documents = load_documents_from_dataframe(df)
        if not documents:
             raise ValueError("Không tạo được document nào từ file CSV.")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi tải documents từ DataFrame: {e}")

    #Tạo Chroma DB và Retriever
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(db_path) or not os.listdir(db_path): # Kiểm tra thư mục tồn tại và không rỗng
        print(f"Không tìm thấy vector store tại {db_path} hoặc thư mục rỗng. Đang tạo mới...\n")
        try:
             chroma_db = create_vector_store(db_path, documents)
             if chroma_db is None:
                 raise ValueError("Tạo vector store thất bại.")
        except Exception as e:
             raise RuntimeError(f"Lỗi khi tạo vector store: {e}")
    else:
        print(f"Đang tải vector store hiện có từ {db_path}\n")
        try:
            if not os.path.exists(os.path.join(db_path, "chroma.sqlite3")):
                 print(f"Cảnh báo: Thư mục {db_path} không chứa file 'chroma.sqlite3'. Có thể DB bị lỗi hoặc chưa hoàn tất. Thử tải lại...")
            chroma_db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
            
            try:
                 chroma_db.similarity_search("test", k=1)
                 #print("Tải Chroma DB thành công.")
            except Exception as load_test_e:
                 raise RuntimeError(f"Không thể sử dụng Chroma DB tại {db_path}: {load_test_e}")

        except Exception as e:
             import traceback
             print(traceback.format_exc())
             raise RuntimeError(f"Lỗi nghiêm trọng khi tải vector store từ {db_path}: {e}")

    # Tạo Chroma retriever
    chroma_retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 1}) 

    # 3. Tạo BM25 Retriever
    #print("Đang khởi tạo BM25 retriever...")
    try:
        bm25_retriever = BM25Retriever.from_documents(documents=documents)
        bm25_retriever.k = 1
    except Exception as e:
        raise RuntimeError(f"Lỗi khi khởi tạo BM25 retriever: {e}")

    # 4. Tạo Ensemble Retriever để kết hợp
    #print("Đang kết hợp retrievers...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[0.5, 0.5]  # Trọng số
    )

    # 5. Truy xuất sử dụng Ensemble Retriever
    try:
        print(f"Đang truy xuất ngữ cảnh kết hợp cho query: '{query}'")
        relevant_chunks = ensemble_retriever.invoke(query)
        print(f"Đã truy xuất được {len(relevant_chunks)} chunks kết hợp.")
    except Exception as e:
         import traceback
         print(traceback.format_exc())
         raise RuntimeError(f"Lỗi khi truy xuất ngữ cảnh kết hợp: {e}")

    # 6. Xây dựng ngữ cảnh từ kết quả kết hợp
    try:
        context = build_context(relevant_chunks)
    except Exception as e:
        raise RuntimeError(f"Lỗi khi xây dựng ngữ cảnh: {e}")

    return {'context': context, 'query': query}

# --- Hàm main ---
def main(query):
    current_dir = "content/rag"
    persistent_directory = os.path.join(current_dir, "db", "chroma_db_pdf")
    os.makedirs(persistent_directory, exist_ok=True)

    template = """ Bạn là một mô hình AI được huấn luyện để tìm tên của bộ truyện dựa trên mô tả hoặc gần giống tiêu đề.
    Question : {query}
    \n
    Context : {context}
    \n
    Nếu không tìm thấy bộ truyện nào giống mô tả hoặc giống tên bộ truyện, hãy trả lời: Dữ liệu của tôi chưa có bộ truyện giống mô tả của bạn hoặc đây không phải là một bộ truyện. Nếu có nhiều kết quả, hãy liệt kê chúng.
    """

    rag_prompt = ChatPromptTemplate.from_template(template)

    try:
        llm = ChatOllama(model='llama3.1')
        print("Kết nối đến Ollama thành công.")
    except Exception as e:
        print(f"\nLỖI KẾT NỐI OLLAMA: {e}")
        print("Hãy đảm bảo Ollama đang chạy và model 'llama3.1' đã được tải (`ollama run llama3.1`).")
        raise RuntimeError(f"Không thể kết nối đến Ollama: {e}")

    str_parser = StrOutputParser()

    rag_chain = (
        # Sử dụng hàm mới get_combined_context
        RunnableLambda(lambda inputs: get_combined_context(inputs['query'], inputs['db_path']))
        | rag_prompt
        | llm
        | str_parser
    )
    try:
        answer = rag_chain.invoke({'query':query, 'db_path':persistent_directory})
        print("Xử lý hoàn tất.")
        return answer
    except Exception as e:
        print(f"\nLỗi trong quá trình thực thi RAG chain: {e}")
        print(traceback.format_exc())
        return "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý yêu cầu của bạn."

if __name__ == "__main__":
    test_query = "truyện về cậu bé rồng" 
    print(f"--- Bắt đầu thử nghiệm với query: '{test_query}' ---")
    try:
        final_answer = main(test_query)
        print("\n--- Kết quả cuối cùng ---")
        print(final_answer)
    except Exception as e:
        print(f"\n--- Đã xảy ra lỗi không mong muốn trong main: {e} ---")
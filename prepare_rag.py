import os
# Set a user-agent so the web loader identifies itself
os.environ["USER_AGENT"] = "medical-qa-rag/1.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Medical fact sheets from WHO
urls = [
    "https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases",
    "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)"
]

loader = WebBaseLoader(urls)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    docs, embedding_function, persist_directory="./chroma_db"
)
vectorstore.persist()
print("Chroma DB created successfully!")
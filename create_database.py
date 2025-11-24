import os
import nltk
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Paths
DATA_PATH = "data/books"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_documents():
    """Load all .md documents from the data/books directory using UTF-8 encoding."""
    print(f"📚 Loading documents from: {DATA_PATH}")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents.")
    return documents


def split_documents(documents):
    """Split large documents into smaller chunks for embedding."""
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(texts)} text chunks.")
    return texts


def create_embeddings():
    """Create embedding model instance."""
    print("🧠 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


def generate_data_store():
    """Load, split, embed, and store documents into FAISS database."""
    documents = load_documents()
    texts = split_documents(documents)
    embeddings = create_embeddings()

    print("💾 Creating FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ FAISS database saved at: {DB_FAISS_PATH}")


def main():
    generate_data_store()


if __name__ == "__main__":
    main()

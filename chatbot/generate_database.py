from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import PyPDF2  # Use PyPDF2 to process PDF files
import os
import shutil

# Define paths for the Chroma database and dataset
CHROMA_PATH = "chroma"
DATA_PATH = "dataset"

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file at once.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the entire PDF.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle cases where extract_text returns None

    return text

def main():
    """Main function to generate the Chroma database."""
    generate_data_store()

def generate_data_store():
    """Generate and save the Chroma vector database."""
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    """
    Load text from a PDF file as documents.

    Returns:
        list[Document]: List containing the loaded document.
    """
    pdf_path = os.path.join(DATA_PATH, "Harry_Potter.pdf")
    text = extract_text_from_pdf(pdf_path)
    documents = [Document(page_content=text)]
    return documents

def split_text(documents: list[Document]):
    """
    Split the loaded documents into smaller chunks.

    Args:
        documents (list[Document]): List of documents to split.

    Returns:
        list[Document]: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Maximum size of each chunk
        chunk_overlap=500,  # Overlap between chunks to ensure continuity
        length_function=len,
        add_start_index=True,  # Include starting index in metadata
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    """
    Save the text chunks to a Chroma vector database.

    Args:
        chunks (list[Document]): List of text chunks to store.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Clear the existing database if it exists

    embedding_function = OllamaEmbeddings(model="nomic-embed-text")  # Define embedding model
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to Chroma.")

if __name__ == "__main__":
    main()
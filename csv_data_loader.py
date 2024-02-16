from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import threading

# Define paths
data_dir = 'data'
chroma_db = 'vectorstore/db_chroma'

def process_csv_file(csv_path):
    # Load the CSV file using CSVLoader
    loader = CSVLoader(file_path=csv_path, encoding="utf-8", csv_args={'delimiter': ','})
    document = loader.load()
    print(f"Loaded CSV file: {csv_path}")

    # Initialize a text splitter to divide documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"Texts splitted for CSV file: {csv_path}")

    # Initialize HuggingFaceEmbeddings using a specific model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    print(f"Embeddings created for CSV file: {csv_path}")

    # Create a vector store using FAISS from the text chunks and embeddings
    db = Chroma.from_documents(texts, embeddings,persist_directory=chroma_db)
    print(f"Vector store created for CSV file: {csv_path}")

    # # Save the vector store locally
    # db.save_local(os.path.join(chroma_db, f"{os.path.basename(csv_path)}_db"))

def process_csv_files_in_parallel():
    # Ensure the directory exists
    if not os.path.exists(chroma_db):
        os.makedirs(chroma_db)

    # List all CSV files in the directory
    csv_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

    # Create threads for each CSV file processing
    threads = []
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        thread = threading.Thread(target=process_csv_file, args=(csv_path,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    process_csv_files_in_parallel()

# this script is to convert the csv file.
# pipleline: csv -> docs -> embedd -> create vector db -> populate vector db


import csv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def prepare_dataset(file_path):
    documents = []
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            kapampangan_word = row["Kapampangan"].strip()
            english_meaning = row["English"].strip()
            
            content = f"Kapampangan word: {kapampangan_word}. English translation: {english_meaning}."
            
            doc = Document(
                page_content=content,
                metadata={"word": kapampangan_word}
            )
            documents.append(doc)
            
    return documents

# 1. Load the data
csv_path = r"C:\vscode\VectorDB\RAG Kapampangan\data\combined.csv"
docs = prepare_dataset(csv_path)

# 2. Initialize a local, open-source embedding model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create the vector database and save it to your target folder
persist_dir = r"C:\vscode\VectorDB\RAG Kapampangan\data"

doc_ids = [str(i) for i in range(len(docs))]

print("Generating embeddings and saving to ChromaDB...")
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    ids=doc_ids,
    persist_directory=persist_dir
)

print(f"Successfully processed and saved {len(docs)} documents to: {persist_dir}")
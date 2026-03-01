# this script is for checking if the chroma db has successfully initialized

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the exact same embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Point to your saved database folder
persist_dir = r"C:\vscode\VectorDB\RAG Kapampangan\data"
vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# 3. Run a test search
query = "morning"
print(f"Searching for: '{query}'\n" + "-"*30)

# Fetch the top 3 closest matches
results = vector_db.similarity_search(query, k=3)

for i, res in enumerate(results):
    print(f"Match {i+1}: {res.page_content}")
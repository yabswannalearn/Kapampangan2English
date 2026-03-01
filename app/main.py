from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI()

class QueryRequests(BaseModel):
    query:str

print("Loading Rag pipeline into memory")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = r"C:\vscode\VectorDB\RAG Kapampangan\data"
vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="kapampangan-model:latest")

system_prompt = (
    "You are an expert Kapampangan translator. "
    "Use the provided dictionary definitions to answer the user's question. "
    "Provide only the translation or definition.\n\n"
    "Translate sentences if the input is a sentence\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("RAG Pipeline Ready")

# API endpoints:

@app.get("/")
def check():
    return {"status": "Ready"}

@app.post("/translate")
def translate(request: QueryRequests):
    try: 
        response = rag_chain.invoke({"input": request.query})
        return {
            "query": request.query,
            "translations": response["answer"].strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
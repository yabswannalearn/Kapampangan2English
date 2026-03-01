from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Load the Vector Database as a Retriever
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = r"C:\vscode\VectorDB\RAG Kapampangan\data"
vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Retrieve the top 3 most relevant dictionary entries
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 2. Initialize Your Local LLM
# NOTE: Replace this with your actual fine-tuned model setup
from langchain_community.llms import Ollama
llm = Ollama(model="kapampangan-model") 



# 3. Create the Prompt Template
# This instructs the model to strictly use the retrieved dictionary context
system_prompt = (
    "You are an expert Kapampangan translator. "
    "Use the provided dictionary definitions to answer the user's question. "
    "Provide only the translation or definition.\n\n"
    "Translate sentences if the input is a sentence"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 4. Construct the RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. Test the Pipeline
print("\nModel loaded successfully! Type 'exit' or 'quit' to stop.")
print("-" * 50)

while True:
    query = input("\nEnter your Kapampangan translation question: ")
    
    if query.lower() in ['exit', 'quit']:
        print("Shutting down...")
        break
        
    print("Generating Answer...")
    response = rag_chain.invoke({"input": query})
    
    print("\nLLM Response:")
    print(response["answer"])
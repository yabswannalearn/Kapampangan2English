import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Diagnostic check
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: PyTorch is not detecting your GPU!")

# 1. Load the Vector Database as a Retriever
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = r"C:\vscode\VectorDB\RAG Kapampangan\data"
vector_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})


# 2. Initialize the Qwen Model
print("Loading Qwen model...")
model_id = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"":"cuda:0"} # Automatically uses GPU if available
)

# Create the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,
    return_full_text=False, # stop echoing the prompt
    temperature=0.1,
    pad_token_id=tokenizer.eos_token_id # end of sequence
)


# add a stop sequence to prevent babbling
# babbling - llm is talking to itself
llm = HuggingFacePipeline(pipeline=pipe).bind(stop=["\n\n", "Context:"])

# 3. Prompt Template

prompt_template = """You are a precise Kapampangan translator. Use ONLY the provided context to answer the question. 

Context:
{context}

Question: {input}
Answer:"""

prompt = PromptTemplate.from_template(prompt_template)

# 4. Construct the RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# 5. Testing the Pipeline
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
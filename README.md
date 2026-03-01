# Kapampangan2English: Fine-Tuned SLM with RAG
An end-to-end pipeline for translating and defining Kapampangan terms using a Fine-Tuned Small Language Model (SLM) augmented by a local Vector Database.



### Overview
This project addresses the limited linguistic support for Kapampangan in mainstream LLMs. By combining a fine-tuned Qwen-1.7B model with a Retrieval-Augmented Generation (RAG) pipeline, the system provides precise translations grounded in verified dictionary data. The application is served via a FastAPI backend and consumed through a Streamlit web interface.

### Technical Stack
* **LLM**: Qwen3-1.7B (Fine-tuned & Quantized to 4-bit GGUF).
* **Orchestration**: LangChain.
* **Inference**: Ollama.
* **Vector DB**: ChromaDB.
* **Embeddings**: all-MiniLM-L6-v2 (Hugging Face).
* **Backend API**: FastAPI & Uvicorn.
* **Frontend UI**: Streamlit.
* **Data Scraping**: Selenium & BeautifulSoup.
* **Data Cleaning**: LLM-assisted normalization (Claude).

---

## The Fine-Tuning Process
This project utilizes a custom fine-tuned model optimized for low-resource Philippine languages. You can find the detailed code and process for the fine-tuning phase here:
**[Fine-Tuned-Translator Repository](https://github.com/yabswannalearn/Fine-Tuned-Translator)**

### Training Parameters & Logic
These parameters were specifically chosen to optimize performance on a GTX 1650:

| Parameter | Value | Technical Rationale |
| :--- | :--- | :--- |
| **Rank (r)** | 16 | Controls the number of trainable parameters in LoRA adapters. 16 is the "sweet spot" for small models to learn new language patterns without overfitting. |
| **Alpha** | 32 | A scaling factor for the weight updates. Setting this to 2x the Rank ensures the learned Kapampangan nuances have a strong influence on the base model's output. |
| **Learning Rate**| 2e-4 | A moderate rate that allows the model to converge on specific translation tasks without "catastrophic forgetting" of its base reasoning capabilities. |
| **Quantization** | 4-bit | Reduces the VRAM requirement by ~70% using BitsAndBytes, allowing a 1.7B model to be trained and run on 4GB consumer GPUs. |
| **Context Length**| 2048 | Optimized to save memory for the RAG "context stuffing" while still allowing for long sentence translations. |

### Dataset Strategy
* **Normalization**: Input data is structured as high-quality Instruction-Response pairs to teach the model a specific "Translator" persona.
* **Multi-tasking**: The model is prepared for expansion into other dialects (like Cebuano) by using specific language identifiers in the training prompt.

---

### Project Structure
* **Data Scraped/**: Python scrapers and raw/processed CSV datasets.
* **scripts/**: 
    * **csv-to-doc.py**: Populates the ChromaDB vector store.
    * **RAG-Finetuned.py**: The terminal-based interactive RAG loop.
* **main.py**: FastAPI application serving the RAG pipeline as a REST endpoint.
* **app.py**: Streamlit frontend providing a web interface for the API.
* **model/**: Contains the Modelfile and fine-tuned weights configuration.

### Key Learnings
* **Architecture**: Decoupling the application into a standalone API backend (FastAPI) and a client frontend (Streamlit) for scalability.
* **Model Optimization**: Fine-tuning domain-specific datasets and converting to GGUF for local inference.
* **Contextual Grounding**: Using RAG to mitigate hallucinations by providing a source of truth via vector search.
* **Hardware Acceleration**: Configuring torch and transformers to utilize CUDA on a GTX 1650.
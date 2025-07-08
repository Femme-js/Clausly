# 🤖 Clausly

**AI-Powered Legal Document Review & Clause Explanation Assistant**

Clausly is a multi-agent GenAI assistant that simplifies the understanding of legal documents. Whether you're reviewing contracts or trying to decode legal jargon, Clausly helps by intelligently answering questions, summarizing clauses, and explaining terms in plain English — all powered by LLMs and Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 📄 **Document Ingestion**  
  Upload contracts, NDAs, or lease agreements (PDF/DOCX), which are parsed and indexed for semantic search.

- 🧭 **Multi-Agent Routing:** 
    Clausly intelligently routes questions between two types of sources:

    📄 Contract-specific answers using vector search over the uploaded document

    🌐 General legal definitions using public knowledge (e.g., Wikipedia)

- 🔍 **Smart Clause Detection**  
  Automatically extracts key clauses like termination, indemnity, payment terms, etc.

- 💬 **Context-Aware QA**  
  Ask questions like “What is the payment clause?” or “What does indemnification mean?”

- 🧠 **Plain English Mode**  
  Get simplified explanations of complex legal terms and clauses.

- 🚩 **Risk Flagging (Beta)**  
  Highlights potentially risky or unusual terms for further review.
 

---

## 🧑‍💻 Tech Stack

- [LangChain](https://www.langchain.com/) + [LangGraph](https://github.com/langchain-ai/langgraph)  
- [Groq LLM](https://groq.com/) / [OpenAI](https://openai.com/) for routing and explanation  
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- [Cassandra](https://www.datastax.com/astra) / [FAISS](https://github.com/facebookresearch/faiss) for vector search  
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) or `python-docx` for parsing  
- Optional: [FastAPI](https://fastapi.tiangolo.com/) UI

---

## 📦 Use Cases

- Legal document review for non-lawyers  
- AI assistant for startups and founders signing agreements  
- Learning legal concepts and contracts  
- Paralegal or compliance support tools  

---

## 🧪 Example Queries

```bash
"What is the payment obligation in this contract?"
"Is there a non-compete clause?"
"Explain indemnity in plain English."
"What does 'force majeure' mean?"

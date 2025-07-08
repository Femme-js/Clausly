import os
from dotenv import load_dotenv

load_dotenv()

def get_config():
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "model_name": "Llama-3.1-70b-Versatile",
        "embedding_model": "all-MiniLM-L6-v2",
        "cass_keyspace": None,
        "cass_session": None,
        "cass_table": "qa_mini_demo",
    }
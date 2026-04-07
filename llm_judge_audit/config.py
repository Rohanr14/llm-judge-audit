import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

config = Config()
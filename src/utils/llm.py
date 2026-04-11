import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(
    model=os.getenv("PRIMARY_MODEL", "gpt-4o"),
    temperature=0,  # deterministic for legal analysis
    max_retries=3,
)

llm_fast = ChatOpenAI(
    model=os.getenv("FAST_MODEL", "gpt-4o-mini"),
    temperature=0,
    max_retries=3,
)

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("No GEMINI_API_KEY found")
    exit(1)

print(f"Testing LangChain Gemini with key: {api_key[:10]}...")

try:
    # Try gemini-1.5-flash
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    res = llm.invoke("Hello")
    print("gemini-1.5-flash Worked:", res.content)
except Exception as e:
    print("gemini-1.5-flash Failed:", str(e))

try:
    # Try gemini-2.0-flash (which was in the list)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    res = llm.invoke("Hello")
    print("gemini-2.0-flash Worked:", res.content)
except Exception as e:
    print("gemini-2.0-flash Failed:", str(e))

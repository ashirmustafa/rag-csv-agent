import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

def run():
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in a .env file or directly.")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    try:
        response = llm.invoke("What is GEN AI?")
        print(f"Response: {response.content}\n")
    except Exception as e:
        print(f"Error in Test: {e}\n")

    print("Testing completed")


import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in a .env file or directly.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

print("--- Test 1: Simple Text Generation ---")
try:
    response = llm.invoke("What is the capital of France?")
    print(f"Response: {response.content}\n")
except Exception as e:
    print(f"Error in Test 1: {e}\n")

print("--- Test 2: Using a ChatPromptTemplate ---")
try:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant."),
            ("human", "Tell me a fun fact about the universe."),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({})
    print(f"Response: {response.content}\n")
except Exception as e:
    print(f"Error in Test 2: {e}\n")

print("--- Test 3: Multimodal (Vision) Capability (Optional) ---")
try:
    llm_vision = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5) 

    image_url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"

    message_with_image = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": image_url},
        ]
    )

    response_vision = llm_vision.invoke([message_with_image])
    print(f"Vision Response: {response_vision.content}\n")

except Exception as e:
    print(f"Error in Test 3 (Multimodal): {e}")
    print("Ensure you are using a multimodal model (e.g., 'gemini-1.5-flash' or 'gemini-pro-vision') and that the image URL is accessible.")

print("--- Testing complete ---")
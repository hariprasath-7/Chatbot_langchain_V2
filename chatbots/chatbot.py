import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Match the working model and your .env key name
api_key = os.getenv("Google_API_Key") or os.getenv("GOOGLE_GENAI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key
)

# Chat history maintained as a standard list of messages
chat_history = [
    SystemMessage(content="You are a helpful and polite AI assistant.")
]

while True:
    user_input = input("User: ")
    
    if user_input.strip().lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break

    # Append user's message to the history
    chat_history.append(HumanMessage(content=user_input))

    # Invoke the model with the entire history
    result = model.invoke(chat_history)

    # Append AI's response to the history
    chat_history.append(AIMessage(content=result.content))

    print(f"AI: {result.text}\n")

print("\n--- Final Chat History ---")
for msg in chat_history:
    print(f"{msg.type.capitalize()}: {msg.content}")
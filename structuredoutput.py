from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv(override=True)

class Review(TypedDict):
    summary: str
    sentiment: str


model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.2,
    max_output_tokens=1024,
)

structured_model = model.with_structured_output(Review)

prompt = "Please provide a brief summary and sentiment analysis of the following review: 'The product quality is excellent, but the delivery was delayed.'"

result = structured_model.invoke(prompt)
print(result)
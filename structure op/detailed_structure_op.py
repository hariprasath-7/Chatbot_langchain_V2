from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv(override=True)

class Review(TypedDict):
    key_themes: Annotated[list[str], "A list of key themes or topics discussed in the review"]
    summary: Annotated[str, "A brief 1-2 sentence summary of the review"]
    sentiment: Annotated[str, "Sentiment of the review: Positive, Negative, or Mixed"]
    pros: Annotated[list[str], "A list of positive aspects mentioned in the review"]
    cons: Annotated[list[str], "A list of negative aspects mentioned in the review"]

# Free tier allowed, separate quota pool from 3.8-flash
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.2,
    max_output_tokens=1024,
)

structured_model = model.with_structured_output(Review)

prompt = "Please provide a brief summary and sentiment analysis of the following review: 'The product quality is excellent, but the delivery was delayed.' highlight the key themes, pros, and cons mentioned in the review. products was delayed, but the quality was good. it is a good product, but the delivery was not satisfactory. The customer is happy with the product quality but disappointed with the delivery time. the product name is 'SuperWidget 3000'."

result = structured_model.invoke(prompt)
print(result)
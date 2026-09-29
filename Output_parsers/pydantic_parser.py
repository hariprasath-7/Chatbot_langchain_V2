from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser, OutputFixingParser, PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.2,
    max_output_tokens=1024,
)

class Person(BaseModel):
    name: str = Field(..., description="The person's name")
    age: int = Field(gt=18, lt=120, description="The person's age")
    email: str = Field(..., description="The person's email address")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""
    You are a helpful assistant. Please provide the information in the specified format.
    {format_instructions}
    """,
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': "Provide a person's name, age, and email address."})
print(result)

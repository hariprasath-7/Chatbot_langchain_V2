from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser, OutputFixingParser

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.2,
    max_output_tokens=1024,
)


schema=[
    ResponseSchema(name="fact1",description="first fact about black hole"),
    ResponseSchema(name="fact2",description="second fact about black hole"),
    ResponseSchema(name="fact3",description="third fact about black hole")
]

parser=StructuredOutputParser.from_response_schemas(schema)

safe_parser=OutputFixingParser.from_llm(llm=model,parser=parser)

template=PromptTemplate(
    template="""
    give me 3 facts about {topic}. Return only valid json instructions that follow this format\n
    {response_format}
    """,
    input_variables=['topic'],
    partial_variables={"response_format":parser.get_format_instructions()}
)

chain=template | model | parser

result=chain.invoke({"topic":"black hole"})
print(result)
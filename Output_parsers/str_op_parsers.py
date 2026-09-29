from langchain_google_genai import ChatGoogleGenerativeAI   
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser      
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.2,
    max_output_tokens=1024,
)

# 1 prompt 
template1 = PromptTemplate(
    template="You are a helpful {domain} expert. Explain in simple terms, the concept of {topic}, in a way that a 10-year-old can understand.", 
    input_variables=["domain", "topic"]
)

# 2 prompt
template2 = PromptTemplate(
    template="Summarize the following explanation into 3 bullet points:\n\n{text}", 
    input_variables=["text"]
) 

parser = StrOutputParser()

# Fixed: wrap the string output from parser into {"text": lambda x: x} for template2
chain = template1 | model | parser | {"text": lambda x: x} | template2 | model | parser

result = chain.invoke({
    'domain': "quantum physics",
    'topic': "entanglement"
})

# Fixed: result is already a str, no .text attribute needed
print(result)
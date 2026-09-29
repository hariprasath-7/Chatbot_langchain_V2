from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.2,
    max_output_tokens=1024,
    timeout=30,
)

prompt1 = PromptTemplate(
    template="You are a helpful {domain} expert. Explain in simple terms, the concept of {topic}, in a way that a 10-year-old can understand.",
    input_variables=["domain", "topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following explanation into 3 bullet points:\n\n{text}",     
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | {"text": lambda x: x} | prompt2 | model | parser 

result = chain.invoke({
    'domain': "quantum physics",
    'topic': "entanglement"
})  

print(result)

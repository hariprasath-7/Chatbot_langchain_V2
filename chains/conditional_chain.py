from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel,Field
from typing import Literal


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.8-flash")

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description="The sentiment of the feedback provided")

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="classify the sentiment of following feedback text into positive or negative  {feedback} and provide the response in following format: {response_format}",
    input_variables=['feedback'],
    partial_variables={"response_format":parser2.get_format_instructions()}
)

classifier_chain=prompt1 | model | parser2 | (lambda x: x.sentiment)

prompt2=PromptTemplate(template="write an appropriate response to this positive feedback {feedback}",
    input_variables=['feedback']
)

prompt3=PromptTemplate(template="write an appropriate response to this negative feedback {feedback}",
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x: x["sentiment"] == "positive", prompt2 | model | parser),
    (lambda x: x["sentiment"] == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "No valid sentiment found")
)

chain=RunnablePassthrough.assign(sentiment=classifier_chain) | branch_chain


result=chain.invoke({"feedback":"The phone is actually horrible. the ui is stuck"})
print(result)
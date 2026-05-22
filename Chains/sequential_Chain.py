from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template= 'give a detailed report on {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template= 'generate a 5 point summary from the following text \n {text}',
    input_variables= ['text']
)

Model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

Parser = StrOutputParser()

chain = prompt1 | Model | Parser | prompt2 | Model | Parser

result = chain.invoke({ 'topic' : 'Improvement of Civic Sense in India'})

print (result)

chain.get_graph().print_ascii_ascii()
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI      #Gemini model integration for LangChain.
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template= 'Generate 5 short interesting facts about {topic}',
    input_variables= ['topic']
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()      #Parser converts model response.

chain = prompt | model | parser      # LCEL syntax (LangChain Expression Language)

result = chain.invoke({'topic' : 'cricket'})

print(result)

chain.get_graph().print_ascii()
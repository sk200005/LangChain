
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 1.8, max_completion_tokens = 15)

result = model.invoke("give a one liner reply to bully")

print(result.content)
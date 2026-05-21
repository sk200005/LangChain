from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

result = client.text_generation(
    "What is the capital of Uganda?",
    model="gpt2"
)

print(result)
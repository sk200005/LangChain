from langchain_google_genai import ChatGoogleGenerativeAI  #LangChain wrapper for Google Gemini models.
from dotenv import load_dotenv                             #Loads environment variables from a .env file.
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()   #Reads .env

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


class Review(BaseModel):          # Always return output in this format.

    key_theme: list[str] = Field(
        description="Write down all key themes discussed in the review"
    )

    summary: str = Field(
        description="A brief summary of the review"
    )

    sentiment: str = Field(
        description="Return overall sentiment: positive, negative or neutral"
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="Write all pros inside the list"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="Write all cons inside the list"
    )


structured_model = model.with_structured_output(Review)    #Convert model into structured model

result = structured_model.invoke(
    "I’ve been using the Google Pixel 8 and overall I’m very happy with it. "
    "The camera quality is excellent — photos look natural with great colors and impressive low-light performance. "
    "The phone feels premium in hand and the compact size makes it comfortable for daily use. "
    "The clean Android experience is one of the best parts: smooth, simple, and without unnecessary apps. "
    "Features like AI photo editing and voice tools are actually useful. "
    "There are a few things that could be better. "
    "Battery life is good for normal use but not exceptional if you use the camera or mobile data heavily. "
    "Charging speed is also slower compared to many phones in the same price range. "
    "During long gaming sessions, the phone can get a little warm."
)

print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
print("Key Theme:", result.key_theme)
print("Pros:", result.pros)
print("Cons:", result.cons)
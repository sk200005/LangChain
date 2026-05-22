from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Used to execute multiple chains togetherin parallel
from langchain.schema.runnable import RunnableParallel

load_dotenv()
model1 = ChatOpenAI()

model2 = ChatAnthropic(
    model_name='claude-3-7-sonnet-20250219'
)


prompt1 = PromptTemplate(     # Generate short notes

    template='''
Generate short and simple notes
from the following text \n {text}''',

    input_variables=['text']
)

prompt2 = PromptTemplate(             # Generate quiz questions

    template='''
Generate 5 short question answers
from the following text
{text}
''',

    input_variables=['text']
)


prompt3 = PromptTemplate(                 # Merge outputs Receives: notes & quiz

    template='''
Merge the provided notes
and quiz into a single document

notes ->
{notes}

quiz ->
{quiz}
''',

    input_variables=[
        'notes',
        'quiz'
    ]
)

parser = StrOutputParser()                   


parallel_chain = RunnableParallel({           # ******   Runs BOTH simultaneously  ****** #

    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

 

merge_chain = (    prompt3 | model1 | parser    )


# =====================================================
# COMPLETE PIPELINE
#
# Parallel Execution
#          ↓
# Merge Results
# =====================================================
chain = (  parallel_chain | merge_chain )

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

Advantages:
Effective in high dimensional spaces.

Uses support vectors.

Memory efficient.

Supports custom kernels.

Disadvantages:

Can overfit.

No direct probability estimates.

Requires dense/sparse compatible input.
"""


result = chain.invoke({

    'text': text
})


print(result)

chain.get_graph().print_ascii()
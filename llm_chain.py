from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline

def load_llm():
    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=pipe)


def get_chain():
    llm = load_llm()

    prompt = PromptTemplate(
        input_variables=["image_context", "question"],
        template="""
You are an image analysis assistant.

Image description:
{image_context}

User question:
{question}

Answer clearly and concisely.
"""
    )

    return llm, prompt
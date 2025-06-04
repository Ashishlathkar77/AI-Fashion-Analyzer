from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

def get_fashion_advice(colors, size_info, detected_items):
    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["colors", "size_info", "detected_items"],
        template="""
        You are an AI fashion assistant. Analyze the following data:
        - Colors: {colors}
        - Size Info: {size_info}
        - Detected Clothing: {detected_items}

        Generate:
        - A product description
        - An outfit suggestion
        - A social media caption
        """
    )

    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Invoke the chain
    response = chain.run({
        "colors": colors,
        "size_info": size_info,
        "detected_items": detected_items
    })

    return response
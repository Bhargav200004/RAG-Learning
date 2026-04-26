import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


""" LLM model creations"""
model = init_chat_model(
    model="gemini-3-flash-preview",
    model_provider="google-genai",
    api_key=GEMINI_API_KEY
)

"""Tool for the the water creation"""
def get_water_consumption(age : str) -> str:
    """Get water consumption for given age"""
    int_age = int(age)
    if int_age in range(0 , 2):
        return "Breast Milk + 0.5 liters of water consumption for both boys and girls"
    elif int_age in range(1 , 4):
        return "4 cups of water (Around 1 liter of water consumption)"
    elif int_age in range(4, 9):
        return "5 cups of water (Around 1.2 <-> 1.7 liter of water consumption)"
    elif int_age in range(9, 14):
        return "7 cups pf water for girls(Around 1.4 <-> 2.1 liter of water consumption) \n 8 cups pf water for boys(Around 1.5 <-> 2.4 liter of water consumption)"
    elif int_age in range(14, 18):
        return "7 to 8 cups pf water for girls(Around 1.6 <-> 2.2 liter of water consumption) \n 8 to 11 cups pf water for boys(Around 2 <-> 3 liter of water consumption)"
    else:
        return "Women 9 cups of water consumption (Around 2.2 <-> 2.7 liter of water consumption) \n 12 to 13 cups of water consumption for men (Around 3+ liter of water consumption)"

"""Agent creation """
agent = create_agent(
    model = model,
    tools=[get_water_consumption],
    system_prompt="You are a water consumption checker whose help to tell about how much water should human consume through age"
)

"""Agent message array"""
messages = [
        {
            "role" : "user",
            "content" : "How much water should i consume at age of 15?"
        },
]


"""Agent invocation with message array"""
result = agent.invoke(
    {
        "messages" : messages
    }
)

print(result["messages"][-1].content_blocks)



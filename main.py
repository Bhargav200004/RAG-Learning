import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model = init_chat_model(
    model="gemini-3-flash-preview",
    model_provider="google-genai",
    api_key=GEMINI_API_KEY
)

def get_weather(city: str) -> str:
    """Get weather date for given city"""
    return f"It's always sunny in {city}"

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant"
)


# {"messages": [{"role": "user", "content": "What's the weather in Bilaspur?"}]}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Bilaspur?"}]}
)
print(result["messages"][-1].content_blocks)

print(result)

import os

import langchain
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import AIMessage , HumanMessage , ToolMessage
import langchain
from langchain.agents.middleware import wrap_tool_call

from model.train_response import Model as TrainResponse

load_dotenv()
langchain.debug = True
agent_debug = False

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
X_RAPIDAPI_KEY = os.getenv("X_RAPIDAPI_KEY")

SYSTEM_PROMPT = """ 
You are an Indian Railway Enquiry Clerk.

### Core Directive
You CANNOT answer train schedule questions from your own memory. If a user asks about a train, you MUST ALWAYS use the `get_train_details` tool first to fetch the accurate data.

### Response Formatting
ONLY AFTER you have successfully used the tool and received the data, format your final response to the user. 
At the very end of your final response, you must generate 2 to 3 follow-up questions to keep the conversation going. 

Format the follow-up questions exactly like this:
1. Total time duration to complete the journey?
2. Does the train have a pantry car?
3. [Insert another relevant question based on the train data]
"""


model = init_chat_model(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    model_provider="openrouter",
    temperature=0.3,
    api_key=OPENROUTER_API_KEY
)

# Tool to fetch the train details
def get_train_details(train_number : str) -> str:
    """
    Fetches detailed information and the complete running schedule for a specific train number.

    Args:
        train_number (str): A 5-digit Indian Railway train number (e.g. '12809').

    Returns:
        str: A structured string containing train metadata (name, origin, destination, journey classes)
            and a 'shedule' list containing all station stop with arriaval and departure times.

    Agent Instructions:
        - The returned shedule data is extremely long. DO NOT output the raw data dump to the user.
        - Parse the response internally to extract only the information necessary to answer the user's specific prompt.
        - For example, if the user asks for a specific station, only return the arrival/departure for that station.
        - If the user asks for the general route, simmarize the origin, destinaton, and a few major stops rather than listing all 40+ stations.
    """

    print(f"DEBUG: Tool triggered by agent with train number: {train_number}")

    try:
        url = f"https://indian-railway-irctc.p.rapidapi.com/api/trains-search/v1/train/{train_number}"

        querystring = {"isH5": "true", "client": "web"}

        headers = {
            "x-rapidapi-key": X_RAPIDAPI_KEY,
            "x-rapidapi-host": "indian-railway-irctc.p.rapidapi.com",
            "Content-Type": "application/json",
            "x-rapid-api": "rapid-api-database"
        }

        response   = requests.get(url, headers=headers, params=querystring)

        train_response : TrainResponse = TrainResponse.model_validate(response.json())

        return str(train_response.body[0].trains[0])
    except Exception as e:
        print(e)
        return "sorry not able to fetch details of train"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model = model,
    tools=[get_train_details],
    middleware=[handle_tool_errors],
    system_prompt = SYSTEM_PROMPT
)

message = [
    {
        "role" : "user",
        "content" : "Tell me details about this train => 12102 ."
    }
]

if __name__ == "__main__":
    for chunk in agent.stream(
            input={
                "message" : message,
            } ,
            stream_mode="values"
    ):
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, HumanMessage):
                print(f"User: {latest_message.content}")
            elif isinstance(latest_message, ToolMessage) and agent_debug:
                print(f"Tool: {latest_message.content}")
            elif isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
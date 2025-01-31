from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel

# Initialize the OpenAI model
model = OpenAIModel(model_name='gpt-4o')

# System prompt for the chatbot
SYSTEM_PROMPT = """
You are a psychological consultant chatbot. Your role is to provide thoughtful, reflective, 
and supportive responses. You help users explore their thoughts, emotions, and challenges 
without providing direct medical advice. Keep responses conversational and empathetic.
"""

# Create an agent
agent = Agent(model)

# Define the input model
class ChatRequest(BaseModel):
    message: str

# Define the output model
class ChatResponse(BaseModel):
    response: str

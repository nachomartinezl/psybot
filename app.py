from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agent import agent, ChatRequest, ChatResponse  # Import agent and models

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate response using the agent
        run_result = await agent.run(request.message)

        # Extract the response text from RunResult
        if hasattr(run_result, "data"):
            ai_response = run_result.data  # Correct way to extract text
        else:
            raise ValueError("AI response does not contain 'data' field")

        return ChatResponse(response=ai_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

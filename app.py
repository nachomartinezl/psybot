from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from agent import agent, ChatRequest, ChatResponse  # Import agent and models
from twilio.twiml.messaging_response import MessagingResponse

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
async def chat(request: Request):
    try:
        form = await request.form()
        print("Received Twilio Data:", form)  # Debugging line
        
        sender = form.get("From")
        message = form.get("Body")

        if not sender or not message:
            raise HTTPException(status_code=400, detail="Invalid request: Missing sender or message")
        
        # Generate AI response using the agent
        run_result = await agent.run(message)
        if hasattr(run_result, "data"):
            ai_response = run_result.data  # Extract response
        else:
            raise ValueError("AI response does not contain 'data' field")
        
        # Generate Twilio TwiML Response
        twiml_response = MessagingResponse()
        twiml_response.message(ai_response)
        
        return PlainTextResponse(str(twiml_response), media_type="application/xml")
    
    except Exception as e:
        print("Error Processing Request:", str(e))  # Debugging line
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
def home():
    return {"message": "WhatsApp Bot is Running!"}

# Run locally with:
# uvicorn whatsapp_bot:app --host 0.0.0.0 --port 8000 --reload

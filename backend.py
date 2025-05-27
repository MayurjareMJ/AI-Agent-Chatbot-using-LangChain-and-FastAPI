from pydantic import BaseModel
from typing import List

from pydantic import BaseModel

class RequestState(BaseModel):
    model_name: str  # ✅ Correct
    model_provider: str  # ✅ Correct 
    system_prompt: str  # ✅ Correct
    messages: List[str]  # ✅ Correct
    allow_search: bool  # ✅ Correct

# class RequestState(BaseModel):
#     model_name=str
#     model_provider:str
#     system_prompt :str
#     messages:List[str]
#     allow_search: bool




from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent
ALLOWED_MODEL_NAMES=["gpt-3.5-turbo", "gpt-4", "groq-llm"]


app=FastAPI(title="LangGraph AI Agent")


app=FastAPI(title="LanGraph Ai Agent")

@app.post("/chat")
def chat_endpoint(request:RequestState):
    """
    API Endpoint to intreact with the Chatbot using Langchain and search tools.
    It Dynamically selects the model specifics in the request
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error":"Invalid model name. Kindly select a valid AI model."}
    llm_id=request.model_name
    query=request.messages
    allow_search=request.allow_search
    system_prompt=request.system_prompt
    provider=request.model_provider
    # Check if the model name is allowed
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Please select a valid AI model."}     
    
    
    # Create the AI agent with the selected model
    response=get_response_from_ai_agent(
        llm_id=request.model_name,
        query=request.messages,
        allow_search=request.allow_search,
        system_prompt=request.system_prompt,
        provider=request.model_provider
    )

    llm_id=request.model_name
    query=request.messages
    allow_search=request.allow_serach
    system_prompt=request.system_prompt
    provider=request.model_provider
    

    response =get_response_from_ai_agent(llm_id,query,allow_search)
    return response



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)














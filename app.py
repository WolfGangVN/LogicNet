import asyncio
import time
import fastapi
import cachetools
import logging

logging.basicConfig(level=logging.INFO)
import openai
from pydantic import BaseModel
app = fastapi.FastAPI()
cache = cachetools.TTLCache(maxsize=1024, ttl=60)
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
openai_client = openai.AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-1234567890",
)
class ChatRequest(BaseModel):
    message: str
    
async def generate(messages: list):
    try:
        import time
        starttime = time.time()
        response = await openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=2048,
            temperature=0.8,
        )
        logging.info(f"Generated in {time.time() - starttime} seconds")
        return response
    except Exception as e:
        logging.error(f"Error in forward: {e}")
        return None        
    
async def chat_handler(question: str):
    try:
        logging.info(f"Received question: {question}")
        reasoning_messages = [
            {"role": "user", "content": question},
        ]
        logging.info(f"Starting reasoning...")
        reasoning_response = await generate(reasoning_messages)
        logic_reasoning = reasoning_response.choices[0].message.content
        
        full_messages = reasoning_messages + [
            {"role": "assistant", "content": logic_reasoning},
            {
                "role": "user",
                "content": "Give me the final short answer as a sentence. Don't reasoning anymore, just say the final answer in math latex.",
            },
        ]    
        
        logging.info(f"Starting full response...")
        full_response = await generate(full_messages)
        logic_answer = full_response.choices[0].message.content
        return {
            "logic_reasoning": logic_reasoning,
            "logic_answer": logic_answer
        }
    except Exception as e:
        logging.error(f"Error in chat_handler: {e}")
        return {
            "logic_reasoning": None,
            "logic_answer": None
        }
    

def hash_key(request: ChatRequest):
    import hashlib
    return hashlib.md5(request.message.encode()).hexdigest()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        starttime = time.time()
        key = hash_key(request)
        
        if key in cache:
            logging.info(f"Cache hit for {request.message}")
            
            while True:
                if cache[key] is not None:
                    logging.info(f"Returning cached response for {request.message}, took {time.time() - starttime} seconds")
                    return cache[key]
                
                if time.time() - starttime > 20:
                    logging.info(f"Timeout for {request.message}")
                    return {
                        "logic_reasoning": None,
                        "logic_answer": None
                    }      
                await asyncio.sleep(0.2)
            
        logging.info(f"Cache miss for {request.message}")
        cache[key] = None   
        response = await chat_handler(request.message)
        cache[key] = response
        logging.info(f"Returning response for {request.message}, took {time.time() - starttime} seconds")
        return response
    except Exception as e:
        logging.error(f"Error in chat: {e}")
        return {
            "logic_reasoning": None,
            "logic_answer": None
        }
        
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=51321)

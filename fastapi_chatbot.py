from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import asyncio
from datetime import datetime
import logging
import os
from langgraph_new import (
    process_guest_query,
    MockHotelService,
    GuestState,
    Intent
)
from langchain.prompts import ChatPromptTemplate


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hotel Services Chatbot API",
    description="API for interacting with the hotel services chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize hotel service
hotel_service = MockHotelService()

# Models for request/response
class ChatMessage(BaseModel):
    message: str
    guest_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    hotel_code: Optional[str] = "12939"  # Add default hotel code
    user_id: Optional[str] = "589164"   # Add default user ID

class ChatResponse(BaseModel):
    response: str
    metadata: Dict[str, Any]
    timestamp: str
    insights_data: Optional[Dict[str, Any]] = None  # Add insights data field

# Store conversation history
class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_message(self, guest_id: str, message: str, response: Dict[str, Any]):
        if guest_id not in self.conversations:
            self.conversations[guest_id] = []
        
        self.conversations[guest_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "response": response
        })
    
    def get_history(self, guest_id: str) -> List[Dict[str, Any]]:
        return self.conversations.get(guest_id, [])

# Initialize conversation manager
conversation_manager = ConversationManager()

# --- Custom Exceptions ---
class InsightsError(Exception):
    """Base exception for insights-related errors"""
    pass

class InsightsDataError(InsightsError):
    """Raised when there's an error fetching or processing insights data"""
    pass

class InsightsCategoryError(InsightsError):
    """Raised when there's an error with insights category"""
    pass

class InsightsAuthError(InsightsError):
    """Raised when there's an authentication/authorization error with insights"""
    pass

# --- Error Handling Functions ---
def handle_insights_error(e: Exception) -> Dict[str, Any]:
    """Handle insights-specific errors and return appropriate error response"""
    if isinstance(e, InsightsDataError):
        return {
            "error": "Failed to fetch insights data",
            "details": str(e),
            "type": "insights_data_error",
            "status_code": 503
        }
    elif isinstance(e, InsightsCategoryError):
        return {
            "error": "Invalid insights category",
            "details": str(e),
            "type": "insights_category_error",
            "status_code": 400
        }
    elif isinstance(e, InsightsAuthError):
        return {
            "error": "Authentication error accessing insights",
            "details": str(e),
            "type": "insights_auth_error",
            "status_code": 401
        }
    else:
        return {
            "error": "Unexpected error processing insights",
            "details": str(e),
            "type": "insights_unknown_error",
            "status_code": 500
        }

def validate_insights_response(response: Any) -> bool:
    """Validate the format of insights response"""
    if response is None:
        return False
    if isinstance(response, str):
        # Check if the string contains meaningful content
        return bool(response.strip())
    if isinstance(response, list):
        # Check if list contains any non-empty strings
        return any(isinstance(item, str) and bool(item.strip()) for item in response)
    if isinstance(response, dict):
        # Check if dict contains any non-empty string values
        return any(isinstance(v, str) and bool(v.strip()) for v in response.values())
    return False

@app.get("/")
async def read_root():
    """Serve the chat interface."""
    return FileResponse("static/index.html")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """
    Process a chat message and return a response.
    Maintains conversation history for each guest.
    """
    try:
        # Validate hotel and user credentials
        if not chat_message.hotel_code or not chat_message.user_id:
            raise InsightsAuthError("Hotel code and user ID are required for insights")

        # Prepare metadata with hotel and user info
        metadata = chat_message.metadata or {}
        if chat_message.guest_id:
            metadata["guest_id"] = chat_message.guest_id
        metadata["hotel_code"] = chat_message.hotel_code
        metadata["user_id"] = chat_message.user_id
        
        # Process the query with visualization enabled
        try:
            result = await process_guest_query(
                question=chat_message.message,
                metadata=metadata,
                hotel_service=hotel_service,
                visualize=True,
                graph_filename_prefix="fastapi_graph"
            )
        except Exception as e:
            logger.error(f"Error in process_guest_query: {str(e)}", exc_info=True)
            if "insights" in chat_message.message.lower():
                raise InsightsDataError(f"Failed to process insights request: {str(e)}")
            raise
        
        # Log the final response sent to the UI
        logger.info(f"Final response sent to UI: {result['response']}")
        
        # Handle insights-specific response processing
        insights_data = None
        insights_error = None
        try:
            if "responses" in result.get("metadata", {}):
                insights_response = result["metadata"]["responses"].get("INSIGHTS")
                if insights_response:
                    # Validate insights response using the new validation function
                    if not validate_insights_response(insights_response):
                        raise InsightsDataError("Empty or invalid insights response")
                    
                    # Process the insights response
                    if isinstance(insights_response, dict):
                        # If it's a dict, use the first non-empty string value
                        insights_text = next((v for v in insights_response.values() 
                                           if isinstance(v, str) and v.strip()), None)
                    elif isinstance(insights_response, list):
                        # If it's a list, join non-empty strings
                        insights_text = "\n".join(item for item in insights_response 
                                                if isinstance(item, str) and item.strip())
                    else:
                        # If it's a string, use it directly
                        insights_text = insights_response
                    
                    insights_data = {
                        "insights": insights_text,
                        "category": result["metadata"].get("insights_category"),
                        "timestamp": datetime.utcnow().isoformat(),
                        "hotel_code": chat_message.hotel_code,
                        "user_id": chat_message.user_id
                    }
                    
                    # Log insights generation
                    logger.info(f"Generated insights for hotel {chat_message.hotel_code}")
                    
        except Exception as e:
            logger.error(f"Error processing insights data: {str(e)}", exc_info=True)
            insights_error = handle_insights_error(e)
            if 'error' in result.get('metadata', {}):
                logger.info("Using fallback response due to insights error")
        
        # Add to conversation history if guest_id is provided
        if chat_message.guest_id:
            conversation_manager.add_message(
                guest_id=chat_message.guest_id,
                message=chat_message.message,
                response=result
            )
        
        # Prepare response
        response = ChatResponse(
            response=result["response"],
            metadata={
                **result["metadata"],
                "insights_error": insights_error  # Include any insights errors
            },
            timestamp=datetime.utcnow().isoformat(),
            insights_data=insights_data
        )
        
        return response
        
    except InsightsError as e:
        error_data = handle_insights_error(e)
        logger.error(f"Insights error: {error_data}", exc_info=True)
        raise HTTPException(
            status_code=error_data["status_code"],
            detail=error_data
        )
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

@app.get("/conversation/{guest_id}")
async def get_conversation_history(guest_id: str):
    """
    Retrieve conversation history for a specific guest.
    """
    try:
        history = conversation_manager.get_history(guest_id)
        return {
            "guest_id": guest_id,
            "history": history,
            "message_count": len(history)
        }
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation history: {str(e)}"
        )

# WebSocket endpoint for real-time chat
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, guest_id: str):
        await websocket.accept()
        self.active_connections[guest_id] = websocket
    
    def disconnect(self, guest_id: str):
        if guest_id in self.active_connections:
            del self.active_connections[guest_id]
    
    async def send_message(self, guest_id: str, message: Dict[str, Any]):
        if guest_id in self.active_connections:
            await self.active_connections[guest_id].send_json(message)

connection_manager = ConnectionManager()

@app.websocket("/ws/{guest_id}")
async def websocket_endpoint(websocket: WebSocket, guest_id: str):
    """
    WebSocket endpoint for real-time chat functionality.
    Maintains a persistent connection for each guest.
    """
    await connection_manager.connect(websocket, guest_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process the message
            try:
                # Validate hotel and user credentials
                hotel_code = message_data.get("hotel_code", "12939")
                user_id = message_data.get("user_id", "589164")
                if not hotel_code or not user_id:
                    raise InsightsAuthError("Hotel code and user ID are required for insights")

                # Prepare metadata with hotel and user info
                metadata = message_data.get("metadata", {})
                metadata["guest_id"] = guest_id
                metadata["hotel_code"] = hotel_code
                metadata["user_id"] = user_id
                
                # Process the query
                try:
                    result = await process_guest_query(
                        question=message_data["message"],
                        metadata=metadata,
                        hotel_service=hotel_service,
                        visualize=False
                    )
                except Exception as e:
                    logger.error(f"Error in process_guest_query: {str(e)}", exc_info=True)
                    if "insights" in message_data["message"].lower():
                        raise InsightsDataError(f"Failed to process insights request: {str(e)}")
                    raise
                
                # Handle insights-specific response processing
                insights_data = None
                insights_error = None
                try:
                    if "responses" in result.get("metadata", {}):
                        insights_response = result["metadata"]["responses"].get("INSIGHTS")
                        if insights_response:
                            # Validate insights response using the new validation function
                            if not validate_insights_response(insights_response):
                                raise InsightsDataError("Empty or invalid insights response")
                            
                            # Process the insights response
                            if isinstance(insights_response, dict):
                                # If it's a dict, use the first non-empty string value
                                insights_text = next((v for v in insights_response.values() 
                                                   if isinstance(v, str) and v.strip()), None)
                            elif isinstance(insights_response, list):
                                # If it's a list, join non-empty strings
                                insights_text = "\n".join(item for item in insights_response 
                                                        if isinstance(item, str) and item.strip())
                            else:
                                # If it's a string, use it directly
                                insights_text = insights_response
                            
                            insights_data = {
                                "insights": insights_text,
                                "category": result["metadata"].get("insights_category"),
                                "timestamp": datetime.utcnow().isoformat(),
                                "hotel_code": hotel_code,
                                "user_id": user_id
                            }
                            
                            # Log insights generation
                            logger.info(f"Generated insights for hotel {hotel_code}")
                            
                except Exception as e:
                    logger.error(f"Error processing insights data: {str(e)}", exc_info=True)
                    insights_error = handle_insights_error(e)
                
                # Add to conversation history
                conversation_manager.add_message(
                    guest_id=guest_id,
                    message=message_data["message"],
                    response=result
                )
                
                # Send response back to client
                await connection_manager.send_message(
                    guest_id,
                    {
                        "type": "response",
                        "data": {
                            "response": result["response"],
                            "metadata": {
                                **result["metadata"],
                                "insights_error": insights_error  # Include any insights errors
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                            "insights_data": insights_data
                        }
                    }
                )
                
            except InsightsError as e:
                error_data = handle_insights_error(e)
                logger.error(f"Insights error in WebSocket: {error_data}", exc_info=True)
                await connection_manager.send_message(
                    guest_id,
                    {
                        "type": "error",
                        "data": error_data
                    }
                )
            except Exception as e:
                logger.error(f"Error processing websocket message: {str(e)}", exc_info=True)
                await connection_manager.send_message(
                    guest_id,
                    {
                        "type": "error",
                        "data": {
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                )
                
    except WebSocketDisconnect:
        connection_manager.disconnect(guest_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        connection_manager.disconnect(guest_id)

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API status.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "hotel_chatbot"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
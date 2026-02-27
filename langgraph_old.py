from typing import Annotated, Dict, List, Optional, TypedDict, Union, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
import logging
from datetime import datetime
from enum import Enum
import asyncio
from dataclasses import dataclass
from typing import Literal
import httpx
from functools import lru_cache
import os
from dotenv import load_dotenv
from IPython.display import Image, display


# --- Load Environment Variables ---
load_dotenv()

# --- OpenAI Configuration (use env; never commit real keys) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")  # or "gpt-3.5-turbo" for testing

# Initialize OpenAI client with API key
@lru_cache()
def get_openai_client() -> ChatOpenAI:
    """Get cached OpenAI client instance."""
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0  # Default temperature, can be overridden per call
    )

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants and Enums ---
class Intent(str, Enum):
    QA = "QA"  # Question Answering
    ORE = "ORE"  # Offer Generation
    CHECKIN = "CHECKIN"  # Check-in Process
    CHECKOUT = "CHECKOUT"  # Check-out Process
    CONCIERGE = "CONCIERGE"  # Concierge Services
    FEEDBACK = "FEEDBACK"  # Guest Feedback
    LOST_FOUND = "LOST_FOUND"  # Lost & Found

# --- State Management ---
@dataclass
class GuestState(TypedDict):
    """State maintained throughout the guest interaction workflow."""
    guest_question: Annotated[str, "The original guest query", {"aggregate": "last"}]  # Use last value in concurrent updates
    intents: Annotated[List[Intent], "Detected intents"]
    active_agents: Annotated[List[str], "Agents that processed the query", {"aggregate": "extend"}]
    responses: Annotated[Dict[str, Dict[str, str]], "Agent responses keyed by agent and intent", {"aggregate": "merge"}]
    metadata: Annotated[Dict[str, Any], "Additional context"]
    timestamp: Annotated[str, "When the interaction started"]
    processing_complete: Annotated[bool, "Flag to track processing status"] = False
    intent_confidence: Annotated[Dict[str, float], "Confidence scores for intents"] = Field(default_factory=dict)
    primary_intent: Annotated[Optional[Intent], "Primary intent"] = None
    final_response: Annotated[str, "Final response from the merged agent"] = None

def create_initial_state(guest_question: str, metadata: Optional[Dict] = None) -> GuestState:
    """Initialize a new guest interaction state."""
    return GuestState(
        guest_question=guest_question,
        intents=[],
        active_agents=[],
        responses={},
        metadata=metadata or {},
        timestamp=datetime.utcnow().isoformat(),
        processing_complete=False
    )

# --- Mock Hotel Service ---
class MockHotelService:
    """Mock service providing dummy hotel data for testing."""
    
    def __init__(self):
        # Dummy data for testing
        self.mock_rooms = {
            "2024-03-20": [
                {"room_number": "101", "type": "Deluxe", "status": "available", "price": 150},
                {"room_number": "102", "type": "Suite", "status": "available", "price": 250},
                {"room_number": "103", "type": "Deluxe", "status": "occupied", "price": 150},
            ],
            "2024-03-21": [
                {"room_number": "101", "type": "Deluxe", "status": "available", "price": 150},
                {"room_number": "102", "type": "Suite", "status": "available", "price": 250},
                {"room_number": "103", "type": "Deluxe", "status": "available", "price": 150},
            ]
        }
        
        self.mock_guests = {
            "G12345": {
                "guest_id": "G12345",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1234567890",
                "preferences": {
                    "room_type": "Deluxe",
                    "floor_preference": "high",
                    "special_requests": "Extra pillows"
                },
                "loyalty_points": 1500,
                "stay_history": [
                    {"check_in": "2024-02-01", "check_out": "2024-02-03", "room_type": "Deluxe"},
                    {"check_in": "2023-12-15", "check_out": "2023-12-20", "room_type": "Suite"}
                ]
            },
            "G12346": {
                "guest_id": "G12346",
                "name": "Jane Smith",
                "email": "jane.smith@example.com",
                "phone": "+1987654321",
                "preferences": {
                    "room_type": "Suite",
                    "floor_preference": "low",
                    "special_requests": "Early check-in"
                },
                "loyalty_points": 2500,
                "stay_history": [
                    {"check_in": "2024-01-15", "check_out": "2024-01-20", "room_type": "Suite"}
                ]
            }
        }
        
        self.mock_offers = [
            {
                "id": "OFFER001",
                "name": "Weekend Getaway",
                "description": "Special weekend rates with complimentary breakfast",
                "valid_dates": {"start": "2024-03-01", "end": "2024-04-30"},
                "discount": "20%",
                "conditions": "Minimum 2-night stay"
            },
            {
                "id": "OFFER002",
                "name": "Loyalty Member Special",
                "description": "Extra 10% off for loyalty members",
                "valid_dates": {"start": "2024-03-01", "end": "2024-12-31"},
                "discount": "10%",
                "conditions": "Loyalty membership required"
            },
            {
                "id": "OFFER003",
                "name": "Early Bird Booking",
                "description": "Book 30 days in advance and save",
                "valid_dates": {"start": "2024-03-01", "end": "2024-12-31"},
                "discount": "15%",
                "conditions": "Book at least 30 days in advance"
            }
        ]
    
    async def get_room_availability(self, date: str) -> Dict[str, Any]:
        """Get mock room availability for a specific date."""
        try:
            # Simulate API delay
            await asyncio.sleep(0.5)
            return {
                "date": date,
                "available_rooms": self.mock_rooms.get(date, []),
                "total_rooms": len(self.mock_rooms.get(date, [])),
                "available_count": sum(1 for room in self.mock_rooms.get(date, []) if room["status"] == "available")
            }
        except Exception as e:
            logger.error(f"Error fetching mock room availability: {str(e)}")
            return {"error": str(e), "available_rooms": []}
    
    async def get_guest_profile(self, guest_id: str) -> Dict[str, Any]:
        """Get mock guest profile information."""
        try:
            # Simulate API delay
            await asyncio.sleep(0.5)
            guest_info = self.mock_guests.get(guest_id, {})
            if not guest_info:
                return {"error": "Guest not found", "guest_info": {}}
            return {"guest_info": guest_info}
        except Exception as e:
            logger.error(f"Error fetching mock guest profile: {str(e)}")
            return {"error": str(e), "guest_info": {}}
    
    async def get_special_offers(self, guest_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get mock special offers, optionally personalized for a guest."""
        try:
            # Simulate API delay
            await asyncio.sleep(0.5)
            offers = self.mock_offers.copy()
            
            # Add personalized offers for specific guests
            if guest_id and guest_id in self.mock_guests:
                guest = self.mock_guests[guest_id]
                if guest["loyalty_points"] > 2000:
                    offers.append({
                        "id": "OFFER004",
                        "name": "VIP Member Exclusive",
                        "description": "Special rates for VIP members",
                        "valid_dates": {"start": "2024-03-01", "end": "2024-12-31"},
                        "discount": "25%",
                        "conditions": "VIP membership required"
                    })
            
            return offers
        except Exception as e:
            logger.error(f"Error fetching mock special offers: {str(e)}")
            return []

# --- Enhanced Intent Classification ---
class IntentClassification(BaseModel):
    """Schema for LLM-based intent classification."""
    intents: List[str] = Field(description="List of detected intents")
    confidence: Dict[str, float] = Field(description="Confidence scores for each intent")
    primary_intent: str = Field(description="The most relevant intent")

def classify_intents(state: GuestState) -> GuestState:
    """
    Rule-based intent classification as a fallback when LLM classification fails.
    Uses simple keyword matching to identify intents.
    """
    question = state["guest_question"].lower()
    
    # Define keyword mappings for each intent
    intent_keywords = {
        Intent.QA: ["what", "how", "when", "where", "why", "can you", "tell me", "explain"],
        Intent.ORE: ["offer", "deal", "promotion", "discount", "special", "package", "rate", "price"],
        Intent.CHECKIN: ["check in", "check-in", "arrival", "register", "registration"],
        Intent.CHECKOUT: ["check out", "check-out", "departure", "leave", "exit"],
        Intent.CONCIERGE: ["concierge", "help", "assist", "service", "book", "reserve", "arrange"],
        Intent.FEEDBACK: ["feedback", "review", "comment", "complaint", "suggestion", "rating"],
        Intent.LOST_FOUND: ["lost", "found", "missing", "misplaced", "forgot", "left behind"]
    }
    
    # Initialize confidence scores
    confidence_scores = {}
    detected_intents = []
    
    # Check for each intent's keywords
    for intent, keywords in intent_keywords.items():
        score = sum(1 for keyword in keywords if keyword in question) / len(keywords)
        if score > 0:
            detected_intents.append(intent)
            confidence_scores[intent.value] = score
    
    # If no intents detected, default to QA
    if not detected_intents:
        detected_intents = [Intent.QA]
        confidence_scores[Intent.QA.value] = 1.0
    
    # Update state
    state["intents"] = detected_intents
    state["intent_confidence"] = confidence_scores
    state["primary_intent"] = detected_intents[0]  # First detected intent as primary
    
    logger.info(f"Rule-based classified intents: {[i.value for i in detected_intents]} with confidence: {confidence_scores}")
    return state

async def llm_classify_intents(state: GuestState) -> GuestState:
    """
    Use LLM to classify guest queries into intents with confidence scores.
    Falls back to rule-based classification if LLM fails.
    """
    try:
        # Get cached OpenAI client with lower temperature for more consistent output
        llm = get_openai_client().with_config({"temperature": 0.1})
        
        # Construct prompt for intent classification with explicit JSON format
        messages = [
            SystemMessage(content="""You are a JSON-only response system for classifying hotel guest queries.
            Your task is to analyze the guest's query and return a JSON object with the detected intents.
            
            Valid intents are: QA, ORE, CHECKIN, CHECKOUT, CONCIERGE, FEEDBACK, LOST_FOUND
            
            You MUST return a JSON object in this exact format:
            {
                "intents": ["INTENT1", "INTENT2"],
                "confidence": {"INTENT1": 0.9, "INTENT2": 0.8},
                "primary_intent": "INTENT1"
            }
            
            Example:
            Input: "What time is check-in and do you have any special offers?"
            Output: {"intents": ["CHECKIN", "ORE"], "confidence": {"CHECKIN": 0.9, "ORE": 0.8}, "primary_intent": "CHECKIN"}
            
            Rules:
            1. Return ONLY the JSON object
            2. No other text or formatting
            3. Must be valid JSON
            4. Must include all required fields
            5. Intents must be from the valid list above
            6. Confidence scores should be between 0 and 1
            7. Primary intent should be the most relevant intent"""),
            HumanMessage(content=state["guest_question"])
        ]
        
        # Generate classification with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get response and log it for debugging
                result = await llm.ainvoke(messages)
                response_text = result.content.strip()
                logger.info(f"Raw LLM response (attempt {attempt + 1}): {response_text}")
                
                # Try to parse JSON directly
                try:
                    classification = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on attempt {attempt + 1}: {str(e)}\nResponse text: {response_text}")
                    if attempt == max_retries - 1:
                        raise
                    continue
                
                # Validate required fields
                required_fields = ["intents", "confidence", "primary_intent"]
                if not all(field in classification for field in required_fields):
                    logger.warning(f"Missing required fields on attempt {attempt + 1}. Got: {list(classification.keys())}")
                    if attempt == max_retries - 1:
                        raise ValueError("Missing required fields in classification")
                    continue
                
                # Validate intent values
                valid_intents = {intent.value for intent in Intent}
                if not all(intent in valid_intents for intent in classification["intents"]):
                    logger.warning(f"Invalid intent values on attempt {attempt + 1}. Got: {classification['intents']}, Valid: {valid_intents}")
                    if attempt == max_retries - 1:
                        raise ValueError("Invalid intent values in classification")
                    continue
                
                # Create a new state object with the classification
                new_state = GuestState(
                    guest_question=state["guest_question"],
                    intents=[Intent(intent) for intent in classification["intents"]],
                    active_agents=state["active_agents"].copy(),
                    responses=state["responses"].copy(),
                    metadata=state["metadata"].copy(),
                    timestamp=state["timestamp"],
                    processing_complete=state["processing_complete"],
                    intent_confidence=classification["confidence"],
                    primary_intent=Intent(classification["primary_intent"]),
                    final_response=None
                )
                
                logger.info(f"LLM classified intents: {classification['intents']} with confidence: {classification['confidence']}")
                return new_state
                
            except Exception as e:
                logger.warning(f"Retry {attempt + 1}/{max_retries} due to: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                continue
                
    except Exception as e:
        logger.error(f"LLM intent classification failed: {str(e)}")
        # Fallback to rule-based classification
        return classify_intents(state)
    
    return state

# --- Additional Agent Nodes ---
async def concierge_agent_node(state: GuestState) -> Dict:
    """Handle concierge service requests and assistance."""
    if Intent.CONCIERGE not in state["intents"]:
        return {"__merge__": {"responses": {}, "active_agents": []}}

    try:
        llm = get_openai_client()
        
        # Get guest context if available
        guest_context = ""
        if "guest_id" in state["metadata"]:
            mock_service = MockHotelService()
            guest_profile = await mock_service.get_guest_profile(state["metadata"]["guest_id"])
            if "guest_info" in guest_profile:
                guest_context = f"\nGuest Context: {json.dumps(guest_profile['guest_info'])}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hotel concierge assistant. Provide concise assistance in 2-3 sentences.
            Focus on the most relevant information and next steps. Keep responses brief and actionable."""),
            ("human", "{question}")
        ])
        
        response = await (prompt | llm).ainvoke({"question": state["guest_question"]})
        
        return {
            "__merge__": {
                "responses": {Intent.CONCIERGE: response.content},
                "active_agents": ["concierge_agent"]
            }
        }
    except Exception as e:
        logger.error(f"Error in concierge agent: {str(e)}")
        return {
            "__merge__": {
                "responses": {Intent.CONCIERGE: "I apologize, but I'm having trouble processing your concierge request. Please visit the concierge desk for assistance."},
                "active_agents": ["concierge_agent"]
            }
        }

async def feedback_agent_node(state: GuestState) -> Dict:
    """Handle guest feedback, reviews, and complaints."""
    if Intent.FEEDBACK not in state["intents"]:
        return {"__merge__": {"responses": {}, "active_agents": []}}

    try:
        llm = get_openai_client()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hotel feedback management assistant. Provide brief, empathetic responses in 2-3 sentences.
            Acknowledge feedback and provide next steps concisely. Keep it professional and to the point."""),
            ("human", "{question}")
        ])
        
        response = await (prompt | llm).ainvoke({"question": state["guest_question"]})
        
        return {
            "__merge__": {
                "responses": {Intent.FEEDBACK: response.content},
                "active_agents": ["feedback_agent"]
            }
        }
    except Exception as e:
        logger.error(f"Error in feedback agent: {str(e)}")
        return {
            "__merge__": {
                "responses": {Intent.FEEDBACK: "Thank you for your feedback. I apologize, but I'm having trouble processing it right now. Please email us at feedback@hotel.com or speak with our manager."},
                "active_agents": ["feedback_agent"]
            }
        }

async def lost_found_agent_node(state: GuestState) -> Dict:
    """Handle lost and found item reports and queries."""
    if Intent.LOST_FOUND not in state["intents"]:
        return {"__merge__": {"responses": {}, "active_agents": []}}

    try:
        llm = get_openai_client()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hotel lost and found assistant. Provide clear, concise guidance in 2-3 sentences.
            Focus on essential next steps and procedures. Keep responses brief and actionable."""),
            ("human", "{question}")
        ])
        
        response = await (prompt | llm).ainvoke({"question": state["guest_question"]})
        
        return {
            "__merge__": {
                "responses": {Intent.LOST_FOUND: response.content},
                "active_agents": ["lost_found_agent"]
            }
        }
    except Exception as e:
        logger.error(f"Error in lost & found agent: {str(e)}")
        return {
            "__merge__": {
                "responses": {Intent.LOST_FOUND: "I apologize, but I'm having trouble processing your lost and found request. Please visit the front desk for immediate assistance."},
                "active_agents": ["lost_found_agent"]
            }
        }

# --- Agent Nodes ---
async def qa_agent_node(state: GuestState) -> Dict:
    """Handle general questions about the hotel, services, policies, etc."""
    logger.info(f"QA agent processing state: {json.dumps(state, default=str)}")
    
    if Intent.QA not in state["intents"]:
        logger.info("QA intent not found, skipping QA agent")
        return {"__merge__": {"responses": {}, "active_agents": []}}

    try:
        llm = get_openai_client()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful hotel concierge. Provide clear, concise answers in 2-3 sentences maximum.
            Focus on the most relevant information only. Avoid unnecessary details or explanations."""),
            ("human", "{question}")
        ])
        
        # Generate response
        logger.info("Generating QA response...")
        chain = prompt | llm
        response = await chain.ainvoke({"question": state["guest_question"]})
        logger.info(f"QA response generated: {response.content[:100]}...")
        
        result = {
            "__merge__": {
                "responses": {Intent.QA: response.content},
                "active_agents": ["qa_agent"]
            }
        }
        logger.info(f"QA agent returning: {json.dumps(result, default=str)}")
        return result
    except Exception as e:
        logger.error(f"Error in QA agent: {str(e)}", exc_info=True)
        result = {
            "__merge__": {
                "responses": {Intent.QA: "I apologize, but I'm having trouble processing your question right now. Please try again or contact the front desk."},
                "active_agents": ["qa_agent"]
            }
        }
        logger.info(f"QA agent returning error response: {json.dumps(result, default=str)}")
        return result

async def ore_agent_node(state: GuestState) -> Dict:
    """Generate personalized offers or promotions based on guest context."""
    if Intent.ORE not in state["intents"]:
        return {"__merge__": {"responses": {}, "active_agents": []}}

    try:
        llm = get_openai_client().with_config({"temperature": 0.7})
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hotel sales agent. Provide a brief, personalized offer in 2-3 sentences.
            Include only the most relevant offer details and any key conditions. Keep it concise and compelling."""),
            ("human", "{question}")
        ])
        
        response = await (prompt | llm).ainvoke({"question": state["guest_question"]})
        
        return {
            "__merge__": {
                "responses": {Intent.ORE: response.content},
                "active_agents": ["ore_agent"]
            }
        }
    except Exception as e:
        logger.error(f"Error in offer generation agent: {str(e)}")
        return {
            "__merge__": {
                "responses": {Intent.ORE: "I apologize, but I'm unable to generate an offer at this time. Please contact our sales team for assistance."},
                "active_agents": ["ore_agent"]
            }
        }

async def checkin_agent_node(state: GuestState) -> Dict:
    """Handle check-in related queries and processes."""
    if Intent.CHECKIN not in state["intents"]:
        return {"__merge__": {"responses": {}, "active_agents": []}}

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hotel check-in assistant. Provide clear, concise check-in information in 2-3 sentences.
            Focus on essential details like timing, requirements, and next steps. Avoid lengthy explanations."""),
            ("human", "{question}")
        ])
        
        response = await (prompt | llm).ainvoke({"question": state["guest_question"]})
        
        return {
            "__merge__": {
                "responses": {Intent.CHECKIN: response.content},
                "active_agents": ["checkin_agent"]
            }
        }
    except Exception as e:
        logger.error(f"Error in check-in agent: {str(e)}")
        return {
            "__merge__": {
                "responses": {Intent.CHECKIN: "I apologize, but I'm having trouble processing your check-in request. Please visit the front desk for assistance."},
                "active_agents": ["checkin_agent"]
            }
        }

async def checkout_agent_node(state: GuestState) -> Dict:
    """Handle check-out related queries and processes."""
    if Intent.CHECKOUT not in state["intents"]:
        return {"__merge__": {"responses": {}, "active_agents": []}}

    try:
        llm = get_openai_client()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a hotel check-out assistant. Provide brief, clear check-out information in 2-3 sentences.
            Include only essential details about timing, procedures, and requirements. Keep it concise."""),
            ("human", "{question}")
        ])
        
        response = await (prompt | llm).ainvoke({"question": state["guest_question"]})
        
        return {
            "__merge__": {
                "responses": {Intent.CHECKOUT: response.content},
                "active_agents": ["checkout_agent"]
            }
        }
    except Exception as e:
        logger.error(f"Error in check-out agent: {str(e)}")
        return {
            "__merge__": {
                "responses": {Intent.CHECKOUT: "I apologize, but I'm having trouble processing your check-out request. Please visit the front desk for assistance."},
                "active_agents": ["checkout_agent"]
            }
        }

# --- Merge Node ---
async def merge_responses(state: GuestState) -> GuestState:
    """Combine responses from all active agents into a coherent message."""
    try:
        logger.info(f"Starting merge_responses with state: {json.dumps(state, default=str)}")
        
        # Flatten responses from all agents
        all_responses = {}
        for agent_responses in state["responses"].values():
            all_responses.update(agent_responses)
        logger.info(f"Flattened responses: {json.dumps(all_responses, default=str)}")
        
        if not all_responses:
            logger.warning("No responses found to merge")
            state["final_response"] = "I apologize, but no responses were generated. Please try again or contact the front desk."
            state["processing_complete"] = True
            return state
        
        # Sort responses by intent priority
        priority_order = {
            Intent.CHECKIN: 1,
            Intent.CHECKOUT: 2,
            Intent.ORE: 3,
            Intent.QA: 4,
            Intent.CONCIERGE: 5,
            Intent.FEEDBACK: 6,
            Intent.LOST_FOUND: 7
        }
        
        # Combine responses in priority order
        sorted_responses = sorted(
            all_responses.items(),
            key=lambda x: priority_order.get(x[0], 999)
        )
        logger.info(f"Sorted responses: {json.dumps(sorted_responses, default=str)}")
        
        # Modify the response format to be more concise
        combined_response = "\n".join(
            f"{intent.value}: {response.strip()}"  # Remove extra newlines and make it more compact
            for intent, response in sorted_responses
        )
        
        # Make the agent summary more concise
        agent_summary = f"\n(Generated by: {', '.join(state['active_agents'])})"
        
        # Create new state with the more concise response
        new_state = GuestState(
            guest_question=state["guest_question"],
            intents=state["intents"],
            active_agents=state["active_agents"],
            responses=state["responses"],
            metadata=state["metadata"],
            timestamp=state["timestamp"],
            processing_complete=True,
            intent_confidence=state["intent_confidence"],
            primary_intent=state["primary_intent"],
            final_response=combined_response + agent_summary
        )
        
        logger.info(f"Successfully merged responses. Final response length: {len(new_state['final_response'])}")
        logger.info(f"Active agents: {new_state['active_agents']}")
        logger.info(f"Processing complete: {new_state['processing_complete']}")
        
        return new_state
        
    except Exception as e:
        logger.error(f"Error in merge_responses: {str(e)}", exc_info=True)
        logger.error(f"State at error: {json.dumps(state, default=str)}")
        # Create error state with final_response
        error_state = GuestState(
            guest_question=state["guest_question"],
            intents=state["intents"],
            active_agents=state["active_agents"],
            responses=state["responses"],
            metadata=state["metadata"],
            timestamp=state["timestamp"],
            processing_complete=True,
            intent_confidence=state["intent_confidence"],
            primary_intent=state["primary_intent"],
            final_response="I apologize, but I'm having trouble combining the responses. Please try again or contact the front desk."
        )
        return error_state

async def update_state_with_agent_response(state: GuestState, agent_response: Dict) -> GuestState:
    """Update state with agent response data."""
    logger.info(f"Updating state with agent response: {json.dumps(agent_response, default=str)}")
    
    if "__merge__" in agent_response:
        merge_data = agent_response["__merge__"]
        # Update responses
        if "responses" in merge_data:
            logger.info(f"Processing responses from merge data: {json.dumps(merge_data['responses'], default=str)}")
            for intent, response in merge_data["responses"].items():
                if intent not in state["responses"]:
                    state["responses"][intent] = {}
                    logger.info(f"Created new response entry for intent: {intent}")
                state["responses"][intent].update({intent: response})
                logger.info(f"Updated response for intent {intent}")
        
        # Update active agents
        if "active_agents" in merge_data:
            logger.info(f"Processing active agents from merge data: {merge_data['active_agents']}")
            state["active_agents"].extend(merge_data["active_agents"])
            logger.info(f"Updated active agents. New list: {state['active_agents']}")
    
    logger.info(f"State after update: {json.dumps(state, default=str)}")
    return state

def visualize_graph(graph: StateGraph, filename: str = "hotel_services_graph") -> None:
    """Generate and display a visualization of the graph."""
    try:
        # Create a new directed graph
        dot = graphviz.Digraph(comment='Hotel Services Graph')
        dot.attr(rankdir='LR')  # Left to right layout
        
        # Add nodes
        for node in graph.nodes:
            # Style nodes based on their type
            if node == "classify_intents":
                dot.node(node, node, shape='diamond', style='filled', fillcolor='lightblue')
            elif node == "merge_responses":
                dot.node(node, node, shape='diamond', style='filled', fillcolor='lightgreen')
            elif "agent" in node:
                dot.node(node, node, shape='box', style='filled', fillcolor='lightyellow')
            else:
                dot.node(node, node, shape='ellipse')
        
        # Add edges
        for node, edges in graph.edges.items():
            for target, condition in edges.items():
                if isinstance(condition, str):
                    # For conditional edges, add the condition as edge label
                    dot.edge(node, target, label=condition)
                else:
                    # For regular edges
                    dot.edge(node, target)
        
        # Save and display the graph
        dot.render(filename, format='png', cleanup=True)
        display(Image(filename + '.png'))
        logger.info(f"Graph visualization saved as {filename}.png")
        
    except Exception as e:
        logger.error(f"Error generating graph visualization: {str(e)}")
        raise

def create_guest_services_graph() -> StateGraph:
    """Create the LangGraph workflow for guest services with enhanced agents."""
    workflow = StateGraph(GuestState)
    
    # Add nodes with proper state handling
    workflow.add_node("classify_intents", llm_classify_intents)
    
    # Create a router function for each agent that properly handles state updates
    def create_agent_router(agent_name: str, agent_func):
        async def router(state: GuestState) -> Dict:
            logger.info(f"{agent_name} router processing state: {json.dumps(state, default=str)}")
            try:
                result = await agent_func(state)
                logger.info(f"{agent_name} router returning: {json.dumps(result, default=str)}")
                # Update state with agent response
                updated_state = await update_state_with_agent_response(state, result)
                return updated_state
            except Exception as e:
                logger.error(f"Error in {agent_name} router: {str(e)}", exc_info=True)
                raise
        return router
    
    # Add agent nodes with routers
    workflow.add_node("qa_agent", create_agent_router("qa_agent", qa_agent_node))
    workflow.add_node("ore_agent", create_agent_router("ore_agent", ore_agent_node))
    workflow.add_node("checkin_agent", create_agent_router("checkin_agent", checkin_agent_node))
    workflow.add_node("checkout_agent", create_agent_router("checkout_agent", checkout_agent_node))
    workflow.add_node("concierge_agent", create_agent_router("concierge_agent", concierge_agent_node))
    workflow.add_node("feedback_agent", create_agent_router("feedback_agent", feedback_agent_node))
    workflow.add_node("lost_found_agent", create_agent_router("lost_found_agent", lost_found_agent_node))
    
    # Add the merge node
    workflow.add_node("merge_responses", merge_responses)
    
    # Add sequential edges based on intent priority
    priority_order = {
        Intent.CHECKIN: 1,
        Intent.CHECKOUT: 2,
        Intent.ORE: 3,
        Intent.QA: 4,
        Intent.CONCIERGE: 5,
        Intent.FEEDBACK: 6,
        Intent.LOST_FOUND: 7
    }
    
    # Create a router function for sequential processing
    def route_to_next_agent(state: GuestState) -> str:
        if not state["intents"]:
            return "merge_responses"
            
        # Get the highest priority intent that hasn't been processed
        processed_intents = {intent for intent, _ in state["responses"].items()}
        remaining_intents = [intent for intent in state["intents"] if intent not in processed_intents]
        
        if not remaining_intents:
            return "merge_responses"
            
        # Find the highest priority remaining intent
        next_intent = min(remaining_intents, key=lambda x: priority_order.get(x, 999))
        
        # Map intent to agent
        intent_to_agent = {
            Intent.QA: "qa_agent",
            Intent.ORE: "ore_agent",
            Intent.CHECKIN: "checkin_agent",
            Intent.CHECKOUT: "checkout_agent",
            Intent.CONCIERGE: "concierge_agent",
            Intent.FEEDBACK: "feedback_agent",
            Intent.LOST_FOUND: "lost_found_agent"
        }
        
        return intent_to_agent[next_intent]
    
    # Add conditional edges for sequential processing
    workflow.add_conditional_edges(
        "classify_intents",
        route_to_next_agent,
        {
            "qa_agent": "qa_agent",
            "ore_agent": "ore_agent",
            "checkin_agent": "checkin_agent",
            "checkout_agent": "checkout_agent",
            "concierge_agent": "concierge_agent",
            "feedback_agent": "feedback_agent",
            "lost_found_agent": "lost_found_agent",
            "merge_responses": "merge_responses"
        }
    )
    
    # Add edges from each agent back to the router
    for agent in ["qa_agent", "ore_agent", "checkin_agent", "checkout_agent", 
                 "concierge_agent", "feedback_agent", "lost_found_agent"]:
        workflow.add_conditional_edges(
            agent,
            route_to_next_agent,
            {
                "qa_agent": "qa_agent",
                "ore_agent": "ore_agent",
                "checkin_agent": "checkin_agent",
                "checkout_agent": "checkout_agent",
                "concierge_agent": "concierge_agent",
                "feedback_agent": "feedback_agent",
                "lost_found_agent": "lost_found_agent",
                "merge_responses": "merge_responses"
            }
        )
    
    # Set entry and exit points
    workflow.set_entry_point("classify_intents")
    workflow.set_finish_point("merge_responses")
    
    # Add visualization after graph is created
    try:
        visualize_graph(workflow)
    except Exception as e:
        logger.warning(f"Could not generate graph visualization: {str(e)}")
    
    return workflow

# --- Main Execution ---
async def process_guest_query(
    question: str,
    metadata: Optional[Dict] = None,
    hotel_service: Optional[MockHotelService] = None,
    visualize: bool = False  # Add visualization parameter
) -> Dict:
    """
    Process a guest query through the enhanced agent workflow.
    Args:
        question: The guest's query
        metadata: Optional metadata about the guest
        hotel_service: Optional hotel service instance
        visualize: Whether to generate and display the graph visualization
    """
    service = hotel_service or MockHotelService()
    try:
        logger.info("="*50)
        logger.info("Starting new query processing session")
        logger.info(f"Input question: {question}")
        logger.info(f"Input metadata: {json.dumps(metadata, default=str)}")
        
        # Create graph
        logger.info("Creating guest services graph...")
        graph = create_guest_services_graph()
        logger.info("Graph created successfully")
        
        # Visualize if requested
        if visualize:
            try:
                visualize_graph(graph, filename=f"hotel_services_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            except Exception as e:
                logger.warning(f"Could not generate graph visualization: {str(e)}")
        
        logger.info("Compiling graph...")
        compiled_graph = graph.compile()
        logger.info("Graph compiled successfully")
        
        # Initialize state
        logger.info("Creating initial state...")
        initial_state = create_initial_state(question, metadata)
        logger.info(f"Initial state created: {json.dumps(initial_state, default=str)}")
        
        # Execute workflow
        logger.info("Starting graph execution...")
        try:
            logger.info("Invoking graph with initial state...")
            final_state = await compiled_graph.ainvoke(initial_state)
            logger.info("Graph execution completed successfully")
            logger.info(f"Final state after graph execution: {json.dumps(final_state, default=str)}")
        except Exception as graph_error:
            logger.error(f"Error during graph execution: {str(graph_error)}", exc_info=True)
            logger.error(f"State at time of error: {json.dumps(final_state if 'final_state' in locals() else {}, default=str)}")
            raise
        
        # Validate final state
        logger.info("Validating final state...")
        if "final_response" not in final_state:
            logger.error(f"Missing final_response in state. Available keys: {list(final_state.keys())}")
            logger.error(f"Full state content: {json.dumps(final_state, default=str)}")
            raise KeyError("final_response not found in final state")
        
        # Log completion
        logger.info("Query processing completed successfully")
        logger.info(f"Active agents: {final_state['active_agents']}")
        logger.info(f"Detected intents: {[i.value for i in final_state['intents']]}")
        logger.info(f"Response count: {len(final_state.get('responses', {}))}")
        logger.info(f"Processing timestamp: {final_state['timestamp']}")
        
        result = {
            "response": final_state["final_response"],
            "metadata": {
                "intents": [intent.value for intent in final_state["intents"]],
                "intent_confidence": final_state.get("intent_confidence", {}),
                "primary_intent": final_state.get("primary_intent", "").value,
                "active_agents": final_state["active_agents"],
                "timestamp": final_state["timestamp"],
                **final_state["metadata"]
            }
        }
        logger.info(f"Returning result: {json.dumps(result, default=str)}")
        logger.info("="*50)
        return result
        
    except Exception as e:
        logger.error("="*50)
        logger.error("Error in process_guest_query")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Error details:", exc_info=True)
        logger.error(f"Last known state: {json.dumps(final_state if 'final_state' in locals() else {}, default=str)}")
        logger.error("="*50)
        
        return {
            "response": "I apologize, but I'm having trouble processing your request. Please try again or contact the front desk.",
            "metadata": {
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.utcnow().isoformat(),
                "error_details": {
                    "traceback": str(e.__traceback__),
                    "state_at_error": json.dumps(final_state if 'final_state' in locals() else {}, default=str)
                }
            }
        }

# --- Example Usage ---
if __name__ == "__main__":
    async def main():
        # Initialize mock service
        mock_service = MockHotelService()
        
        # Example guest queries
        queries = [
            "What time is check-in and do you have any special offers for my stay?",
            "I lost my phone in the restaurant, can you help?",
            "The room service was excellent, I want to leave a review",
            "Can the concierge help me book a tour for tomorrow?"
        ]
        
        # Test with different guest profiles
        test_metadata = [
            {
                "guest_id": "G12345",
                "room_type": "Deluxe",
                "check_in_date": "2024-03-20"
            },
            {
                "guest_id": "G12346",
                "room_type": "Suite",
                "check_in_date": "2024-03-21"
            }
        ]
        
        # Process each query with different guest profiles
        for query in queries:
            for metadata in test_metadata:
                print("\n" + "="*50)
                print("Guest Query:", query)
                print("Guest Profile:", metadata["guest_id"])
                # Add visualize=True for the first query only
                result = await process_guest_query(
                    query, 
                    metadata, 
                    mock_service,
                    visualize=(query == queries[0] and metadata == test_metadata[0])
                )
                print("\nFinal Response:", result["response"])
                print("\nMetadata:", json.dumps(result["metadata"], indent=2))
                print("="*50)

    # Run examples
    asyncio.run(main())

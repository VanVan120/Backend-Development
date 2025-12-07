import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

SYSTEM_INSTRUCTION = """
You are a helpful, empathetic, and professional medical assistant for an Oral Cancer AI Platform.
Your goal is to help users understand their analysis results.

STRICT GUIDELINES:
1. **No Diagnosis**: You are an AI, not a doctor. Never say "You have cancer" or "You are healthy". Instead say "The analysis suggests..." or "The model detected...".
2. **Always Advise Consultation**: End every significant medical answer with a recommendation to see a specialist (Dentist or Pathologist).
3. **Context Aware**: Use the provided 'Analysis Context' to answer questions. If the context says 'Tumour Detected', do not say everything looks fine.
4. **Tone**: Calm, informative, and supportive.
"""

def get_chat_response(user_message: str, analysis_context: dict) -> str:
    """
    Generates a response using Google Gemini (or OpenAI fallback).
    """
    if not GEMINI_API_KEY:
        return "Error: AI Chat service is not configured (Missing API Key)."

    try:
        # Construct the prompt with context
        context_str = json.dumps(analysis_context, indent=2)
        
        full_prompt = f"""
        [System]
        {SYSTEM_INSTRUCTION}

        [Analysis Context]
        {context_str}

        [User Question]
        {user_message}
        """

        # Use gemini-2.5-flash for faster response times
        model = genai.GenerativeModel('gemma-3-27b-it')
        response = model.generate_content(full_prompt)
        
        return response.text

    except Exception as e:
        print(f"Chat Error Details: {e}")
        # Return the actual error for debugging purposes (in development)
        return f"I encountered an error connecting to the AI service: {str(e)}"

import sys
import os
import shutil
import uuid
import importlib.util
import base64
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Import new services
from report_service import generate_expert_report, generate_public_report
from chat_service import get_chat_response

# --- Security & Configuration Check ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "replace_with_your_key_here":
    print("\n" + "="*60)
    print("CRITICAL ERROR: Missing or Invalid API Key")
    print("="*60)
    print("The application requires a valid GEMINI_API_KEY to function.")
    print("Please follow these steps:")
    print("1. Create a file named '.env' in the project root.")
    print("2. Copy the contents of '.env.example' into '.env'.")
    print("3. Replace 'replace_with_your_key_here' with your actual Gemini API key.")
    print("="*60 + "\n")
    sys.exit(1)

# --- Setup Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_A_DIR = os.path.join(BASE_DIR, "Model A")
MODEL_B_DIR = os.path.join(BASE_DIR, "Model B")
MODEL_TRIAGE_DIR = os.path.join(BASE_DIR, "Model Triage")

# Add directories to sys.path to allow internal imports (like stain_utils in Model A)
sys.path.append(MODEL_A_DIR)
sys.path.append(MODEL_B_DIR)
sys.path.append(MODEL_TRIAGE_DIR)

# --- Dynamic Imports ---
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load Triage
print("Loading Triage Model...")
triage_module = load_module("triage_inference_pkg", os.path.join(MODEL_TRIAGE_DIR, "triage_inference.py"))
TriageRouter = triage_module.TriageRouter

# Load Model A
print("Loading Model A...")
model_a_module = load_module("model_a_inference_pkg", os.path.join(MODEL_A_DIR, "inference_model.py"))
ModelAInference = model_a_module.ModelAInference

# Load Model B
print("Loading Model B...")
model_b_module = load_module("model_b_inference_pkg", os.path.join(MODEL_B_DIR, "inference_model.py"))
OralHygieneModel = model_b_module.OralHygieneModel

# --- Initialize App ---
app = FastAPI(title="Unified Oral Disease Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Templates
templates = Jinja2Templates(directory="static/html")

# --- Initialize Models ---
# Triage
triage_model_path = os.path.join(MODEL_TRIAGE_DIR, "triage_router.pth")
triage_router = TriageRouter(model_path=triage_model_path)

# Model A
model_a_path = os.path.join(MODEL_A_DIR, "model_a.pth")
# Check if model exists, else handle gracefully
if os.path.exists(model_a_path):
    model_a = ModelAInference(model_path=model_a_path)
else:
    print(f"Warning: {model_a_path} not found.")
    model_a = None

# Model B
model_b_path = os.path.join(MODEL_B_DIR, "models", "best.pt")
if os.path.exists(model_b_path):
    model_b = OralHygieneModel(model_path=model_b_path)
else:
    print(f"Warning: {model_b_path} not found.")
    model_b = None

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about")
def read_about(request: Request):
    return templates.TemplateResponse("views/about.html", {"request": request})

@app.get("/department")
def read_department(request: Request):
    return templates.TemplateResponse("views/department.html", {"request": request})

@app.get("/pages")
def read_pages(request: Request):
    return templates.TemplateResponse("views/pages.html", {"request": request})

@app.get("/blog")
def read_blog(request: Request):
    return templates.TemplateResponse("views/blog.html", {"request": request})

@app.get("/contact")
def read_contact(request: Request):
    return templates.TemplateResponse("views/contact.html", {"request": request})

# --- Legal Pages ---
@app.get("/privacy-policy")
def privacy_policy(request: Request):
    return templates.TemplateResponse("views/legal/privacy_policy.html", {"request": request})

@app.get("/terms-of-service")
def terms_of_service(request: Request):
    return templates.TemplateResponse("views/legal/terms_of_service.html", {"request": request})

@app.get("/cookie-policy")
def cookie_policy(request: Request):
    return templates.TemplateResponse("views/legal/cookie_policy.html", {"request": request})

@app.get("/hipaa-compliance")
def hipaa_compliance(request: Request):
    return templates.TemplateResponse("views/legal/hipaa_compliance.html", {"request": request})

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    print(f"--- Starting Analysis for {file.filename} ---")
    # 1. Save File
    file_ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_ext}"
    filepath = os.path.join(TEMP_DIR, filename)
    
    print(f"Saving file to {filepath}...")
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Run Triage
        print("Running Triage Model...")
        triage_result = triage_router.predict(filepath)
        print(f"Triage Result: {triage_result}")
        # triage_result is a string: 'Clinical' or 'Histopathological'
        
        detected_type = triage_result
        
        response_data = {
            "triage_result": triage_result,
            "final_analysis": None,
            "model_used": None
        }

        # 3. Route to specific model
        if detected_type == "Histopathological":
            print("Routing to Model A (Histopathology)...")
            if model_a:
                # Model A predict method takes image_path
                print("Running Model A Inference...")
                analysis = model_a.predict(filepath)
                print("Model A Inference Complete.")
                response_data["final_analysis"] = analysis
                response_data["model_used"] = "Model A (Histopathology)"
                
                # Generate AI Suggestion
                try:
                    suggestion_prompt = "Based on these histopathology results, provide a brief, professional suggestion for the medical professional reviewing this case (max 2 sentences)."
                    ai_suggestion = get_chat_response(suggestion_prompt, analysis)
                    response_data["ai_suggestion"] = ai_suggestion
                except Exception as e:
                    print(f"AI Suggestion Error: {e}")
                    response_data["ai_suggestion"] = "AI suggestion unavailable at this time."

            else:
                print("Error: Model A is not loaded.")
                response_data["error"] = "Model A is not loaded."

        elif detected_type == "Clinical":
            print("Routing to Model B (Clinical)...")
            if model_b:
                # Model B predict method takes image_path
                print("Running Model B Inference...")
                analysis = model_b.predict(filepath)
                print("Model B Inference Complete.")
                response_data["final_analysis"] = analysis
                response_data["model_used"] = "Model B (Clinical Screening)"

                # Generate AI Suggestion
                try:
                    suggestion_prompt = "Based on these clinical screening results, provide a brief, empathetic suggestion for the patient (max 2 sentences). Focus on hygiene or next steps."
                    ai_suggestion = get_chat_response(suggestion_prompt, analysis)
                    response_data["ai_suggestion"] = ai_suggestion
                except Exception as e:
                    print(f"AI Suggestion Error: {e}")
                    response_data["ai_suggestion"] = "AI suggestion unavailable at this time."

            else:
                print("Error: Model B is not loaded.")
                response_data["error"] = "Model B is not loaded."
        
        else:
            print(f"Error: Unknown image type detected: {detected_type}")
            response_data["error"] = "Unknown image type detected."

        print("Analysis Finished. Sending response.")
        return JSONResponse(content=response_data)

    except Exception as e:
        print("EXCEPTION DURING ANALYSIS:")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if os.path.exists(filepath):
            print(f"Cleaning up temp file: {filepath}")
            os.remove(filepath)

# --- New Endpoints ---

class ReportRequest(BaseModel):
    report_type: str  # "expert" or "public"
    analysis_data: Dict[str, Any]
    image_base64: Optional[str] = None

class SendReportRequest(ReportRequest):
    email: str

def send_report_email_task(email: str, sender_email: str, api_key: str, pdf_bytes: bytes, filename: str, image_bytes: Optional[bytes] = None):
    try:
        url = "https://api.brevo.com/v3/smtp/email"
        headers = {
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json"
        }

        # Encode PDF to base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        attachments = [
            {
                "content": pdf_base64,
                "name": filename
            }
        ]

        # If image exists, attach it too
        if image_bytes:
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')
            attachments.append({
                "content": img_base64,
                "name": "analysis_image.jpg"
            })

        payload = {
            "sender": {"name": "Oral AI Team", "email": sender_email},
            "to": [{"email": email}],
            "subject": "Your Oral AI Analysis Report",
            "htmlContent": """
            <html>
                <body>
                    <h2>Your Analysis Report is Ready</h2>
                    <p>Dear User,</p>
                    <p>Please find attached the medical analysis report generated by the Oral AI Platform.</p>
                    <p>We have also attached the analyzed image for your reference.</p>
                    <p>If you have any questions, please consult with a medical professional.</p>
                    <br>
                    <p>Best regards,</p>
                    <p>The Oral AI Team</p>
                </body>
            </html>
            """,
            "attachment": attachments
        }

        response = requests.post(url, json=payload, headers=headers)
        if not response.ok:
            print(f"Failed to send report email: {response.text}")
        else:
            print(f"Report email sent successfully to {email}")

    except Exception as e:
        print(f"Background Report Email Error: {e}")
        import traceback
        traceback.print_exc()

@app.post("/api/send-report")
async def send_report_endpoint(request: SendReportRequest, background_tasks: BackgroundTasks):
    try:
        # 1. Generate PDF (Same logic as /generate-report)
        image_bytes = None
        if request.image_base64:
            if "," in request.image_base64:
                header, encoded = request.image_base64.split(",", 1)
            else:
                encoded = request.image_base64
            image_bytes = base64.b64decode(encoded)

        if request.report_type == "expert":
            pdf_bytes = generate_expert_report(request.analysis_data, image_bytes)
            filename = "Expert_Analysis_Report.pdf"
        elif request.report_type == "public":
            pdf_bytes = generate_public_report(request.analysis_data, image_bytes)
            filename = "Oral_Health_Screening.pdf"
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")

        # 2. Get Email Config
        sender_email = os.getenv("EMAIL_SENDER")
        api_key = os.getenv("BREVO_API_KEY")
        
        if sender_email: sender_email = sender_email.strip()
        if api_key: api_key = api_key.strip()

        if not sender_email or not api_key:
            print("Simulation: Email configuration missing. Skipping email.")
            return {"message": "Email configuration missing (Simulation Mode)"}

        # 3. Send Email in Background
        background_tasks.add_task(send_report_email_task, request.email, sender_email, api_key, pdf_bytes, filename, image_bytes)

        return {"message": f"Report generated and sent to {request.email}"}

    except Exception as e:
        print(f"Send Report Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-report")
async def generate_report_endpoint(request: ReportRequest):
    try:
        image_bytes = None
        if request.image_base64:
            # Remove header if present (e.g., "data:image/jpeg;base64,")
            if "," in request.image_base64:
                header, encoded = request.image_base64.split(",", 1)
            else:
                encoded = request.image_base64
            image_bytes = base64.b64decode(encoded)

        if request.report_type == "expert":
            pdf_bytes = generate_expert_report(request.analysis_data, image_bytes)
            filename = "Expert_Analysis_Report.pdf"
        elif request.report_type == "public":
            pdf_bytes = generate_public_report(request.analysis_data, image_bytes)
            filename = "Oral_Health_Screening.pdf"
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"Report Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Ensure context is a dict if None
        ctx = request.context if request.context is not None else {}
        response_text = get_chat_response(request.message, ctx)
        return {"reply": response_text}
    except Exception as e:
        print(f"Chat Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

import requests

# --- Newsletter Subscription ---
class SubscribeRequest(BaseModel):
    email: str

def send_email_task(email: str, base_url: str, sender_email: str, recipient_email: str, api_key: str):
    try:
        url = "https://api.brevo.com/v3/smtp/email"
        headers = {
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json"
        }

        # 1. Send Notification to Admin (You)
        payload_admin = {
            "sender": {"name": "Oral AI Bot", "email": sender_email},
            "to": [{"email": recipient_email}],
            "replyTo": {"email": email},
            "subject": f"New Subscriber: {email}",
            "textContent": f"""
            New Newsletter Subscription
            ---------------------------
            
            A new user has joined your newsletter!
            
            Subscriber Email: {email}
            
            (This email was sent automatically by your Oral AI Platform)
            """
        }
        response_admin = requests.post(url, json=payload_admin, headers=headers)
        if not response_admin.ok:
            print(f"Failed to send admin email: {response_admin.text}")

        # 2. Send Confirmation to Subscriber (The User)
        payload_user = {
            "sender": {"name": "Oral AI Team", "email": sender_email},
            "to": [{"email": email}],
            "subject": "Welcome to Oral AI!",
            "htmlContent": f"""
            <html>
                <body>
                    <h2>Welcome to Oral AI!</h2>
                    <p>Hi there,</p>
                    <p>Thank you for subscribing to the Oral AI newsletter!</p>
                    <p>We are thrilled to have you with us. You will now receive the latest updates on AI medical breakthroughs and oral health technology.</p>
                    <p><strong><a href="{base_url}/pages">Click here to visit our News Page</a></strong></p>
                    <br>
                    <p>Best regards,</p>
                    <p>The Oral AI Team</p>
                </body>
            </html>
            """
        }
        response_user = requests.post(url, json=payload_user, headers=headers)
        if not response_user.ok:
            print(f"Failed to send user email: {response_user.text}")
        else:
            print(f"Email sent successfully to {email}")

    except Exception as e:
        print(f"Background Email Error: {e}")
        import traceback
        traceback.print_exc()

@app.post("/api/subscribe")
async def subscribe_newsletter(request: SubscribeRequest, background_tasks: BackgroundTasks):
    sender_email = os.getenv("EMAIL_SENDER")
    recipient_email = os.getenv("EMAIL_RECIPIENT")
    api_key = os.getenv("BREVO_API_KEY")
    
    # Get Base URL from env or default to the Hugging Face Space URL
    base_url = os.getenv("BASE_URL", "https://ivanjun-oral-ai-cancer-disease-detection.hf.space")

    # Clean up credentials
    if sender_email: sender_email = sender_email.strip()
    if recipient_email: recipient_email = recipient_email.strip()
    if api_key: api_key = api_key.strip()

    if not sender_email or not api_key or not recipient_email:
        # Fallback for demo purposes if not configured
        print(f"Simulation: Newsletter subscription for {request.email}")
        return {"message": "Subscribed successfully (Simulation Mode - Configure .env for real emails)"}

    # Add email task to background
    background_tasks.add_task(send_email_task, request.email, base_url, sender_email, recipient_email, api_key)

    return {"message": "Subscribed successfully"}

if __name__ == "__main__":
    import uvicorn
    # Using 127.0.0.1 instead of 0.0.0.0 to prevent Windows browser errors
    uvicorn.run(app, host="127.0.0.1", port=8000)

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
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
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

# --- Newsletter Subscription ---
class SubscribeRequest(BaseModel):
    email: str

@app.post("/api/subscribe")
async def subscribe_newsletter(request: SubscribeRequest):
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    recipient_email = os.getenv("EMAIL_RECIPIENT")
    # Get Base URL from env or default to the Hugging Face Space URL
    base_url = os.getenv("BASE_URL", "https://ivanjun-oral-ai-cancer-disease-detection.hf.space")

    # Clean up credentials (remove potential spaces or newlines)
    if sender_email: sender_email = sender_email.strip()
    if sender_password: sender_password = sender_password.strip().replace(" ", "") # Remove spaces from app password
    if recipient_email: recipient_email = recipient_email.strip()

    if not sender_email or not sender_password or not recipient_email:
        # Fallback for demo purposes if not configured
        print(f"Simulation: Newsletter subscription for {request.email}")
        # In a real scenario, you might want to return an error or log this.
        # For now, we return success to the frontend so the UI updates.
        return {"message": "Subscribed successfully (Simulation Mode - Configure .env for real emails)"}

    try:
        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)

        # 1. Send Notification to Admin (You)
        msg_admin = MIMEMultipart()
        msg_admin['From'] = f"Oral AI Bot <{sender_email}>"
        msg_admin['To'] = recipient_email
        msg_admin['Reply-To'] = request.email
        msg_admin['Subject'] = f"New Subscriber: {request.email}"

        body_admin = f"""
        New Newsletter Subscription
        ---------------------------
        
        A new user has joined your newsletter!
        
        Subscriber Email: {request.email}
        
        (This email was sent automatically by your Oral AI Platform)
        """
        msg_admin.attach(MIMEText(body_admin, 'plain'))
        server.sendmail(sender_email, recipient_email, msg_admin.as_string())

        # 2. Send Confirmation to Subscriber (The User)
        msg_user = MIMEMultipart()
        msg_user['From'] = f"Oral AI Team <{sender_email}>"
        msg_user['To'] = request.email
        msg_user['Subject'] = "Welcome to Oral AI!"

        body_user = f"""
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
        msg_user.attach(MIMEText(body_user, 'html'))
        server.sendmail(sender_email, request.email, msg_user.as_string())

        # Close connection
        server.quit()

        return {"message": "Subscribed successfully"}
    except Exception as e:
        print(f"Email Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Using 127.0.0.1 instead of 0.0.0.0 to prevent Windows browser errors
    uvicorn.run(app, host="127.0.0.1", port=8000)

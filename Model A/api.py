from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
import uuid
from inference_model import ModelAInference

# --- Configuration ---
app = FastAPI(title="Model A: OSCC Analysis API", version="1.0")

# CORS Setup (Allow Frontend Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files - Frontend developer works here
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global Model Instance
model_inference = None
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize the model on server startup."""
    global model_inference
    # Ensure model_a.pth exists or handle it gracefully
    model_inference = ModelAInference(model_path="model_a.pth")
    print("✅ Model A API is ready.")

@app.get("/")
def read_root():
    # Serves the frontend entry point
    return FileResponse('static/index.html')

@app.post("/predict_wsi")
async def predict_wsi(file: UploadFile = File(...)):
    """
    Endpoint to analyze a WSI patch.
    Accepts an image file, saves it temporarily, runs inference, and cleans up.
    """
    if not model_inference:
        raise HTTPException(status_code=503, detail="Model not initialized.")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename to prevent collisions
    file_ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    temp_filename = f"{uuid.uuid4()}.{file_ext}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Run Inference
        result = model_inference.predict(temp_path)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

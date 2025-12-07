from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
import os
import uuid
from inference_model import OralHygieneModel

app = FastAPI(title="Oral Cancer Screening Model B API")

# Enable CORS for frontend collaboration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Model
# NOTE: This path must point to your trained weights. 
MODEL_PATH = "models/best.pt"
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model not found at {MODEL_PATH}. API will return errors for predictions until trained.")
    model = None
else:
    model = OralHygieneModel(MODEL_PATH)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
def root():
    return FileResponse('static/index.html')

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image and get screening results.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save temp file
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"{uuid.uuid4()}.{file_ext}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run inference
        result = model.predict(temp_path)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
             
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting API server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

from flask import Flask, request, jsonify, send_from_directory
import os
from triage_inference import TriageRouter
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='', static_folder='static')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Model
print("Loading Triage Model...")
router = TriageRouter()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Perform inference
            result = router.predict(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if result == "Unknown":
                return jsonify({
                    'error': 'Image content not recognized. Please upload a valid Clinical or Histopathological oral image.'
                }), 400

            return jsonify({
                'class': result,
                'message': 'Classification successful'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory

from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy import ndimage
import logging
from typing import List, Tuple, Dict
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class TextDetectionCNN(nn.Module):
    def __init__(self):
        super(TextDetectionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return torch.sigmoid(x)

class DrawingProcessor:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.logger = self._setup_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.preprocess_params = {
            'gaussian_kernel': (5, 5),
            'gaussian_sigma': 1,
            'adaptive_block_size': 11,
            'adaptive_C': 2,
            'canny_lower': 50,
            'canny_upper': 150,
            'morph_kernel_size': 3,
            'target_size': (256, 256)
        }

        self.text_model = TextDetectionCNN().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        self.initialize_ocr()

    def _setup_logger(self):
        logger = logging.getLogger('DrawingProcessor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        return logger

    def initialize_ocr(self):
        try:
            self.ocr_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-'
        except Exception as e:
            self.logger.error(f"Error initializing OCR: {str(e)}")
            raise

    def preprocess_drawing(self, image):
        try:
            self.logger.info("Starting preprocessing pipeline")
            results = {'original': image.copy()}

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            results['grayscale'] = gray

            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            denoised = cv2.GaussianBlur(gray,
                                       self.preprocess_params['gaussian_kernel'],
                                       self.preprocess_params['gaussian_sigma'])
            denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
            results['denoised'] = denoised

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            results['enhanced'] = enhanced

            binary = cv2.adaptiveThreshold(enhanced, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV,
                                         self.preprocess_params['adaptive_block_size'],
                                         self.preprocess_params['adaptive_C'])
            results['binary'] = binary

            edges = cv2.Canny(enhanced,
                            self.preprocess_params['canny_lower'],
                            self.preprocess_params['canny_upper'])
            results['edges'] = edges

            kernel = np.ones((self.preprocess_params['morph_kernel_size'],
                            self.preprocess_params['morph_kernel_size']),
                           np.uint8)
            cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            results['cleaned'] = cleaned

            self.logger.info("Preprocessing completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def detect_text_regions(self, preprocessed_results: dict) -> List[Tuple[int, int, int, int]]:
        try:
            self.text_model.eval()
            with torch.no_grad():
                enhanced_img = preprocessed_results['enhanced']
                resized_img = cv2.resize(enhanced_img,
                                       self.preprocess_params['target_size'])

                tensor = self.transform(resized_img).unsqueeze(0)
                tensor = tensor.to(self.device)

                predictions = self.text_model(tensor)

            text_regions = self._process_predictions(predictions, enhanced_img.shape)
            self.logger.info(f"Detected Text Regions: {text_regions}")
            return text_regions

        except Exception as e:
            self.logger.error(f"Error in text detection: {str(e)}")
            return []

    def _process_predictions(self, predictions: torch.Tensor,
                           image_shape: Tuple) -> List[Tuple[int, int, int, int]]:
        try:
            predictions = predictions.cpu().detach().numpy()

            binary = np.zeros(self.preprocess_params['target_size'][::-1],
                            dtype=np.uint8)
            binary[predictions[0, 1] > 0.5] = 255

            binary = cv2.resize(binary, (image_shape[1], image_shape[0]))

            kernel = np.ones((5,5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

            regions = []
            min_area = 100

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append((x, y, w, h))

            return regions

        except Exception as e:
            self.logger.error(f"Error processing predictions: {str(e)}")
            return []

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        try:
            scale = 2
            image = cv2.resize(image, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = cv2.fastNlMeansDenoising(image)
            _, image = cv2.threshold(image, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return image

        except Exception as e:
            self.logger.error(f"Error in OCR preprocessing: {str(e)}")
            return image

    def recognize_text(self, image: np.ndarray,
                      regions: List[Tuple[int, int, int, int]]) -> Dict[str, str]:
        try:
            results = {}
            for i, (x, y, w, h) in enumerate(regions):
                roi = image[y:y+h, x:x+w]
                roi = self._preprocess_for_ocr(roi)

                text = pytesseract.image_to_string(roi, config=self.ocr_config)
                text = self._clean_text(text)

                if text:
                    results[f'region_{i}'] = {
                        'text': text,
                        'coordinates': (x, y, w, h)
                    }

            return results

        except Exception as e:
            self.logger.error(f"Error in text recognition: {str(e)}")
            return {}

    def _clean_text(self, text: str) -> str:
        try:
            text = text.strip()
            text = text.replace('\n', ' ')
            return text
        except Exception as e:
            self.logger.error(f"Error in text cleaning: {str(e)}")
            return text
        

def plot_results(results: dict, text_regions: List[Tuple[int, int, int, int]] = None):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Engineering Drawing Processing Results', fontsize=16)

    stages = ['original', 'grayscale', 'denoised',
              'enhanced', 'binary', 'cleaned']

    for idx, stage in enumerate(stages):
        row = idx // 3
        col = idx % 3
        img = results[stage].copy()

        if text_regions and stage == 'original':
            for x, y, w, h in text_regions:
                # Calculate circle parameters
                center_x = x + w//2
                center_y = y + h//2
                radius = int(max(w, h) / 1.5)

                # Draw red circle
                cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), 2)
                # Draw center point
                cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), -1)

        axes[row, col].imshow(img, cmap='gray' if stage != 'original' else None)
        axes[row, col].set_title(f'Stage: {stage.capitalize()}', fontsize=12)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

def plot_text_regions_with_circles(image: np.ndarray, text_regions: List[Tuple[int, int, int, int]]):
    plt.figure(figsize=(15, 10))

    img = image.copy()

    for x, y, w, h in text_regions:
        # Calculate circle parameters
        center_x = x + w//2
        center_y = y + h//2
        radius = int(max(w, h) / 1.5)

        # Draw red circle
        cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), 2)
        # Draw center point
        cv2.circle(img, (center_x, center_y), 2, (0, 0, 255), -1)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Text Regions', fontsize=14)
    plt.axis('off')
    plt.show()



app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the 'uploads' folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to validate file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_crop_objects(model_path, source_path, output_dir, conf=0.5):
    """
    Detect objects using YOLO and save each detected object as a separate image.

    Args:
        model_path: Path to the YOLO model weights
        source_path: Path to the source images
        output_dir: Directory to save cropped images
        conf: Confidence threshold for detections
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the YOLO model
    model = YOLO(model_path)

    # Run inference
    results = model(source_path, conf=conf)

    # Process each image
    for i, result in enumerate(results):
        # Get the original image
        orig_img = result.orig_img

        # Process each detection
        for j, box in enumerate(result.boxes):
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].int().cpu().numpy()

            # Get class name
            class_id = int(box.cls)
            class_name = result.names[class_id]

            # Get confidence
            conf = float(box.conf)

            # Crop the detection
            cropped = orig_img[y1:y2, x1:x2]

            # Convert to PIL Image
            cropped_img = Image.fromarray(cropped)

            # Save the cropped image
            save_path = os.path.join(output_dir, f'detection_{i}_{j}_{class_name}_{conf:.2f}.jpg')
            cropped_img.save(save_path)

            print(f"Saved detection: {save_path}")

# Model path
model_path = './best.pt'
source_path = './Train'
output_dir = './uploads'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for("results", image_name=filename))  # Redirect to results page
    return render_template("index.html")

'''
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for("results", image_name=filename))  # Redirect to results page
    return render_template("index.html")

@app.route("/results/<image_name>")
def results(image_name):

    processor = DrawingProcessor(debug_mode=True)

        # Process image
    preprocessed_results = processor.preprocess_drawing(image)

        # Detect text regions
    text_regions = processor.detect_text_regions(preprocessed_results)

        # Recognize text
    text_results = processor.recognize_text(image, text_regions)

        # Display all processing stages with circles
    plot_results(preprocessed_results, text_regions)

        # Display original image with circles around text
    plot_text_regions_with_circles(image, text_regions)

        # Print recognized text
    print("\nRecognized Text:")
    for region_id, data in text_results.items():
        print(f"Region {region_id}:")
        print(f"Text: {data['text']}")
        print(f"Location: {data['coordinates']}")
        print("---")

    print("\nProcessing Parameters:")
    for param, value in processor.preprocess_params.items():
        print(f"{param}: {value}")
    
    detect_and_crop_objects(model_path, source_path, output_dir, conf=0.5)
    return render_template("results.html", image_name=image_name,)
'''

@app.route("/results/<image_name>")
def results(image_name):
    # Load and process the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image = cv2.imread(image_path)

    processor = DrawingProcessor(debug_mode=True)

    # Process image
    preprocessed_results = processor.preprocess_drawing(image)

    # Detect text regions
    text_regions = processor.detect_text_regions(preprocessed_results)

    # Recognize text
    text_results = processor.recognize_text(image, text_regions)

    # Display all processing stages with circles
    #plot_results(preprocessed_results, text_regions)

    # Display original image with circles around text
    #plot_text_regions_with_circles(image, text_regions)

    # Print recognized text
    print("\nRecognized Text:")
    for region_id, data in text_results.items():
        print(f"Region {region_id}:")
        print(f"Text: {data['text']}")
        print(f"Location: {data['coordinates']}")
        print("---")

    print("\nProcessing Parameters:")
    for param, value in processor.preprocess_params.items():
        print(f"{param}: {value}")
    
    # Run object detection (YOLO)
    detect_and_crop_objects(model_path, image_path, app.config['UPLOAD_FOLDER'], conf=0.5)

    # Get the cropped detection images
    detection_images = os.listdir(app.config['UPLOAD_FOLDER'])
    detection_images = [img for img in detection_images if img.endswith(".jpg")]


    return render_template("results.html", image_name=image_name, text_results=text_results, detection_images=detection_images)

'''
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'../uploads/{filename}'))'''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
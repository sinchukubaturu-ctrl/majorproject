import os
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from dcm_processor import save_user_details, create_dicom_from_image
from database import get_connection, get_all_patients
import psycopg2

# Initialize Flask
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Folder configurations
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["PRED_FOLDER"] = os.path.join("static", "predicted")
app.config["DICOM_FOLDER"] = os.path.join("static", "dicoms")
app.config["REPORT_FOLDER"] = os.path.join("reports")

# Create required directories if not exist
for folder in [app.config["UPLOAD_FOLDER"], app.config["PRED_FOLDER"], app.config["DICOM_FOLDER"], app.config["REPORT_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)

# Load YOLO model
MODEL_PATH = "bestfinal.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Tumor type mappings
CLASS_MAPPING = {
    "class_0": "Glioma",
    "class_1": "Meningioma",
    "class_2": "Pituitary Tumor",
    "class_3": "No Tumor"
}

# ---------------- HELPER FUNCTIONS ----------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_user_details(user_data):
    """Insert or update patient record in PostgreSQL."""
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO patient_records (
                patient_id, name, age, gender, date_of_scan,
                symptoms, family_history, head_injury_notes,
                other_conditions, image_filename, pred_label,
                confidence, dicom_filename, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (patient_id)
            DO UPDATE SET
                pred_label = EXCLUDED.pred_label,
                confidence = EXCLUDED.confidence,
                dicom_filename = EXCLUDED.dicom_filename,
                timestamp = CURRENT_TIMESTAMP;
        """, (
            user_data.get("patient_id"),
            user_data.get("name"),
            user_data.get("age"),
            user_data.get("gender"),
            user_data.get("date_of_scan"),
            user_data.get("symptoms"),
            user_data.get("family_history"),
            user_data.get("head_injury_notes"),
            user_data.get("other_conditions"),
            user_data.get("image_filename"),
            user_data.get("pred_label"),
            user_data.get("confidence"),
            user_data.get("dicom_filename")
        ))

        conn.commit()
        cur.close()
        conn.close()
        print("[SUCCESS] Patient record saved/updated in PostgreSQL.")
    except Exception as e:
        print(f"[ERROR] Database save failed: {e}")


def annotate_image(image_path, results, save_path):
    """Annotate detection boxes on image."""
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 24) if os.path.exists("arial.ttf") else ImageFont.load_default()

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].tolist()]
                pred_index = int(box.cls.item())
                raw_label = results[0].names[pred_index]
                label = CLASS_MAPPING.get(raw_label, raw_label)
                conf = float(box.conf.item())
                text = f"{label} ({conf:.2f})"

                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                text_bbox = draw.textbbox((int(x1), int(y1)), text, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((int(x1), int(y1)), text, fill="white", font=font)

        img.save(save_path)
    except Exception as e:
        print(f"[ERROR] Annotation failed: {e}")
        shutil.copy(image_path, save_path)

# ---------------- ROUTES ----------------

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction and store results."""
    try:
        details = {
            "patient_id": request.form.get("patient_id", f"PID{datetime.now().strftime('%y%m%d%H%M%S')}"),
            "name": request.form.get("name"),
            "age": request.form.get("age"),
            "gender": request.form.get("gender"),
            "date_of_scan": request.form.get("date_of_scan"),
            "symptoms": request.form.get("symptoms"),
            "family_history": request.form.get("family_history"),
            "head_injury_notes": request.form.get("head_injury_notes", "N/A"),
            "other_conditions": request.form.get("other_conditions", "N/A")
        }

        # File validation
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No file selected.")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Invalid file format. Only JPG/PNG allowed.")
            return redirect(url_for("index"))

        # Save upload
        filename = secure_filename(f"{details['patient_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        details["image_filename"] = filename

        # Run YOLO model
        if model is None:
            raise Exception("Model failed to load.")
        results = model(save_path, conf=0.25, iou=0.45, verbose=False, save=False)

        # Get prediction
        detections = []
        has_detection = results and len(results[0].boxes) > 0

        if has_detection:
            for box in results[0].boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                raw_label = results[0].names[cls]
                mapped_label = CLASS_MAPPING.get(raw_label, raw_label)
                detections.append({"name": mapped_label, "conf": conf})
            best = max(detections, key=lambda d: d["conf"])
            pred_label, confidence = best["name"], best["conf"] * 100
        else:
            pred_label, confidence = "No Tumor", 0.0

        # Annotate and save
        pred_fname = f"pred_{filename}"
        pred_path = os.path.join(app.config["PRED_FOLDER"], pred_fname)
        annotate_image(save_path, results, pred_path)

        # Generate DICOM
        dicom_fname = f"{details['patient_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.dcm"
        dicom_path = os.path.join(app.config["DICOM_FOLDER"], dicom_fname)
        if create_dicom_from_image(save_path, dicom_path, details):
            details["dicom_filename"] = dicom_fname
        else:
            details["dicom_filename"] = "Creation Failed"

        # Add prediction data
        details.update({
            "pred_label": pred_label,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "annotated_url_path": url_for('static', filename=f"predicted/{pred_fname}")
        })

        # Save record to PostgreSQL
        save_user_details(details)

        print(f"[INFO] Prediction complete for {details['patient_id']}")
        return render_template("report.html", **details)

    except Exception as e:
        print(f"[FATAL ERROR] Prediction failed: {e}")
        flash(f"Error: {e}")
        return redirect(url_for("index"))


@app.route('/report/<patient_id>')
def report(patient_id):
    """Fetch single patient report."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patient_records WHERE patient_id = ?", (patient_id,))
    record = cur.fetchone()
    cur.close()
    conn.close()

    if record:
        details = {
            "patient_id": record[0],
            "name": record[1],
            "age": record[2],
            "gender": record[3],
            "date_of_scan": record[4],
            "symptoms": record[5],
            "family_history": record[6],
            "head_injury_notes": record[7],
            "other_conditions": record[8],
            "pred_label": record[9],
            "confidence": record[10],
            "dicom_filename": record[11],
        }
        return render_template("report.html", **details)
    return "No record found."


@app.route('/admin')
def admin():
    """Admin dashboard showing all patients."""
    records = get_all_patients()
    records.sort(key=lambda r: r[13])
    return render_template('admin.html', records=records)


@app.route('/download/<filename>')
def download_dicom(filename):
    """Download DICOM file."""
    try:
        return send_from_directory(app.config["DICOM_FOLDER"], filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

if __name__ == "__main__":
    app.run(debug=True)

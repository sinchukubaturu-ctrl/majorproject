import os
import csv
import numpy as np
from PIL import Image
from datetime import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

# --- Configuration ---
# Define CSV file path (relative to the project root where app.py runs)
CSV_FILE_PATH = 'patient_records.csv' 
CSV_HEADERS = [
    "timestamp", "patient_id", "name", "age", "gender", "symptoms", 
    "dicom_filename", "processed_image_filename", "pred_label", "confidence"
]

# --- CSV Functions ---

def initialize_csv():
    """Ensures the patient record CSV exists with headers."""
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)
            
def save_user_details(details):
    """Appends patient details and prediction to the CSV."""
    initialize_csv()
    
    # Prepare data row based on defined headers. Use 'str' to handle all data types for CSV writing.
    row_data = [str(details.get(h, "N/A")) for h in CSV_HEADERS]
    
    # Append to CSV
    try:
        with open(CSV_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_data)
        print(f"[CSV] Details for {details.get('patient_id', 'Unknown')} saved to CSV.")
    except Exception as e:
        print(f"[ERROR] Error saving to CSV: {e}")

# --- Image-to-DICOM Creation Function ---

def create_dicom_from_image(image_path, dicom_output_path, patient_details):
    """
    Creates a basic DICOM file from a standard image (JPG/PNG) and patient metadata.
    
    Args:
        image_path (str): Path to the source JPG/PNG file.
        dicom_output_path (str): Path where the new .dcm file should be saved.
        patient_details (dict): Dictionary containing patient info from the form.
        
    Returns:
        bool: True on success, False on failure.
    """
    try:
        # 1. Read the image and prepare pixel data
        img = Image.open(image_path).convert('L') # Convert to Grayscale ('L' mode)
        pixel_array = np.array(img, dtype=np.uint8) # Ensure 8-bit unsigned integer data type
        
        # 2. Create the FileDataset (metadata container)
        file_meta = Dataset()
        # Use a standard UID for MR Image Storage
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4' 
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian

        ds = FileDataset(dicom_output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        
        # 3. Insert required metadata
        ds.PatientID = patient_details.get("patient_id", "UNKNOWN")
        ds.PatientName = patient_details.get("name", "ANONYMOUS")
        ds.PatientSex = patient_details.get("gender", "O").upper() # DICOM standard is uppercase
        ds.PatientAge = str(patient_details.get("age", 0)) + 'Y'
        
        # Acquisition/Study Times
        now = datetime.now()
        ds.StudyDate = now.strftime('%Y%m%d')
        ds.StudyTime = now.strftime('%H%M%S')
        
        ds.Modality = 'MR'
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4' 
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        
        # 4. Insert pixel data attributes
        ds.Rows, ds.Columns = pixel_array.shape
        ds.PixelData = pixel_array.tobytes()
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0  # 0 for unsigned integer

        # 5. Save the DICOM file
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(dicom_output_path)
        
        print(f"[DICOM] File created and saved to {dicom_output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Error creating DICOM file from image: {e}")
        return False
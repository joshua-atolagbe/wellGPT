'''

Process PDF files to extract text using OCR.

'''

import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def pdf2Text(pdf_path):

    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            #convert the image from OCR to text
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            text += pytesseract.image_to_string(img_cv, lang='eng')
    return text

# save as text
def saveExtractedText(pdf_path):
    extracted_text = pdf2Text(pdf_path)
    #remove extrawhitespaces
    extracted_text = ' '.join(extracted_text.split())

    #save pdf2text extracted text
    with open(os.path.join("processed", f"{pdf_path.split('/')[-1]}.txt"), "w") as f:
        f.write(extracted_text)

if __name__ == "__main__":
    pdf_directory = "wellcompletionreport"
    
    # Get a list of all PDF files in the directory
    report_paths = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    # Create output directory if it doesn't exist
    os.makedirs("processed", exist_ok=True)

    # Process PDFs in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        _ = list(tqdm(executor.map(saveExtractedText, report_paths), total=len(report_paths)))
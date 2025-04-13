'''
Download well completion reports (Scanned PDFs)
'''

import os
import requests
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from bs4 import BeautifulSoup
import re

def clean_wellname(name):
    return re.sub('[<>:"/\\|?*]', '_', name)

def getPDFSize(url):
    response = requests.head(url)
    if response.status_code == 200:
        file_size = response.headers.get('Content-Length')
        if file_size:
            file_size_mb = f"{(int(file_size) / (1024 * 1024)):.2f}MB"  # Convert bytes to megabytes
            return file_size_mb
        else:
            return "Unknown size"
    else:
        return "Error fetching size"
    
def remove_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def downloadReports(pdf_url, wellname):

    response = requests.get(pdf_url)
    if response.status_code == 200:
        # wellname = clean_wellname(wellname)
        pdf_path = os.path.join("wellcompletionreport", pdf_url.split('/')[-1])
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        #save to drive
        # !cp -r pdf_path -d "/content/drive/My Drive/LLM/pdfs"
        print(f"Saved to drive: {pdf_path}")
        print()
    else:
        print(f"PDF not found for URL: {pdf_url}")

if __name__ == "__main__":
    
    # Load the CSV file
    pdfs_url = pd.read_csv('well_completion_report_links2.csv')

    # create directory if it doesn't exist
    os.makedirs("wellcompletionreport", exist_ok=True)

    # Download PDFs in parallel
    with ProcessPoolExecutor(max_workers=10) as executor:
        _ = list(tqdm(executor.map(downloadReports, pdfs_url.link, pdfs_url.well), total=len(pdfs_url.link))) #took 5mins to download
'''
Scrap well completion pdf links
'''

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm


#check each link and get all links ending with .pdf
pattern = r".*[Cc]ompletion[_]?[Rr]eport\.pdf$" 
pattern2 = r".*[Cc]ompletion[_]?[Rr]eport(_[Aa]nd[_]?[Ll]og)?\.pdf$"

def wellCompletionReportLink(link, wells):

    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')
    links_ = [a['href'] for a in soup.find_all('a', href=True)]

    #retain only links that has completion report in them, using regex
    pdf_links = [link for link in links_ if re.match(pattern2, link)]

    return wells, pdf_links

if __name__ == "__main__":

    df = pd.read_csv('wellbore_links.csv')

    with ProcessPoolExecutor(max_workers=10) as executor:
        report_links = list(tqdm(executor.map(wellCompletionReportLink, df.link, df.wellname), total=len(df.link)))

    pdf_links = pd.DataFrame(report_links, columns=['well', 'link'])
    pdfs = pdf_links.explode('link').dropna(subset=['link']).reset_index(drop=True)

    pdfs.to_csv('well_completion_report_links2.csv', index=False)
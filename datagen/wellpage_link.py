'''
Scrap all wellbore links
'''

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# Function to fetch well name for each link
def fetch_wellname(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')
    wellname = soup.find('div', id='title-and-controls').find('h1').text
    return wellname, link

if __name__ == "__main__":
    # URL to scrape
    url = "https://factpages.sodir.no/en/wellbore/PageView/Exploration/All"
    pattern = r"https://factpages\.sodir\.no/en/wellbore/PageView/Exploration/All/\d+"

    # Fetch and parse the main page
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all links matching the pattern
    links = [a['href'] for a in soup.find_all('a', href=True)]
    filtered_links = sorted([link for link in links 
                             if re.match(pattern, link)])

    # Parallelize requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(fetch_wellname, filtered_links),
                             total=len(filtered_links)))

    # Save to DataFrame
    df = pd.DataFrame(results, columns=['wellname', 'link'])

    df.to_csv('wellbore_links.csv', index=False)

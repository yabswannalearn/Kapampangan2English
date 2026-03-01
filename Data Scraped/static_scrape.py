import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

url = "https://kapampangantraveller.com/blog/"

# Added User-Agent to prevent the server from blocking the automated request
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

data = []

# Iterate through all bulleted list items on the page
for li in soup.find_all('li'):
    text = li.get_text(strip=True)
    
    # Split the string at the first instance of a hyphen, en-dash, or colon
    parts = re.split(r'[-–:]', text, maxsplit=1)
    
    if len(parts) == 2:
        english = parts[0].strip()
        kapampangan = parts[1].strip()
        
        # Filter out random UI elements by ensuring both sides have text 
        # and the English side isn't a massive paragraph
        if english and kapampangan and len(english) < 100:
            data.append([kapampangan, english])

df = pd.DataFrame(data, columns=['Kapampangan', 'English'])
df.to_csv("kapampangan_traveller.csv", index=False)

print(f"Successfully extracted {len(df)} pairs to kapampangan_traveller.csv")
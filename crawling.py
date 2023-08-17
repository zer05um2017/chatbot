import requests
from bs4 import BeautifulSoup

url = 'https://www.kahp.or.kr/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

for link in soup.find_all('a'):
    href = link.get('href')
    if href and href.endswith('.html'):
        file_url = url + href
        r = requests.get(file_url, stream=True)
        with open(href.split('/')[-1], 'wb') as file:
            for chunk in r.iter_content(chunk_size=1024):
                file.write(chunk)

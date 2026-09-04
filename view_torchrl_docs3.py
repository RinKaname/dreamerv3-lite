import urllib.request
from bs4 import BeautifulSoup
url = "https://docs.pytorch.org/rl/main/reference/dreamer_v3.html"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')

    # Just look at headers and paragraphs
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
        print(tag.get_text())
except Exception as e:
    print(f"Error fetching docs: {e}")

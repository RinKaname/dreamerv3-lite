import urllib.request
from bs4 import BeautifulSoup
url = "https://docs.pytorch.org/rl/main/reference/dreamer_v3.html"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for i, line in enumerate(lines):
        if "DreamerV3" in line:
            start = i
            break
    for i in range(start, start+100):
        print(lines[i])
except Exception as e:
    print(f"Error fetching docs: {e}")

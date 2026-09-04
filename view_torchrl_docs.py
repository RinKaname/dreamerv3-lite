import urllib.request
url = "https://docs.pytorch.org/rl/main/reference/dreamer_v3.html"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    print("Successfully fetched docs")
    # Quick extract of some text to see what it covers
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for i, line in enumerate(lines[:50]):
        print(line)
except Exception as e:
    print(f"Error fetching docs: {e}")

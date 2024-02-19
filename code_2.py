import requests
from bs4 import BeautifulSoup

url = 'https://fishbase.ropensci.org/species/'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# create an empty list to store the image links and names
image_links = []

# find all <img> tags on the page
for img in soup.find_all('img'):
    # get the value of the 'src' attribute, which contains the URL of the image
    src = img['src']
    # get the filename from the URL using regular expressions
    import re
    match = re.search(r'/(www.fishbase.org\/\d+\/\d+\/\d+\/\d+\)\.(jpg|jpeg)$/', src)
    if match:
        filename = match.group(1)
        # add the image link and name to the list
        image_links.append((filename, src))

# print the image links and names
for i, (filename, src) in enumerate(image_links):
    print(f"{i+1}. {filename} - {src}\n")

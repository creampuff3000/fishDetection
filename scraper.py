from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# enable headless mode in Selenium
options = Options()
options.add_argument('--headless=new')

driver = webdriver.Chrome(
    options=options, 
    # other properties...
)

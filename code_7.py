# import stuff
# get species common name from file
# get on fishBase
# search common name
# go to first page
# collect image with img tag
# output image into file
# repeat until common names are finished
# from bs4 import BeautifulSoup
# import requests
import csv
import os
import requests
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
url = "https://www.google.com/search?sca_esv=598932482&rlz=1C5MACD_enUS1040US1040&q=fish.&tbm=isch&source=lnms&sa=X&ved=2ahUKEwiD4NnDguODAxVqJUQIHb0NCu0Q0pQJegQIDRAB&biw=1512&bih=858&dpr=2"
folder_path = '/Users/justinluo/projects/fishDetection/output'
options = Options()

driver = webdriver.Chrome(
    options=options, 
)

driver.get(url)
commonName = ""
scientificName = ""
with open('SpeciesList.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        count = 0
        commonName = row[5]
        scientificName = row[3] + " " + row[4]
        name_input = driver.find_element(By.XPATH, f'//*[@id="REsRA"]')
        name_input.clear()
        name_input.send_keys(commonName)
        name_input.send_keys(Keys.RETURN)
        for _ in range(100):
            div_element = driver.find_element(By.CLASS_NAME, 'mJxzWe')
            img = driver.find_element(By.TAG_NAME, 'img')
            img_link = img.get_attribute('src')
            data = requests.get(img_link)
            file_path = os.path.join(folder_path, f'{commonName}.jpg')
            with open(file_path, 'ab') as file:
                file.write(data.content)
while(True):
    pass
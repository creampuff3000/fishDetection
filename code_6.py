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
url = "https://www.fishbase.org.au/v4"
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
        name_input = driver.find_element(By.XPATH, f'//*[@id="main-search-input"]')
        name_input.send_keys(commonName)
        name_input.send_keys(Keys.RETURN)
        button = driver.find_element(By.XPATH, f'//span[text()="{scientificName}"]')
        button.click()
        buttons = driver.find_elements(By.XPATH, f'//*[@id="species-images-ow"]/div[3]')
        length = len(buttons)
        print(buttons)
        print(length)
        for _ in range(length):
            button2 = driver.find_element(By.XPATH, f'//*[@id="species-images-ow"]/div[3]/button[{count + 1}]')
            button2.click()
            img = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, f'//*[@id="species-images-item{count}"]/a/img')))
            img_link = img.get_attribute('src')
            data = requests.get(img_link)
            file_path = os.path.join(folder_path, f'{commonName}.jpg')
            with open(file_path, 'ab') as file:
                file.write(data.content)
while(True):
    pass
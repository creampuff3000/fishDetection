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
# page = requests.get(url)
# soup = BeautifulSoup(requests.get(url).content, 'html.parser')
options = Options()

driver = webdriver.Chrome(
    options=options, 
)

driver.get(url)
commonName = "California Scorpionfish"
scientificName = "Scorpaena guttata"
# name_input = WebDriverWait(driver, 10).until(
#     EC.presence_of_element_located((By.ID, 'main-search-input'))
# )
name_input = driver.find_element(By.ID, "main-search-input")
name_input.send_keys(commonName)
name_input.send_keys(Keys.RETURN)
button = driver.find_element(By.XPATH, f'//span[text()="{scientificName}"]')
button.click()
thing = driver.find_element(By.XPATH, f'//*[@id="biology"]')
# thing2 = driver.find_elements(By.ID, 'size')
thing.click()
biology = driver.find_element(By.XPATH, f'//*[@id="biology"]/div/p')
biology_text = biology.text
print(commonName)
print(biology_text)
# print(thing2)
# driver.back()
# driver.back()
# driver.quit()
while(True):
    pass
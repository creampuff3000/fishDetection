import csv
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time

url = "https://www.google.com/search?sca_esv=598932482&rlz=1C5MACD_enUS1040US1040&q=fish.&tbm=isch&source=lnms&sa=X&ved=2ahUKEwiD4NnDguODAxVqJUQIHb0NCu0Q0pQJegQIDRAB&biw=1512&bih=858&dpr=2"
folder_path = '/Users/justinluo/projects/fishDetection/output'

options = webdriver.ChromeOptions()


driver = webdriver.Chrome(options=options)

driver.get(url)


def collect_images(common_name, counter, limit): # function for easier functionality
    try:
        img_elements = driver.find_elements(By.XPATH, '//img[@data-src]')
        
        for i, img in enumerate(img_elements):
            img_link = img.get_attribute('data-src')
            
            response = requests.get(img_link, stream=True) # stream makes it so it doesn't download all the images at the same time, reducing lag
            content_length = int(response.headers.get('content-length', 0)) # getting dimensions
            
            if content_length >= 1024: # making sure it doesn't download website icons
                data = response.content
                
                file_path = os.path.join(folder_path, f'{common_name}_{counter + i + 1}.jpg') # naming image
                with open(file_path, 'ab') as file:
                    file.write(data)
                
                if counter + i + 1 >= limit: # making sure it isn't at limit
                    return
    except Exception as e:
        print(f"Error collecting images for {common_name}: {e}")

# ...

with open('SpeciesList.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        common_name = row[5]                     
        scientific_name = row[3] + " " + row[4]
        image_counter = 0 
        images_per_species_limit = 100 
        
        name_input = driver.find_element(By.XPATH, '//input[@name="q"]')
        name_input.clear()
        name_input.send_keys(common_name + " fish")
        name_input.send_keys(Keys.RETURN)
        
        time.sleep(2)  
        
        while image_counter < images_per_species_limit:
            collect_images(common_name, image_counter, images_per_species_limit)
            image_counter += len(driver.find_elements(By.XPATH, '//img[@data-src]'))
            
            ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform() # scrolling down
            time.sleep(2)

driver.quit()


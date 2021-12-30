# https://stackoverflow.com/questions/38081021/using-selenium-on-mac-chrome
# Install the web driver package
import urllib
from io import StringIO

import selenium
import time

# from selenium.webdriver.chrome import webdriver
from PIL import Image
from google.auth.transport import requests
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import urllib.request



import pymongo

#https://medium.com/geekculture/scraping-images-using-selenium-f35fab26b122
path = "C:\Program Files (x86)\Google\chromedriver.exe"
photofolder = "/Users/stuar/Desktop/TrainingData/FashionGen"

urbanoutfitters = "https://www.urbanoutfitters.com/womens-tops?page=8"

# Reiss works! https://www.reiss.com/us/mens/coats-jackets/jackets/
#Best website yet!!! SSENSE

site = "https://www.ssense.com/en-us/men/designers/barena/clothing"
x = 12200


s = Service(path)

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument(f'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, service=s)


temp = urbanoutfitters
print(site)
driver.get(site)
# print(driver.page_source)
time.sleep(3)

# visited = []
#
# while(len(div)):
#     s = div[-1]
#     div.pop()
#     if s not in visited:
#         visited.append(s)
#         for x in s.find_elements(By.CLASS_NAME, "c-pwa-tile-tiles")



# images = driver.find_elements(By.TAG_NAME, "img")
# images = driver.find_elements(By.CSS_SELECTOR, "[data-auto-id='image']")

path = "C:/Users/stuar/Desktop/TrainingData/temp/"

h = driver.get_window_size()
height = h['height']
current = 0
for g in range(500):
    driver.execute_script(f"window.scrollTo(0, {current})")
    images = WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.TAG_NAME, "img")))
    images = list(set(images))
    current += height
    print(g)


time.sleep(2)
print("Parsing")
print(len(images))

for i in images:
    href = i.get_attribute("srcset")
    dest = (path+"mens_" + str(x)+".jpg")
    # print(i.get_attribute("class"))
    # print("photo: ", dest)
    # urllib.request.urlretrieve(href, dest)
    # print(href)
    print(href)
    print(x)

    try:
        urllib.request.urlretrieve(href, dest)
    except:
        print("Image error")
    x += 1

# src = img.get_attribute('src')
# print(src)
# # download the image
# urllib.urlretrieve(src, "captcha.png")

time.sleep(2)




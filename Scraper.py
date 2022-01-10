# https://stackoverflow.com/questions/38081021/using-selenium-on-mac-chrome
# Install the web driver package
import time

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import urllib.request


path = "C:\Program Files (x86)\Google\chromedriver.exe"
current = "https://www.ssense.com/en-us/men/designers/dries-van-noten/clothing?page=9"

women = "https://www.ssense.com/en-us/women/clothing?page=8"

x = 27000
s = Service(path)

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument(
    f'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(options=options, service=s)

driver.get(current)

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


for i in images:
    href = i.get_attribute("srcset")
    dest = (path + "mens_" + str(x) + ".jpg")
    # TODO UNCOMMENT TO TAILOR BOT TO WEBSITE
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



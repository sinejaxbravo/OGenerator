# https://stackoverflow.com/questions/38081021/using-selenium-on-mac-chrome
# Install the web driver package
import selenium
import time

# from selenium.webdriver.chrome import webdriver
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import pymongo

#https://medium.com/geekculture/scraping-images-using-selenium-f35fab26b122
class Scraper:

    print()
    driver = webdriver.Chrome()
    driver.get("https://unsplash.com/s/photos/mens-fashion")
    time.sleep(2)

    def fetch_image_urls(query: str, max_links_to_fetch: int, wd: webdriver, sleep_between_interactions: int = 1):
        def scroll_to_end(wd):
            wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(sleep_between_interactions)

            # build the google query

        search_url = "https://unsplash.com/s/photos/mens-fashion"

        wd.get(search_url.format(q=query))

        image_urls = set()
        image_count = 0
        results_start = 0
        while image_count < max_links_to_fetch:
            scroll_to_end(wd)

            # get all image thumbnail results
            thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
            number_results = len(thumbnail_results)

            print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

            for img in thumbnail_results[results_start:number_results]:
                # try to click every thumbnail such that we can get the real image behind it
                try:
                    img.click()
                    time.sleep(sleep_between_interactions)
                except Exception:
                    continue

                # extract image urls
                actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
                for actual_image in actual_images:
                    if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                        image_urls.add(actual_image.get_attribute('src'))

                image_count = len(image_urls)

                if len(image_urls) >= max_links_to_fetch:
                    print(f"Found: {len(image_urls)} image links, done!")
                    break
            else:
                print("Found:", len(image_urls), "image links, looking for more ...")
                time.sleep(30)
                return
                load_more_button = wd.find_element_by_css_selector(".mye4qd")
                if load_more_button:
                    wd.execute_script("document.querySelector('.mye4qd').click();")

            # move the result startpoint further down
            results_start = len(thumbnail_results)

        return image_urls

        # headlines = driver.find_elements_by_class_name("story-heading")
        # for headline in headlines:
        #     print(headline.text.strip())

        # # service = Service("/usr/local/bin/chromedriver")
        # # browser = webdriver.Chrome(service=service)
        #
        # # service.get("https://unsplash.com/s/photos/mens-fashion")
        #
        # # Press ⌃R to execute it or replace it with your code.
        # # Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
        # #
        # # #See this https://oxylabs.io/blog/python-web-scraping#libraries
        # # response = requests.get()
        # # response.content
        # # print(response.text)

    def persist_image(folder_path:str,url:str, counter):
        try:
            image_content = requests.get(url).content

        except Exception as e:
            print(f"ERROR - Could not download {url} - {e}")

        try:
            f = open(os.path.join(folder_path, 'jpg' + "_" + str(counter) + ".jpg"), 'wb')
            f.write(image_content)
            f.close()
            print(f"SUCCESS - saved {url} - as {folder_path}")
        except Exception as e:
            print(f"ERROR - Could not save {url} - {e}")



    def search_and_download(search_term: str, driver_path: str, target_path='./images', number_images=10):
        target_folder = os.path.join(target_path, '_'.join(search_term.lower().split(' ')))

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        with webdriver.Chrome(executable_path=driver_path) as wd:
            res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)

        counter = 0
        for elem in res:
            persist_image(target_folder, elem, counter)
            counter += 1

    DRIVER_PATH = './chromedriver'
    search_term = 'cat'
    # num of images you can pass it from here  by default it's 10 if you are not passing
    # number_images = 10
    search_and_download(search_term=search_term, driver_path=DRIVER_PATH)
import time
from bs4 import BeautifulSoup
from lxml import etree
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def proc_selenium():
    # Set up headless Chrome
    options = Options()
    options.headless = True  # Runs Chrome in headless mode (no UI)
    driver = webdriver.Chrome(options=options)

    site_url = "https://archive.is/pestrecy-rt.ru"

    # Load the webpage
    driver.get(site_url)

    # Pause to allow for manual CAPTCHA check if needed
    print("Waiting for you to solve the CAPTCHA...")
    time.sleep(30)  # Adjust time as needed for manual intervention

    # Parse the page source with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Convert BeautifulSoup object to lxml for XPath
    dom = etree.HTML(str(soup))

    # Find all `div` elements with the class "TEXT-BLOCK"
    # Select the second `href` link within each `div` of class "TEXT-BLOCK"
    links = dom.xpath('//div[@class="TEXT-BLOCK"]//a[2]/@href')

    # Close the driver after scraping
    driver.quit()
    return links

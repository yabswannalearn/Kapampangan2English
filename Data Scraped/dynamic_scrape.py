from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import time

# Setup the driver
driver = webdriver.Chrome() 
url = "https://www.tagaloglang.com/kapampangan-to-english/"
driver.get(url)

all_data = []

try:
    # 1. Wait for JavaScript to fully initialize the DataTables structure
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table.dataTable tbody tr"))
    )
    time.sleep(1) # Brief pause to let any ads/layout shifts settle

    for page in range(1, 32): 
        print(f"Scraping page {page}...")
        
        # 2. Try/Except block to catch any stray Stale Element errors dynamically
        retries = 3
        for attempt in range(retries):
            try:
                rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                
                # Remember the first row's text to detect when the page actually flips
                if len(rows) > 0:
                    first_row_text = rows[0].text
                
                page_data = []
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if len(cols) >= 2:
                        kapampangan = cols[0].text.strip()
                        english = cols[1].text.strip()
                        page_data.append([kapampangan, english])
                
                # If successful, add to our master list and break out of the retry loop
                all_data.extend(page_data)
                break 
                
            except StaleElementReferenceException:
                if attempt == retries - 1:
                    raise # If it fails 3 times, throw the error
                print("Table updated during read. Retrying...")
                time.sleep(1)

        # 3. Find and click the 'Next' button if it's not the last page
        if page < 31:
            next_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "button.dt-paging-button.next"))
            )
            
            # Use JS to click in case ads or sticky headers are covering the button
            driver.execute_script("arguments[0].click();", next_button)
            
            # 4. Smart wait: Wait until the first row's text is DIFFERENT from the previous page
            WebDriverWait(driver, 10).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, "table tbody tr")[0].text != first_row_text
            )

finally:
    driver.quit()

# Save to CSV
df = pd.DataFrame(all_data, columns=['Kapampangan', 'English'])
df.to_csv("kapam_2_eng.csv", index=False)
print("Finished! Data saved to kapam_2_eng.csv")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import random

def parse_sravni_cards():
    # Configure Chrome options
    chrome_options = Options()
    # Run in visible mode for debugging
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Add user agent to appear more like a real browser
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    ]
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
    
    # Disable webdriver detection
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    try:
        # Navigate to the URL
        print("Opening website...")
        driver.get("https://www.sravni.ru/karty/")
        
        # Wait for the page to load completely
        print("Waiting for page to load...")
        time.sleep(10)  # Initial wait
        
        # Take a screenshot to see what's loaded
        driver.save_screenshot("initial_page.png")
        print("Saved initial page screenshot")
        
        # Dump page source to analyze
        with open("initial_page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved initial page source")
        
        # Scroll down to load more content
        print("Scrolling to load more content...")
        for i in range(5):
            driver.execute_script(f"window.scrollTo(0, {i * 1000});")
            time.sleep(2)
        
        # Take another screenshot after scrolling
        driver.save_screenshot("after_scroll.png")
        print("Saved post-scroll screenshot")
        
        # Dynamically find all possible card containers
        print("Looking for card elements...")
        
        # Try to find any elements that might be cards
        potential_card_selectors = [
            # Try common card patterns
            "div.card", "div[class*='card']", "div[class*='Card']",
            "article", "div.product", "div.item", "div[class*='product']",
            "div[class*='item']", ".CardLayout_card", "div[class*='CardLayout']",
            # More specific to financial sites
            "div.creditCard", "div[class*='creditCard']", "div[class*='bank-card']",
            # XPath alternatives
            "//div[contains(@class, 'card')]", "//div[contains(@class, 'Card')]",
            "//article", "//div[contains(@class, 'product')]",
            # Try data attributes
            "div[data-qa*='card']", "div[data-test*='card']", "div[data-*='card']",
        ]
        
        # Try each selector and see which one finds elements
        found_cards = False
        card_elements = []
        
        for selector in potential_card_selectors:
            try:
                if selector.startswith("//"):
                    elements = driver.find_elements(By.XPATH, selector)
                else:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                if elements and len(elements) > 3:  # Assuming we should find multiple cards

                    print(f"Found {len(elements)} potential cards with selector: {selector}")
                    
                    # Check if these elements look like cards (have title, description, etc.)
                    sample = elements[0]
                    html = sample.get_attribute("outerHTML")
                    if len(html) > 100 and ("bank" in html.lower() or "card" in html.lower() or "credit" in html.lower()):
                        print(f"Selector {selector} appears to match card elements")
                        card_elements = elements
                        found_cards = True
                        break
            except Exception as e:
                print(f"Error with selector {selector}: {e}")
        
        if not found_cards:
            # If structured approach fails, try to find anything that looks like a card
            print("Structured approach failed, trying to find elements by content...")
            
            try:
                # Look for elements containing card-related text
                card_elements = driver.find_elements(By.XPATH, 
                    "//*[contains(text(), 'карт') or contains(text(), 'Карт') or contains(text(), 'кредит') or contains(text(), 'Кредит')]")
                
                if card_elements and len(card_elements) > 5:
                    # Navigate up a few levels to find container elements
                    potential_containers = []
                    for element in card_elements[:5]:  # Check first few elements
                        try:
                            # Try to go up to parent containers
                            parent = element
                            for _ in range(3):  # Go up to 3 levels
                                parent = parent.find_element(By.XPATH, "..")
                                potential_containers.append(parent)
                        except:
                            pass
                    
                    if potential_containers:
                        # Use the largest container that's not the whole page
                        potential_containers.sort(key=lambda x: len(x.get_attribute("outerHTML")))
                        card_elements = potential_containers[:10]  # Take up to 10 potential card containers
                        found_cards = True
            except Exception as e:
                print(f"Error in content-based approach: {e}")
        
        # If still no cards found
        if not found_cards or not card_elements:
            print("Could not identify card elements on the page. Saving final page state for analysis.")
            driver.save_screenshot("final_state.png")
            with open("final_page_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            return []
        
        # Process the found card elements
        print(f"Processing {len(card_elements)} card elements...")
        
        # Save an example card HTML for debugging
        with open("example_card.html", "w", encoding="utf-8") as f:
            if card_elements:
                f.write(card_elements[0].get_attribute("outerHTML"))
        
        # Extract data from cards
        card_data = []
        for i, card in enumerate(card_elements[:20]):  # Process up to 20 cards
            try:
                card_info = {}
                
                # Save individual card HTML for inspection
                with open(f"card_{i}.html", "w", encoding="utf-8") as f:
                    f.write(card.get_attribute("outerHTML"))
                
                # Try various selectors for card name
                for name_selector in ["h2", "h3", "h4", ".title", "[class*='title']", "[class*='name']", "strong"]:
                    try:
                        elements = card.find_elements(By.CSS_SELECTOR, name_selector)
                        if elements:
                            card_info["card_name"] = elements[0].text
                            break
                    except:
                        pass

                # Try to find bank name
                for bank_selector in ["[class*='bank']", "[data-test*='bank']", "[class*='issuer']", "a", "img"]:
                    try:
                        elements = card.find_elements(By.CSS_SELECTOR, bank_selector)
                        for element in elements:
                            text = element.text
                            if text and len(text) < 50:  # Reasonable bank name length
                                card_info["bank_name"] = text
                                break
                            # Try to get alt text from images
                            if element.tag_name == "img":
                                alt = element.get_attribute("alt")
                                if alt and "банк" in alt.lower():
                                    card_info["bank_name"] = alt
                                    break
                    except:
                        pass
                
                # Try to find features
                features = []
                for feature_selector in ["li", "[class*='feature']", "[class*='benefit']", "ul li", "p"]:
                    try:
                        elements = card.find_elements(By.CSS_SELECTOR, feature_selector)
                        for element in elements:
                            text = element.text.strip()
                            if text and len(text) < 200:  # Reasonable feature text length
                                features.append(text)
                    except:
                        pass
                
                card_info["features"] = features
                
                # Add any other fields you can extract
                
                # Get all text from the card as fallback
                if not card_info.get("card_name"):
                    card_info["card_name"] = card.text[:50] if card.text else "Unknown Card"
                
                card_data.append(card_info)
                print(f"Processed card {i+1}/{len(card_elements[:20])}")
                
            except Exception as e:
                print(f"Error processing card {i}: {e}")
        
        # Save data to CSV
        if card_data:
            df = pd.DataFrame(card_data)
            df.to_csv("sravni_cards.csv", index=False, encoding="utf-8-sig")
            print("Data saved to sravni_cards.csv")
        else:
            print("No card data was extracted")
        
        return card_data
        
    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    parse_sravni_cards()

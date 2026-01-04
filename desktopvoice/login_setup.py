from playwright.sync_api import sync_playwright
import os

# Mac uses /Users/username/... but expanduser handles '~' correctly
PROFILE_DIR = os.path.expanduser("~/.playwright_profile")

def login_procedure_mac():
    with sync_playwright() as p:
        print("Launching Chrome on macOS...")
        
        browser = p.chromium.launch_persistent_context(
            user_data_dir=PROFILE_DIR,
            headless=False,
            channel="chrome", # Looks for /Applications/Google Chrome.app
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
            # On Mac, the 'infobar' that says 'Chrome is being controlled' is aggressive
            ignore_default_args=["--enable-automation"]
        )
        
        page = browser.pages[0]
        page.goto("https://gemini.google.com/app")
        
        print("--- ACTION REQUIRED ---")
        print("1. Log in to Google.")
        print("2. IMPORTANT: If macOS asks 'Terminal wants to control Google Chrome', click OK.")
        print("3. Close the browser window when you see the chat box.")
        
        try:
            page.wait_for_timeout(9999999)
        except:
            print("Setup complete.")

if __name__ == "__main__":
    login_procedure_mac()
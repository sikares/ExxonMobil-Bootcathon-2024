import os
import re
import requests
from bs4 import BeautifulSoup

def scrape_url(url):
    try:
        # Request the HTML page
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        return "", f"Failed to fetch URL: {e}"

    # Load the HTML document
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the <main> tag and extract text
    main_tag = soup.find('main')
    if main_tag:
        main_text = main_tag.get_text(strip=True)
    else:
        main_text = "No <main> tag found"

    return main_text, None

def sanitize_file_name(file_name):
    # Replace all non-alphanumeric characters with underscores
    return re.sub(r'[^a-zA-Z0-9]+', '_', file_name)

def save_to_file(file_path, content):
    try:
        # Create and write to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except OSError as e:
        return f"Failed to write to file: {e}"
    return None

def read_urls(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            urls = [line.strip() for line in file.readlines()]
        return urls, None
    except OSError as e:
        return [], f"Failed to read file: {e}"

def main():
    urls_file = 'found_urls.txt'
    urls, err = read_urls(urls_file)
    if err:
        print(f"Error reading URLs file: {err}")
        return

    # Create data directory if it doesn't exist
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)

    for url in urls:
        text, err = scrape_url(url)
        if err:
            print(f"Error scraping {url}: {err}")
            continue

        # Create a sanitized file name based on the URL
        file_name = sanitize_file_name(url) + ".txt"
        file_path = os.path.join(data_dir, file_name)

        # Save the content to a file
        err = save_to_file(file_path, text)
        if err:
            print(f"Error saving file {file_path}: {err}")
            continue

        print(f"Saved content from {url} to {file_path}")

if __name__ == "__main__":
    main()
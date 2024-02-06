'''
COMS E6111 - Project 1
Sharmista Shastry (ss6950) and Cindy Ruan (cxr2000)
'''

import sys
import requests

def google_search(api_key, engine_id, query):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={engine_id}&q={query}"
    response = requests.get(url)
    data = response.json()
    if 'items' in data:
        return [item for item in data['items']]
    else:
        return []

def display_results(results):
    print("======================")
    for i, result in enumerate(zip(results), 1):
        print(f"Result {i}:")
        print(f"URL: {result.get('link', 'N/A')}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Summary: {result.get('snippet', 'N/A')}")
        print()

def main(api_key, engine_id, precision, query):
    print("Parameters:")
    print(f"Client key  = {api_key}")
    print(f"Engine key  = {engine_id}")
    print(f"Query       = {query}")
    print(f"Precision   = {precision}")
    print("Google Search Results:")

    results = google_search(api_key, engine_id, query)
    while True:
        if not results:
            print("No results. Exiting...")
            break
        display_results(results)
        


if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) != 5:
        print("Usage: python main.py <google api key> <google engine id> <precision> <query>")
        sys.exit(1)

    # Get command line arguments
    api_key = sys.argv[1]
    engine_id = sys.argv[2]
    precision = sys.argv[3]
    query = sys.argv[4]

    # Check if precision is between 0 and 1
    try:
        precision = float(precision)
        if precision < 0 or precision > 1:
            raise ValueError
    except ValueError:
        print("Precision must be between 0 and 1")
        sys.exit(1)

    

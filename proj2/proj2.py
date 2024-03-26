import sys
import requests
from bs4 import BeautifulSoup, Comment
import spacy
from spacy.tokens import Span
from collections import defaultdict

# Function to query Google Custom Search Engine and retrieve URLs for the top-10 webpages
def google_search(api_key, engine_id, query):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={engine_id}&q={query}&num=10"
    try:
        response = requests.get(url)
        data = response.json()
        if 'items' in data:
            return [item['link'] for item in data['items']]
    except Exception as e:
        print(f"Error querying Google Custom Search Engine: {e}")
    return []


# Function to retrieve webpage content and extract plain text using BeautifulSoup
def retrieve_webpage(url):
    try:
        # may still need to fix extraction method as character number is a bit off
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator='\n', strip=True)
            text = ' '.join(text.split())
        
            webpage_length = len(text)
            if webpage_length > 10000:
                text = text[:10000]
                print(f"\tTrimming webpage content from {webpage_length} to 10000 characters")
            return text
    except Exception as e:
        print(f"Error retrieving webpage: {e}")
    return None

def spacy_function(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # sentences = [sent.text for sent in doc.sents]
    # entities = [(ent.text, ent.label_) for ent in doc.ents]
    return doc
        

# function that finds the corresponding relation to input int
def relation(relation_type):
    r = ''
    if relation_type == 1:
        r = "Schools_Attended"
    elif relation_type == 2:
        r = "Work_For"
    elif relation_type == 3:
        r = "Live_In"
    else:
        r = "Top_Member_Employees"
    return r


def main(method, api_key, engine_id, gemini, relation, threshold, query, k_tuples):
    while True:
        print("Parameters:")
        print(f"Client key  = {api_key}")
        print(f"Engine key  = {engine_id}")
        print(f"Gemini key  = {gemini}")
        print(f"Method  = {method}")
        print(f"Relation    = {relation(relation_type)}")
        print(f"Threshold   = {threshold}")
        print(f"Query       = {query}")
        print(f"# of Tuples = {k_tuples}")

        print("Loading necessary libraries; This should take a minute or so ...")
        
        urls = google_search(api_key, engine_id, query)
        if not urls:
            print("No URLs found from Google Custom Search Engine.")
            break

        print("Top 10 URLs:")
        for idx, url in enumerate(urls, start=1):
            print(f"URL ({idx} / 10): {url}")
            print("\tFetching text from url ...")
            webpage_content = retrieve_webpage(url)
            if not webpage_content:
                print(f"Error retrieving content from {url}. Skipping...")
                continue
            else:
                webpage_length = len(webpage_content)
                print(f"\tWebpage length (num characters): {webpage_length}")
                print("\tAnnotating the webpage using spacy...")
                doc = spacy_function(webpage_content)
                # print(f"\tExtracted {len(sentences)} sentences. Processing each sentence one by one to check for the presence of the right pair of named entity types; if so, will run the second pipeline ...")
                

                entities_of_interest = ["ORGANIZATION", "PERSON"] 

                from spanbert import SpanBERT 
                spanbert = SpanBERT("./pretrained_spanbert")  

                # Extract relations
                from spacy_help_functions import extract_relations
                relations = extract_relations(doc, spanbert, entities_of_interest)
                print("Relations: {}".format(dict(relations)))
        # Further processing steps (entity recognition, relation extraction) will go here

      
        break

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python proj2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>")
        sys.exit(1)

    method = sys.argv[1][1:]
    api_key = sys.argv[2]
    engine_id = sys.argv[3]
    gemini = sys.argv[4]
    relation_type = int(sys.argv[5])

    if relation_type < 1 or relation_type > 4:
        print("Relation must be 1, 2, 3, or 4")
        sys.exit(1)

    threshold = float(sys.argv[6])

    if threshold < 0 or threshold > 1:
        print("Threshold must be between 0 and 1")
        sys.exit(1)

    query = sys.argv[7]
    k_tuples = int(sys.argv[8])


    main(method, api_key, engine_id, gemini, relation, threshold, query, k_tuples)

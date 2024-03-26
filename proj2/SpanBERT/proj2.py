import sys
import requests
from bs4 import BeautifulSoup, Comment
import spacy
from spacy.tokens import Span
from collections import defaultdict
from spanbert import SpanBERT
from spacy_help_functions import extract_relations
import os
import google.generativeai as genai

# Load pre-trained SpanBERT model
spanbert = SpanBERT("./pretrained_spanbert")

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
        response = requests.get(url)

        # This will raise an HTTPError for bad responses
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        
        # Get text and clean whitespace
        text = soup.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())

        webpage_length = len(text)
        if webpage_length > 10000:
            text = text[:10000]
        
        return text

    except requests.HTTPError as e:
        print(f"HTTP error retrieving webpage: {e}")
    except requests.RequestException as e:
        print(f"Error retrieving webpage: {e}")
    return None


def spacy_function(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    return doc

def is_valid_pair(pair, relation_type):
    if relation_type == "Schools_Attended":
        return pair[0].label_ == 'PERSON' and pair[1].label_ == 'ORG'
    elif relation_type == "Work_For":
        return pair[0].label_ == 'PERSON' and pair[1].label_ == 'ORG'
    elif relation_type == "Live_In":
        return pair[0].label_ == 'PERSON' and pair[1].label_ in ['LOC', 'GPE', 'FAC']
    elif relation_type == "Top_Member_Employees":
        return pair[0].label_ == 'ORG' and pair[1].label_ == 'PERSON'
    else:
        return False

def run_spanbert(doc, spanbert, relation_type):
    # Extract relations using the helper functions and SpanBERT model
    entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    entity_pairs = extract_relations(doc, spanbert, relation_type, entities_of_interest)

    # If no entity pairs were found, return an empty list
    if not entity_pairs:
        print("No entity pairs found.")
        return []

    # Run SpanBERT on the entity pairs
    relations = spanbert.predict(entity_pairs)
    return relations

# Generate response to prompt
def get_gemini_completion(prompt, model_name, max_tokens, temperature, top_p, top_k, api_key):
    # Initialize a generative model
    model = genai.GenerativeModel(model_name, api_key=api_key)

    # Configure the model with your desired parameters
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    # Generate a response
    response = model.generate_content(prompt, generation_config=generation_config)

    return response.text

def run_gemini(doc, relation_type, gemini_api_key, query, threshold):
    # Extract entity pairs using the helper functions
    entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    entity_pairs = extract_relations(doc, spanbert, relation_type, entities_of_interest, query, threshold)

    # If no entity pairs were found, return an empty list
    if not entity_pairs:
        print("No entity pairs found.")
        return []

    # Convert the entity pairs back to plain text
    sentences = [" ".join(pair) for pair in entity_pairs]

    # Run Gemini on the sentences
    relations = []
    for sentence in sentences:
        try:
            response_text = get_gemini_completion(sentence, 'gemini-pro', 100, 0.2, 1, 32, gemini_api_key)
            relations.append(response_text)
        except Exception as e:
            pass

    return relations

# function that finds the corresponding relation to input int
def relation(relation_type):
    if relation_type == 1:
        return "Schools_Attended"
    elif relation_type == 2:
        return "Work_For"
    elif relation_type == 3:
        return "Live_In"
    elif relation_type == 4:
        return "Top_Member_Employees"
    else:
        raise ValueError("Invalid relation_type. Must be 1, 2, 3, or 4.")


def main(method, api_key, engine_id, gemini, relation_type, threshold, query, k_tuples):
    processed_urls = set()
    processed_queries = set()
    extracted_relations = []
    i = 0

    while True:
        print("\nParameters:")
        print(f"Client key  = {api_key}")
        print(f"Engine key  = {engine_id}")
        print(f"Gemini key  = {gemini}")
        print(f"Method  = {method}")
        print(f"Relation    = {relation(relation_type)}")
        print(f"Threshold   = {threshold}")
        print(f"Query       = {query}")
        print(f"# of Tuples = {k_tuples}")

        print("Loading necessary libraries; This should take a minute or so ...")
        print(f"=========== Iteration: {i} - Query: {query} ===========")

        i = i + 1
        urls = google_search(api_key, engine_id, query)
        if not urls:
            print("No URLs found from Google Custom Search Engine.")
            break

        for idx, url in enumerate(urls, start=1):
            if url in processed_urls:  # If the URL has been processed, skip it
                continue
            processed_urls.add(url)  # Mark the URL as processed

            webpage_content = retrieve_webpage(url)
            if not webpage_content:
                continue

            doc = spacy_function(webpage_content)
            if method == "spanbert":
                new_relations = run_spanbert(doc, spanbert, relation(relation_type), threshold)
            elif method == "gemini":
                new_relations = run_gemini(doc, relation(relation_type), gemini, threshold)
            else:
                print("Unknown method.")
                return

            # Update the set of extracted relations with new ones
            for rel in new_relations:
                if rel not in extracted_relations:
                    extracted_relations.append(rel)
            
            # Check if we have reached the desired number of tuples
            if len(extracted_relations) >= k_tuples:
                break

        if len(extracted_relations) >= k_tuples:
            break
        
        # If no new relations were added in this iteration, or if we've reached the desired number of tuples, stop
        if not new_relations or len(extracted_relations) >= k_tuples:
            print("No new relations found or desired number of tuples reached.")
            break
        
        # Update the query with a tuple not yet used for querying
        query = update_query_with_new_tuple(extracted_relations, processed_queries)

def update_query_with_new_tuple(extracted_relations, processed_queries):
    # Iterate over the extracted relations
    for rel in extracted_relations:
        # Convert the relation to a string
        rel_str = ' '.join([str(elem) for elem in rel])
        # If this relation hasn't been used as a query yet, return it
        if rel_str not in processed_queries:
            return rel_str
    # If all relations have been used as queries, return None
    return None

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


    main(method, api_key, engine_id, gemini, relation_type, threshold, query, k_tuples)

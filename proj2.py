import sys
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.tokens import Span
from spanbert import SpanBERT
from spacy_help_functions import extract_relations
import google.generativeai as genai
import re

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
        response = requests.get(url, timeout=10)

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

    except requests.RequestException as e:
        print(f"Error retrieving webpage: {e}")
    return None

# spacy implementation
def spacy_function(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    return doc


# Function for running SpanBERT model
def run_spanbert(doc, spanbert, relation_type, threshold, query):
    # Define the entities of interest for each relation type
    relation_entities_spanbert = {
        "Schools_Attended": ["PERSON", "ORGANIZATION"],
        "Work_For": ["PERSON","ORGANIZATION"],
        #"Live_In": ["PERSON", "LOCATION", "CITY", "COUNTRY", "STATE_OR_PROVINCE"],
        "Live_In": ["PERSON", "CITY"],
        "Top_Member_Employees": ["ORGANIZATION", "PERSON"]
    }

    # Extract using the helper functions and SpanBERT model
    entities_of_interest = [relation_entities_spanbert[relation_type]]
    relations = extract_relations(doc, spanbert, relation_type, entities_of_interest, threshold)

    # print(f"Extracted Relations: {relations}")

    # Post-process the relations based on the query
    post_processed_relations = []
    for (subj, relation, obj), confidence in relations.items():
        post_processed_relations.append((subj, obj, relation, confidence))
    # print(f"Post Processed Relations: {post_processed_relations}")

    return post_processed_relations, len(list(doc.sents))

# gemini helper
def get_gemini_completion(prompt, model_name, max_tokens, temperature, top_p, top_k, gemini):
    # Initialize a generative model
    genai.configure(api_key=gemini)
    model = genai.GenerativeModel(model_name)

    # Configure the model with your desired parameters
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    # Generate a response
    response = model.generate_content(prompt, generation_config=generation_config)

    return response

# Function for running gemini
def run_gemini(doc, relation_type, gemini, query):
    genai.configure(api_key=gemini)

    relation_entities = {
        "Schools_Attended": {"Subject": "PERSON", "Object": "ORG"},
        "Work_For": {"Subject": "PERSON", "Object": "ORG"},
        "Live_In": {"Subject": "PERSON", "Object": ["LOC", "GPE", "FAC"]},
        "Top_Member_Employees": {"Subject": "ORG", "Object": "PERSON"}
    }

    relation_descriptions = {
    "Schools_Attended": {"Subject": "PERSON", "Object": "ORGANIZATION"},
    "Work_For": {"Subject": "PERSON", "Object": "ORGANIZATION"},
    "Live_In": {"Subject": "PERSON", "Object": ["LOCATION", "CITY", "COUNTRY", "STATE_OR_PROVINCE"]},
    "Top_Member_Employees": {"Subject": "ORGANIZATION", "Object": "PERSON"}
    }

    relation_of_interest = get_relation(relation_type)
    entities_of_interest = relation_entities[relation_of_interest]

    subject_type = entities_of_interest["Subject"]
    object_types = entities_of_interest["Object"]
    object_types = object_types if isinstance(object_types, list) else [object_types]

    sentences = [sent.text for sent in doc.sents if 
                any(ent.label_ == subject_type for ent in sent.ents) and 
                any(ent.label_ == object_type for ent in sent.ents for object_type in object_types)]

    relations = []
    for sentence in sentences:
        # print(f"Subject Type: {subject_type}")
        # print(f"Object Type: {object_type}")

        prompt_text = f"In the sentence: '{sentence}', identify a 'Subject' of type '{subject_type}' and an 'Object' of type '{object_types}' that have a '{relation_of_interest}' relationship similar to the one in the query '{query}'. If such a relationship exists, present your findings in the format 'Subject: <subject>, Object: <object>, Relation: <relation>'. If no such relationship exists, state 'No matching relationship found.'"
        response = get_gemini_completion(prompt_text, 'gemini-pro', 100, 0.2, 1, 32, gemini)
        # print(f"Sentence: {sentence}")
        # print(f"Extracted Response: {response}")

        if not response._result.candidates:
            continue

        if not response._result.candidates[0].content.parts:
            continue

        response_text = response._result.candidates[0].content.parts[0].text

        entity1_match = re.search(r"Subject: ([^,]+),", response_text)
        entity2_match = re.search(r"Object: ([^,]+),", response_text)
        relationship_match = re.search(r"Relation: (.+)", response_text)

        if entity1_match and entity2_match and relationship_match:
            entity1 = entity1_match.group(1)
            entity2 = entity2_match.group(1)
            relationship = relationship_match.group(1)

            relations.append((entity1.strip(), entity2.strip(), relationship.strip(), 1.0))

            # Print the sentence and the extracted relation
            print("\n\t\t=== Extracted Relation ===")
            print(f"\t\tSentence:   {sentence}")
            print(f"\t\tSubject: {entity1.strip()} ; Object: {entity2.strip()} ; Confidence: 1.0")
            print(f"\t\tAdding to set of extracted relations")
            print("\t\t==========")
                        
            # print(f"Processed Relations: {relations}")

    return relations, len(list(doc.sents))


# function that finds the corresponding relation to input int
def get_relation(relation_type):

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

def remove_duplicates(relations):
    unique_relations = set()
    seen_relations = set()

    for relation in relations:
        relation_without_confidence = relation[:3]
        if relation_without_confidence not in seen_relations:
            unique_relations.add(relation)
            seen_relations.add(relation_without_confidence)

    return unique_relations

def main(method, api_key, engine_id, gemini, relation_type, threshold, query, k_tuples):
    processed_urls = set()
    processed_queries = set()
    relations = set()
    i = 0

    while True:
        print("\nParameters:")
        print(f"Client key  = {api_key}")
        print(f"Engine key  = {engine_id}")
        print(f"Gemini key  = {gemini}")
        print(f"Method  = {method}")
        print(f"Relation    = {get_relation(relation_type)}")
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
                print(f"URL ({idx} / 10): {url} has already been processed. Skipping...")
                continue
            
            processed_urls.add(url)  # Mark the URL as processed

            print(f"URL ({idx} / 10): {url}")
            print("\tFetching text from url ...")

            webpage_content = retrieve_webpage(url)
            if not webpage_content:
                continue
            else:
                webpage_length = len(webpage_content)
                print(f"\tWebpage length (num characters): {webpage_length}")
                print("\tAnnotating the webpage using spacy...")

                doc = spacy_function(webpage_content)
                sentences = list(doc.sents)
                print(f"\tExtracted {len(sentences)} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")

                if method == "spanbert":
                    new_relations, num_sentences = run_spanbert(doc, spanbert, get_relation(relation_type), threshold, query)
                    relations.update(new_relations)
                    relations = remove_duplicates(relations)

                elif method == "gemini":
                    new_relations, num_sentences = run_gemini(doc, relation_type, gemini, query)
                    relations.update(new_relations)
                     
                else:
                    print("Unknown method.")
                    return
                
                print(f"\tProcessed {len(sentences)} / {num_sentences} sentences")
                print(f"\tExtracted annotations for {len(new_relations)} out of total {num_sentences} sentences")
                print(f"\tRelations extracted from this website: {len(new_relations)} (Overall: {len(relations)})")
        
        # If we've reached the desired number of tuples, stop
        if len(relations) >= k_tuples:
            print("Desired number of tuples reached.")
            break

        print(f"\tExtracted annotations for {len(new_relations)} out of total {num_sentences} sentences")
        print(f"\tRelations extracted from this website: {len(new_relations)} (Overall: {len(relations)})")
        
        # Update the query with a tuple not yet used for querying
        query = update_query_with_new_tuple(relations, processed_queries, method)

    # After the while loop
    print(f"================== ALL RELATIONS for {get_relation(relation_type)} ({k_tuples}) =================")
    relations = list(relations)
    relations.sort(key=lambda x: x[3], reverse=True)  # Sort the relations by confidence score

    # Only iterate over the first k_tuples elements
    for relation in relations[:k_tuples]:
        if method == "spanbert":
            print(f"Confidence: {relation[3]} \t\t| Subject: {relation[0]} \t\t| Object: {relation[1]}")
        if method == "gemini":
            print(f"Subject: {relation[0]} \t\t| Object: {relation[1]}")

    print(f"Total # of iterations = {i}")

def update_query_with_new_tuple(extracted_relations, processed_queries, method):
    # If -spanbert is specified, sort the relations by extraction confidence
    if method == "spanbert":
        extracted_relations = sorted(extracted_relations, key=lambda x: x[1], reverse=True)

    # Iterate over the extracted relations
    for rel in extracted_relations:
        # Convert the relation to a string
        rel_str = ' '.join([str(elem) for elem in rel[:-2]])
        print(rel_str)
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

'''
COMS E6111 - Project 1
Sharmista Shastry (ss6950) and Cindy Ruan (cxr2000)
'''

import sys
import requests
from rake_nltk import Rake
import nltk
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import math

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def google_search(api_key, engine_id, query):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={engine_id}&q={query}"
    response = requests.get(url)
    data = response.json()
    if 'items' in data:
        # Return top 10 results
        return [item for item in data['items']][:10] 
    else:
        return []

def calculate_precision(results):
    # Get top 10 results
    top_10_results = results[:10]

    # Calculate the number of relevant results
    num_relevant = sum(result['relevant'] for result in top_10_results)

    return num_relevant / 10.0

def display_results(results):
    print("======================")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"URL: {result.get('link', 'N/A')}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Summary: {result.get('snippet', 'N/A')}")
        feedback = input("Is this result relevant? (Y/N): ")
        result['relevant'] = feedback.lower() == 'y'
        print()

def calculate_tf(document):
    terms = word_tokenize(document.lower())
    terms = [stemmer.stem(term) for term in terms if term not in stop_words and term.isalnum()] # Ensure terms are alphanumeric
    term_frequency = Counter(terms)
    
    return term_frequency

def calculate_idf(documents):
    num_documents = len(documents)

    document_frequency = Counter(term for document in documents for term in set(word_tokenize(document.lower())))
    inverse_document_frequency = {term: math.log(num_documents / (1 + frequency)) for term, frequency in document_frequency.items()} # Added 1 to avoid division by zero

    return inverse_document_frequency

def calculate_tfidf(document, idf):
    tf = calculate_tf(document)

    tfidf = {term: frequency * idf.get(term, 0) for term, frequency in tf.items()}
    return tfidf

def rocchio_expand_query(current_query, results, alpha=1.2, beta=0.80, gamma=0.15):
    query_terms = [stemmer.stem(term) for term in word_tokenize(current_query.lower()) if term not in stop_words and term.isalnum()] # Ensure terms are alphanumeric

    relevant_docs = [result for result in results if result.get('relevant', False)]
    non_relevant_docs = [result for result in results if not result.get('relevant', False)]

    all_docs = [' '.join(word_tokenize(result['title'].lower() + ' ' + result['snippet'].lower())) for result in results]
    idf = calculate_idf(all_docs)

    relevant_tfidf = Counter()

    for doc in relevant_docs:
        relevant_tfidf.update(calculate_tfidf(doc['title'] + ' ' + doc['snippet'], idf))

    non_relevant_tfidf = Counter()

    for doc in non_relevant_docs:
        non_relevant_tfidf.update(calculate_tfidf(doc['title'] + ' ' + doc['snippet'], idf))

    relevant_centroid = {term: (beta * freq / len(relevant_docs)) for term, freq in relevant_tfidf.items()}

    non_relevant_centroid = {term: (gamma * freq / len(non_relevant_docs)) for term, freq in non_relevant_tfidf.items()}

    adjusted_query = {term: alpha * query_terms.count(term) + relevant_centroid.get(term, 0) - non_relevant_centroid.get(term, 0)
                      for term in set(query_terms) | set(relevant_centroid) | set(non_relevant_centroid)}
    
    new_query_terms = sorted(adjusted_query, key=adjusted_query.get, reverse=True)[:5] # Limit to top 5 terms
    new_query = ' '.join(new_query_terms)

    return new_query

def main(api_key, engine_id, precision, query):
    print("Parameters:")
    print(f"Client key  = {api_key}")
    print(f"Engine key  = {engine_id}")
    print(f"Query       = {query}")
    print(f"Precision   = {precision}")

    while True:
        print("Google Search Results:")
        results = google_search(api_key, engine_id, query)
        if not results:
            print("No results. Exiting...")
            break
        display_results(results)
        precision_at_10 = calculate_precision(results)
        print("======================")
        print("FEEDBACK SUMMARY")
        print(f"Query: {query}")
        print(f"Precision: {precision_at_10}")

        if precision_at_10 >= precision:
            print("Desired precision reached, done")
            return 
        elif precision_at_10 == 0:
            print("No relevant results. Exiting...")
            return
        else:
            new_keywords = rocchio_expand_query(query, results)
            print("Still below the desired precision")
            print("Indexing results...")
            print("Augmenting by:", set(new_keywords.split()) - set(query.split()))
            query = new_keywords  # Update the query for the next iteration
            

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
    
    main(api_key, engine_id, precision, query)

    
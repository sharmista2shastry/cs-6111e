import sys
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import math

nltk.download('punkt', quiet=True)

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def google_search(api_key, engine_id, query):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={engine_id}&q={query}"
    response = requests.get(url)
    data = response.json()
    if 'items' in data:
        return [item for item in data['items']][:10] 
    else:
        return []

def calculate_precision(results):
    top_10_results = results[:10]
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
    terms = [stemmer.stem(term) for term in terms if term not in stop_words and term.isalnum()]
    term_frequency = Counter(terms)
    return term_frequency

def calculate_idf(documents):
    num_documents = len(documents)
    document_frequency = Counter(term for document in documents for term in set(word_tokenize(document.lower())))
    inverse_document_frequency = {term: math.log(num_documents / (1 + frequency)) for term, frequency in document_frequency.items()}
    return inverse_document_frequency

def calculate_tfidf(document, idf):
    tf = calculate_tf(document)
    tfidf = {term: frequency * idf.get(term, 0) for term, frequency in tf.items()}
    return tfidf

def rocchio_expand_query(current_query, results, alpha=1.2, beta=0.80, gamma=0.15):
    original_query_terms = set(current_query.split())
    query_terms = [stemmer.stem(term.lower()) for term in original_query_terms if term.lower() not in stop_words]

    relevant_docs = [result for result in results if result.get('relevant', False)]
    all_docs = [' '.join(word_tokenize(result['title'].lower() + ' ' + result['snippet'].lower())) for result in results]
    idf = calculate_idf(all_docs)

    relevant_tfidf = Counter()
    for doc in relevant_docs:
        relevant_tfidf.update(calculate_tfidf(doc['title'] + ' ' + doc['snippet'], idf))

    # Select top terms that are not already in the query
    new_terms = [term for term, score in relevant_tfidf.most_common() if term not in query_terms][:2]

    # Add at most 2 new terms to the original query
    updated_query = ' '.join(list(original_query_terms) + new_terms)
    return updated_query

def main(api_key, engine_id, precision, query):
    while True:
        print("Parameters:")
        print(f"Client key  = {api_key}")
        print(f"Engine key  = {engine_id}")
        print(f"Query       = {query}")
        print(f"Precision   = {precision}")

    
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
            print(f"Still below the precision of {precision}")
            print("Indexing results...")
            print("Augmenting by:")
            print("Below desired precision, but can no longer augment the query")
            return
        else:
            new_query = rocchio_expand_query(query, results)
            print(f"Still below the precision of {precision}")
            print("Indexing results...")
            print("Augmenting by:", set(new_query.split()) - set(query.split()))
            query = new_query  # Update the query for the next iteration
            
    

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py <google api key> <google engine id> <precision> <query>")
        sys.exit(1)

    api_key = sys.argv[1]
    engine_id = sys.argv[2]
    precision = float(sys.argv[3])
    if precision < 0 or precision > 1:
        print("Precision must be between 0 and 1")
        sys.exit(1)
    query = sys.argv[4]

    main(api_key, engine_id, precision, query)

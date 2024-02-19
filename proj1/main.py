import sys
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import math

# Download the Punkt tokenizer used by NLTK for word tokenization
nltk.download('punkt', quiet=True)

# Initialize stop words and stemmer
# Stop words are common words that are often removed during text processing
stop_words = set(stopwords.words('english'))
# The Porter stemmer is a simple, rule-based stemmer for English
stemmer = PorterStemmer()

# Function to perform a Google search using the provided API key, engine ID, and query
def google_search(api_key, engine_id, query):
    # Construct the URL for the Google Custom Search JSON API
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={engine_id}&q={query}"
    # Send a GET request to the API
    response = requests.get(url)
    # Parse the JSON response
    data = response.json()
    # If the response contains search results, return the top 10, otherwise return an empty list
    if 'items' in data:
        return [item for item in data['items']][:10] 
    else:
        return []

# Function to calculate the precision of the search results
def calculate_precision(results):
    # Select the top 10 results
    top_10_results = results[:10]
    # Count the number of relevant results
    num_relevant = sum(result['relevant'] for result in top_10_results)
    # Calculate and return the precision
    return num_relevant / 10.0

# Function to display the search results and ask the user for feedback
def display_results(results):
    print("======================")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"URL: {result.get('link', 'N/A')}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Summary: {result.get('snippet', 'N/A')}")
        # Ask the user if the result is relevant
        feedback = input("Is this result relevant? (Y/N): ")
        # Store the user's feedback in the 'relevant' field of the result
        result['relevant'] = feedback.lower() == 'y'
        print()

# Function to calculate term frequency (TF) in a document
def calculate_tf(document):
    # Tokenize the document into terms, convert to lower case
    terms = word_tokenize(document.lower())
    # Stem the terms and remove stop words and non-alphanumeric terms
    terms = [stemmer.stem(term) for term in terms if term not in stop_words and term.isalnum()]
    # Count the frequency of each term
    term_frequency = Counter(terms)
    # Return the term frequency
    return term_frequency

# Function to calculate inverse document frequency (IDF) across a set of documents
def calculate_idf(documents):
    # Count the number of documents
    num_documents = len(documents)
    # Count the frequency of each term across all documents
    document_frequency = Counter(term for document in documents for term in set(word_tokenize(document.lower())))
    # Calculate the inverse document frequency for each term
    inverse_document_frequency = {term: math.log(num_documents / (1 + frequency)) for term, frequency in document_frequency.items()}
    # Return the inverse document frequency
    return inverse_document_frequency

# Function to calculate term frequency-inverse document frequency (TF-IDF) for a document
def calculate_tfidf(document, idf):
    # Calculate the term frequency for the document
    tf = calculate_tf(document)
    # Calculate the TF-IDF for each term
    tfidf = {term: frequency * idf.get(term, 0) for term, frequency in tf.items()}
    # Return the TF-IDF
    return tfidf

def expand_query(current_query, results):
    # Split the current query into terms
    original_query_terms = set(current_query.split())
    # Stem the query terms and remove stop words
    query_terms = [stemmer.stem(term.lower()) for term in original_query_terms if term.lower() not in stop_words]

    # Select the relevant documents from the results
    relevant_docs = [result for result in results if result.get('relevant', False)]
    # Tokenize all documents into words, convert to lower case, and concatenate into a single string
    all_docs = [' '.join(word_tokenize((result.get('title', '').lower() + ' ' + result.get('snippet', '').lower()))) for result in results]
    # Calculate the inverse document frequency for all documents
    idf = calculate_idf(all_docs)

    # Calculate the TF-IDF for the relevant documents
    relevant_tfidf = Counter()
    for doc in relevant_docs:
        relevant_tfidf.update(calculate_tfidf(doc['title'] + ' ' + doc['snippet'], idf))

    # Select the top 2 terms that are not already in the query based on their TF-IDF scores
    new_terms = [term for term, score in relevant_tfidf.most_common() if term not in query_terms][:2]
    
    # Combine the original query terms and the new terms
    combined_terms = list(original_query_terms) + new_terms
    # Sort the combined terms based on their relevance (TF-IDF score)
    combined_terms_sorted = sorted(combined_terms, key=lambda term: relevant_tfidf.get(stemmer.stem(term.lower()), 0), reverse=True)

    # Construct the updated query by combining the sorted terms
    updated_query = ' '.join(combined_terms_sorted)
    
    # Return the updated query
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
            new_query = expand_query(query, results)
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

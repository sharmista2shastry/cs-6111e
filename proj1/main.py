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
nltk.download('stopwords', quiet=True)

# Initialize stop words and stemmer
# Stop words are common words that are often removed during text processing
nltk.download('stopwords', quiet=True)
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

def expand_query(current_query, results, alpha=1, beta=0.75, gamma=0.15):
    # Tokenize the current query into terms, remove stop words, and stem
    query_terms = [word.lower() for word in word_tokenize(current_query) if word.lower() not in stop_words and word.isalnum()]
    stemmed_query_terms = set([stemmer.stem(word) for word in query_terms])
    
    # Aggregate content from relevant and non-relevant documents
    relevant_docs_text = ' '.join([result['title'] + ' ' + result['snippet'] for result in results if result.get('relevant', False)])
    non_relevant_docs_text = ' '.join([result['title'] + ' ' + result['snippet'] for result in results if not result.get('relevant', False) and 'title' in result and 'snippet' in result])
    
    # Calculate TF-IDF vectors for the current query, relevant documents, and non-relevant documents
    all_docs = [relevant_docs_text, non_relevant_docs_text] + [current_query]
    idf = calculate_idf(all_docs)
    query_vector = calculate_tfidf(current_query, idf)
    relevant_vector = calculate_tfidf(relevant_docs_text, idf)
    non_relevant_vector = calculate_tfidf(non_relevant_docs_text, idf)
    
    # Apply the Rocchio algorithm to adjust the query vector
    adjusted_query_vector = {term: (alpha * query_vector.get(term, 0)) +
                                     (beta * relevant_vector.get(term, 0)) -
                                     (gamma * non_relevant_vector.get(term, 0))
                             for term in set(query_vector) | set(relevant_vector) | set(non_relevant_vector)}
    
    # Rank terms by their score in the adjusted query vector, excluding existing query terms
    new_terms = sorted([(term, score) for term, score in adjusted_query_vector.items() if term not in stemmed_query_terms],
                       key=lambda x: x[1], reverse=True)
    
    # Pick up to 2 new terms to add to the query
    added_terms = [term for term, score in new_terms[:2]]
    
    # Combine the original query terms with the new terms
    combined_query = query_terms + added_terms
    
    # Re-rank combined query terms based on their score in the adjusted query vector
    combined_query_sorted = sorted(combined_query, key=lambda term: adjusted_query_vector.get(stemmer.stem(term), 0), reverse=True)
    
    # Convert the stem terms back to their original form, this step requires a mapping from stemmed terms back to original terms which is not implemented here
    # For simplicity, we're using the combined and sorted terms directly
    updated_query = ' '.join(combined_query_sorted)
    
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
        if not results or len(results) < 10:
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

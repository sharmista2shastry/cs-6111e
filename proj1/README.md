## COMS 6111E
## Sharmista Shastry (ss6950) and Cindy Ruan (cxr2000)
## Information Retrieval System with Relevance Feedback and Query Reformulation

## Submitted Files

- proj1.tar.gz
    - main.py
    - requirements.txt
- README.md
- transcript.txt

## Running the Program

To run the program, follow these steps:

1. Navigate to the "proj1" directory where you will find a file named "requirements.txt" housing all necessary packages. Install these packages by executing the following command:

pip install -r requirements.txt

2. Execute the command below to retrieve the Google search results for your specified query:

python main.py <google api key> <google engine id> <precision> <query>

3. During the process, you will be prompted to mark relevant pages to enhance the query for optimal results.

## Internal Design

This Python script is designed to perform a Google search, display the results, ask the user for feedback on the relevance of the results, and refine the search query based on the feedback. The script uses the Rocchio algorithm for query refinement.

Here are the main high-level components of the script:

- Google Search Function: The google_search function performs a Google search using the provided API key, engine ID, and query, and returns the top 10 results.

- Precision Calculation Function: The calculate_precision function calculates the precision of the search results based on user feedback.

- Results Display and Feedback Function: The display_results function displays the search results and asks the user for feedback on their relevance.

- Text Processing Functions: The calculate_tf, calculate_idf, and calculate_tfidf functions perform text processing tasks such as term frequency calculation, inverse document frequency calculation, and TF-IDF calculation.

- Query Expansion Function: The expand_query function uses the Rocchio algorithm to refine the search query based on user feedback.

- Main Function: The main function orchestrates the entire process. It performs a Google search, displays the results, collects user feedback, calculates the precision, and refines the query until the desired precision is reached or no more refinement is possible.

The script uses the following external libraries:

- sys: This standard Python library is used to access command-line arguments.

- requests: This library is used to send HTTP requests to the Google Custom Search JSON API.

- nltk: The Natural Language Toolkit is used for text processing tasks such as tokenization, stop word removal, and stemming.

- collections: This standard Python library is used for its Counter class, which is a dictionary subclass for counting hashable objects.

- math: This standard Python library is used for mathematical operations such as logarithm calculation.

## Query Modification Method

The query modification method in this script is implemented in the expand_query function. It uses the Rocchio algorithm, a classic method in information retrieval for query refinement. The Rocchio algorithm adjusts the original query vector by moving it closer to the centroid of relevant documents and away from the centroid of non-relevant documents.

Here's a brief description of the expand_query function:

- Parse the current query: The function first tokenizes the current query into terms, removes stop words, and applies stemming. The result is a set of query terms.

- Aggregate content from relevant and non-relevant documents: The function aggregates the title and snippet of relevant and non-relevant documents separately.

- Calculate TF-IDF vectors: The function calculates the TF-IDF (Term Frequency-Inverse Document Frequency) vectors for the current query, relevant documents, and non-relevant documents. TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection or corpus.

- Apply the Rocchio algorithm: The function applies the Rocchio algorithm to adjust the query vector. The adjusted query vector is a weighted sum of the original query vector, the vector of relevant documents, and the vector of non-relevant documents. The weights are given by the parameters alpha, beta, and gamma, which can be adjusted to control the influence of each component.

- Rank terms: The function ranks terms by their score in the adjusted query vector, excluding the terms that are already in the original query.

- Add new terms to the query: The function picks up to 2 new terms with the highest scores to add to the query.

- Combine and re-rank query terms: The function combines the original query terms with the new terms, and re-ranks them based on their score in the adjusted query vector to ensure that we have the best word order.

- Update the query: The function updates the query with the combined and re-ranked terms.

The order of the query words in each round is determined by their score in the adjusted query vector. The terms with higher scores are placed earlier in the query. This is done in the step where the combined query terms are re-ranked based on their score in the adjusted query vector.

## Google Custom Search Engine JSON API Key and Engine ID

- API key: AIzaSyBd3QkH6ICMU5WxD6-MrmyqC8wNTJbj_SQ
- Engine ID: 151ebbf4130c44a63
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

This Python script is designed to perform a Google search using a provided API key and engine ID, display the results, ask the user for feedback on the relevance of the results, and then use this feedback to refine the search query and repeat the process. The goal is to improve the precision of the search results over multiple iterations.

The main high-level components of the script are:

1. Google Search Function (google_search): This function sends a GET request to the Google Custom Search JSON API and returns the top 10 search results.

2. Precision Calculation Function (calculate_precision): This function calculates the precision of the search results based on the user's feedback.

3. Results Display and Feedback Function (display_results): This function displays the search results and asks the user for feedback on their relevance.

4. Term Frequency Calculation Function (calculate_tf): This function calculates the term frequency for a given document.

5. Inverse Document Frequency Calculation Function (calculate_idf): This function calculates the inverse document frequency across a set of documents.

6. TF-IDF Calculation Function (calculate_tfidf): This function calculates the term frequency-inverse document frequency (TF-IDF) for a given document.

7. Query Expansion Function (expand_query): This function expands the search query using the relevant documents from the previous iteration.

8. Main Function (main): This function performs the search and query expansion process in a loop until the desired precision is reached or no more results are found.

The script uses the following external libraries:

1. sys: This standard Python library is used to access the command-line arguments.

2. requests: This library is used to send HTTP requests to the Google Custom Search JSON API.

3. nltk: The Natural Language Toolkit (NLTK) is used for word tokenization, stop word removal, and stemming. The script uses the Punkt tokenizer for word tokenization, the English stop words list, and the Porter stemmer.

4. collections: This standard Python library is used for its Counter class, which is a dictionary subclass for counting hashable objects.

5. math: This standard Python library is used for its log function, which is used in the calculation of inverse document frequency.

## Query Modification Method

The query modification method is implemented in the expand_query function. This function goes through a process of selecting and adding new keywords to the original query to improve search results. The goal is to select the most relevant keywords from the documents that were marked as relevant in the previous iteration.

Here's a detailed description of how the expand_query function works:

1. Tokenization and Stemming: The function first tokenizes the original query into individual terms, converts them to lower case, and stems them using the Porter stemmer. This is done to normalize the terms for comparison with the terms in the documents.

2. Relevant Documents Selection: The function then selects the documents that were marked as relevant in the previous iteration. This is based on the assumption that these documents contain the most relevant terms for the query.

3. Document Tokenization: All documents, including both relevant and non-relevant ones, are tokenized into words, converted to lower case, and concatenated into a single string. This is done to prepare the documents for the calculation of inverse document frequency (IDF).

4. IDF Calculation: The function calculates the IDF for all documents. IDF is a measure of how much information a given word provides, i.e., if it's common or rare across all documents. The IDF is used later to calculate the term frequency-inverse document frequency (TF-IDF).

5. TF-IDF Calculation: The function calculates the TF-IDF for the relevant documents. TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection or corpus. It's the product of two statistics, term frequency (TF) and IDF.

6. New Terms Selection: The function selects the top 2 terms that are not already in the query based on their TF-IDF scores. These are the terms that are most relevant to the query and are added to the original query terms.

7. Query Terms Sorting: The function sorts the combined terms (original query terms and new terms) based on their relevance, which is determined by their TF-IDF scores. The terms with higher scores are considered more relevant and are placed earlier in the query.

8. Updated Query Construction: Finally, the function constructs the updated query by combining the sorted terms. This updated query is used for the next iteration of the search.

This query modification method is designed to improve the precision of the search results over multiple iterations by continuously refining the query based on user feedback.

## Google Custom Search Engine JSON API Key and Engine ID

- API key: AIzaSyBd3QkH6ICMU5WxD6-MrmyqC8wNTJbj_SQ
- Engine ID: 151ebbf4130c44a63
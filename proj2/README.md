## COMS 6111E
## Sharmista Shastry (ss6950) and Cindy Ruan (cxr2000)
## Implementation of SpanBERT and Google Gemini

## Submitted Files

- proj2.tar.gz
    - proj2.py
    - requirements_proj2.txt
    - spacy_help_functions.py
- README.md
- transcript.txt

## Running the Program

To run the program, follow these steps:

1. If necessary, you should install Python 3.9 and create a Python 3.9 virtual environment to develop and test your code, as follows:
    First, make sure that you're using a fresh directory for the project, with no other virtual environments currently activated.
    Install Python 3.9, by running:
   ```
    sudo apt update
    sudo apt install python3.9
    sudo apt install python3.9-venv
   ```
    Create a new virtual environment named dbproj:
   ``` python3.9 -m venv dbproj ```
    To ensure the correct installation of Python 3.9, run:
    ```source dbproj/bin/activate
    python --version``` . This command should return ‘Python 3.9.5’.
    Also, when using the commands apt or apt-get below, you may get an error that says "ModuleNotFoundError: No module named 'apt_pkg'." In this case, then please perform the following steps:
   ```
    cd /usr/lib/python3/dist-packages
    sudo ln -s apt_pkg.cpython-36m-x86_64-linux-gnu.so apt_pkg.so
    cd ~
    sudo pip3 install --upgrade google-api-python-client
   ```
    Finally, note that some of the commands below may generate warnings (not errors) when you run them. As long as these are labeled “Warnings” (and not “Errors”), you can feel free to ignore them.

3. Your program will rely on:
    The Google Custom Search API. In your new dbproj virtual environment, run:
    pip3 install --upgrade google-api-python-client
    The Beautiful Soup toolkit:
    pip3 install beautifulsoup4
    spaCy:
    sudo apt-get update
    pip3 install -U pip setuptools wheel
    pip3 install -U spacy
    python3 -m spacy download en_core_web_lg
    We have implemented the scripts for downloading and running the pre-trained SpanBERT classifier for the purpose of this project:
    git clone https://github.com/larakaracasu/SpanBERT
    cd SpanBERT
    pip3 install -r requirements.txt
    bash download_finetuned.sh
    The Google Gemini API to extract the above relations from text documents by exploiting large language models, as a state-of-the-art alternative to SpanBERT:
    pip install -q -U google-generativeai

4. Unzip the proj2.tar.gz file by running:
    tar -xvzf proj2.tar.gz

3. Move proj2.py, spacy_help_functions.py, and requirements_proj2.txt to the SpanBERT folder. If there is already a spacy_help_functions.py file in the SpanBERT folder, replace it with the one provided in the proj2 zip.

4. Navigate to the "SpanBERT" directory where you will now house the file named "requirements_proj2.txt" housing all necessary packages. Install these packages by executing the following command:
    pip install -r requirements_proj2.txt

5. Execute the command below within the SpanBERT folder to retrieve the Google search results for your specified query:
    python3 proj2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>

## Internal Design

This python script is designed to perform a Google search and display the user results. The user can choose to use either SpanBERT or Google Gemini to extract relations from the URLs. The script will then display the relations extracted from the text documents. The script will also display the top k search results from the Google search. The user can specify the minimum number of relations to extract from the text documents, the confidence of the relations, and the query to search for.

Main high-level components:

- Google search function: searches for the top 10 results of a given user query.

- Webpage retrieval function: retrieves the plain text of a given url using BeautifulSoup.

- spaCy function: processes plain text using spaCy to return a structured implementation.

- SpanBERT function: extracts relations from the structured implementation using SpanBERT.

- Gemini completion function: helper function for Gemini, retrieves the response to a given prompt.

- Gemini function: extracts relations from the structured implementation using Google Gemini.

- Get relation function: correlates input integer to a relation type.

- Remove duplicates function: only gets unique relations from the list of relations.

- Main function: calls the Google search function, retrieves the top 10 search results, and extracts relations from the text documents using SpanBERT or Google Gemini based on user input.

- Update query function: updates the query to include the relations extracted from the text documents that haven't been used yet.

The script uses the following external libraries:

- sys: This standard Python library is used to access command-line arguments.

- requests: This library is used to send HTTP requests to the Google Custom Search JSON API.

- BeautifulSoup: This library is used to parse HTML documents and extract data from webpages.

- spaCy: This Python NLP library is used for processing and analyzing text data for tokenization.

- spanbert: SpanBERT is a pre-trained model for NLP tasks, specifically used here for extracting relations from webpages.

- spacy_help_functions: This file is imported to use its extract_relations function for the SpanBERT implementation.

- genai: This library is used interact with Google's Gemini AI model to extract relations.

- re: This library provides support for regular expressions in Python.

## Parsing web content

- retrieve_webpage: This function uses the requests and BeautifulSoup libraries to clean the webpage text and shorten the length if needed. 

- spacy_function: this function uses spaCy to parse the text returned from retrieve_webpage.

## SpanBERT method

The run_spanbert function is designed to extract relations from a parsed webpage using the pretrained SpanBERT model.

Here is a more detailed overview of the function:

- Definition of Entity Types: The function defines a dictionary, relation_entities_spanbert, that maps each relation type to a list of entity types relevant to that relation type.

- Mapping of Relation Types: Another dictionary, internal_relations_map, maps each relation type to its corresponding SpanBERT model relation.

- Extraction of Relations: The function calls the extract_relations function to extract relations from the document using SpanBERT. It provides the document, SpanBERT model, entity types of interest, and a threshold for relation extraction.

- Post-processing of Relations: Extracted relations are post-processed to filter out irrelevant relations. This involves removing prefixes from relation names and checking if they match any internal relations specified for the given relation type.

- Return Values: The function returns the post-processed relations along with the number of sentences in the document.

## Google Gemini Model

The run_gemini function is designed to extract relations from a parsed webpage using Google Gemini.

Here is a more detailed overview of the function:

- Configuration: The function configures the Gemini API with the provided API key.

- Entity Types: The function defines a dictionary mapping each relation type to the corresponding entity types involved in that relation.

- Extracting Relevant Sentences: The function identifies sentences in the document that contain entities relevant to the specified relation type.

- Prompt Generation: For each relevant sentence, the function generates a prompt instructing the Gemini model to identify subject and object entities and specify their types based on the relation type.

- Sending Prompts to Gemini: The function sends the generated prompts to the Gemini model and retrieves responses.

- Parsing Responses: The function parses the responses to extract the identified subject, object, and relationship type using regular expressions.

- Validation and Storage: The function validates the extracted entities against the expected types and stores valid relations along with their confidence scores.

- Output: The function then returns the extracted relations along with the total number of sentences processed in the document.

## KEYS

JSON API key: AIzaSyBd3QkH6ICMU5WxD6-MrmyqC8wNTJbj_SQ

Engine ID: 151ebbf4130c44a63















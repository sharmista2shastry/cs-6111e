# spaCy-SpanBERT: Relation Extraction from Web Documents

This repository integrates spaCy with pre-trained SpanBERT. It is a fork from [SpanBERT](https://github.com/facebookresearch/SpanBERT) by Facebook Research, which contains code and models for the paper: [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529).

We have adapted the SpanBERT scripts to support relation extraction from general documents beyond the TACRED dataset. We extract entities using spaCy and classify relations using SpanBERT. This code has been used for the purpose of the Advanced Database Systems Course at Columbia University.

## Install Requirements
Note: these instructions match the instructions on the class webpage. Feel free to follow those if more convenient.

Do the following within your CS6111 VM instance.

1. First, install Python 3.9:
```bash
sudo apt update
sudo apt install python3.9
sudo apt install python3.9-venv
```

2. Then, create a virtual environment running Python 3.9:

```bash
python3.9 -m venv dbproj
```

3. To ensure correct installation of Python 3.9 within your virtual environment:
```bash
source dbproj/bin/activate
python --version
```
The above should return 'Python 3.9.5'

4. Within your new virtual environment, install requirements and download spacy's en_core_web_lg:
```bash
sudo apt-get update
pip3 install -U pip setuptools wheel
pip3 install -U spacy
python3 -m spacy download en_core_web_lg
```

## Download Pre-Trained SpanBERT (Fine-Tuned in TACRED)
SpanBERT has the same model configuration as [BERT](https://github.com/google-research/bert) but it differs in
both the masking scheme and the training objectives.

* Architecture: 24-layer, 1024-hidden, 16-heads, 340M parameters
* Fine-tuning Dataset: [TACRED](https://nlp.stanford.edu/projects/tacred/) ([42 relation types](https://github.com/gkaramanolakis/SpanBERT/blob/master/relations.txt))

5. To download the fine-tuned SpanBERT model run: 

```bash
git clone https://github.com/larakaracasu/SpanBERT
cd SpanBERT
pip3 install -r requirements.txt
bash download_finetuned.sh
```

## Run Spacy-SpanBERT 
The code below shows how to extract relations between entities of interest from raw text: 

```python
raw_text = "Bill Gates stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."

entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]

# Load spacy model
import spacy
nlp = spacy.load("en_core_web_lg")  

# Apply spacy model to raw text (to split to sentences, tokenize, extract entities etc.)
doc = nlp(raw_text)  

# Load pre-trained SpanBERT model
from spanbert import SpanBERT 
spanbert = SpanBERT("./pretrained_spanbert")  

# Extract relations
from spacy_help_functions import extract_relations
relations = extract_relations(doc, spanbert, entities_of_interest)
print("Relations: {}".format(dict(relations)))
# Relations: {('Bill Gates', 'per:employee_of', 'Microsoft'): 1.0, ('Microsoft', 'org:top_members/employees', 'Bill Gates'): 0.992, ('Satya Nadella', 'per:employee_of', 'Microsoft'): 0.9844}
```

You can directly run this example via the example_relations.py file.

## Directly Apply SpanBERT (without using spaCy)

```python
from spanbert import SpanBERT
bert = SpanBERT(pretrained_dir="./pretrained_spanbert")
```
Input is a list of dicts, where each dict contains the sentence tokens ('tokens'), the subject entity information ('subj'), and object entity information ('obj'). Entity information is provided as a tuple: (\<Entity Name\>, \<Entity Type\>, (\<Start Location\>, \<End Location\>))

```python
examples = [
        {'tokens': ['Bill', 'Gates', 'stepped', 'down', 'as', 'chairman', 'of', 'Microsoft'], 'subj': ('Bill Gates', 'PERSON', (0,1)), "obj": ('Microsoft', 'ORGANIZATION', (7,7))},
        {'tokens': ['Bill', 'Gates', 'stepped', 'down', 'as', 'chairman', 'of', 'Microsoft'], 'subj': ('Microsoft', 'ORGANIZATION', (7,7)), 'obj': ('Bill Gates', 'PERSON', (0,1))},
        {'tokens': ['Zuckerberg', 'began', 'classes', 'at', 'Harvard', 'in', '2002'], 'subj': ('Zuckerberg', 'PERSON', (0,0)), 'obj': ('Harvard', 'ORGANIZATION', (4,4))}
        ]
preds = bert.predict(examples)
```

Output is a list of the same length as the input list, which contains the SpanBERT predictions and confidence scores

```python
print("Output: ", preds)
# Output: [('per:employee_of', 0.99), ('org:top_members/employees', 0.98), ('per:schools_attended', 0.98)]
```

## Contact
If you have any questions, please contact Lara Karacasu `<lk2859@columbia.edu>` (CS6111 TA).

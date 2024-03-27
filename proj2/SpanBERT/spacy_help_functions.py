import spacy
from collections import defaultdict

spacy2bert = { 
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "GPE": "LOCATION", 
        "LOC": "LOCATION",
        "DATE": "DATE"
        }

bert2spacy = {
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "CITY": "GPE",
        "COUNTRY": "GPE",
        "STATE_OR_PROVINCE": "GPE",
        "DATE": "DATE"
        }


def get_entities(sentence, entities_of_interest):
    return [(e.text, spacy2bert[e.label_]) for e in sentence.ents if e.label_ in spacy2bert]

valid_entity_types = {
    'Schools_Attended': ('PERSON', 'ORGANIZATION'),
    'Work_For': ('PERSON', 'ORGANIZATION'),
    'Live_In': ('PERSON', ['LOCATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY']),
    'Top_Member_Employees': ('ORGANIZATION', 'PERSON'),
}

def extract_relations(doc, spanbert, relation_type, entities_of_interest=None, query=None, conf=0.5):
    num_sentences = len([s for s in doc.sents])
    res = defaultdict(int)
    query = set(query.lower().split()) if query else None

    for i, sentence in enumerate(doc.sents, start=1):
        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        examples = []
        
        for ep in entity_pairs:
            subj_type = ep[1][1]
            obj_type = ep[2][1]
            valid_subj_type, valid_obj_types = valid_entity_types[relation_type]
            if subj_type != valid_subj_type or obj_type not in valid_obj_types:
                continue

            # Check if the entity pair is relevant to the query
            subj = ep[1][0].lower()
            obj = ep[2][0].lower()

            if query and (subj not in query or obj not in query):
                continue

            examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})

        if not examples:
            continue

        preds = spanbert.predict(examples)

        for ex, pred in list(zip(examples, preds)):
            relation = pred[0]
            if relation == 'no_relation':
                continue
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]
            if confidence > conf:
                print("\n\t\t=== Extracted Relation ===")
                print(f"\t\tSentence: {' '.join(ex['tokens'])}")
                print(f"\t\tSubject: {subj} ; Object: {obj} ;")
                print("\tAdding to set of extracted relations")
                print("\t\t==========")
                if res[(subj, relation, obj)] < confidence:
                    res[(subj, relation, obj)] = confidence

        if i % 5 == 0:
            print(f"\tProcessed {i} / {num_sentences} sentences")

    print()
    print(f"\tExtracted annotations for {len(res)} out of total {num_sentences} sentences")
    print(f"\tRelations extracted from this website: {len(res)} (Overall: {len(res)})")
    print()
    
    return res


def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    '''
    Input: a spaCy Sentence object and a list of entities of interest
    Output: list of extracted entity pairs: (text, entity1, entity2)
    '''
    if entities_of_interest is not None:
        entities_of_interest = {bert2spacy[b] for b in entities_of_interest if b in bert2spacy}
    ents = sents_doc.ents # get entities for given sentence

    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if entities_of_interest is not None and e1.label_ not in entities_of_interest:
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if entities_of_interest is not None and e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                continue

            if (1 <= (e2.start - e1.end) <= window_size):

                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0

                # Find end of sentence
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc

                if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start-gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start-gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs


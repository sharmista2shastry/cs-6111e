def relation(relation):
    r = ''
    if relation == 1:
        r = "Schools_Attended"
    elif relation == 2:
        r = "Work_For"
    elif relation == 3:
        r = "Live_In"
    else:
        r = "Top_Member_Employees"
    return r


def main(method, api_key, engine_id, gemini, relation, threshold, query, k_tuples):
    while True:
        print("Parameters:")
        print(f"Client key  = {api_key}")
        print(f"Engine key  = {engine_id}")
        print(f"Gemini key  = {gemini}")
        print(f"Method  = {method}")
        print(f"Relation    = {relation(relation)}")
        print(f"Threshold   = {threshold}")
        print(f"Query       = {query}")
        print(f"# of Tuples = {k_tuples}")

        print("Loading necessary libraries; This should take a minute or so ...")


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python proj2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>")
        sys.exit(1)

    method = sys.argv[1][1:]
    api_key = sys.argv[2]
    engine_id = sys.argv[3]
    gemini = sys.argv[4]
    relation = int(sys.argv[5])

    if relation < 1 or relation > 4:
        print("Relation must be 1, 2, 3, or 4")
        sys.exit(1)

    threshold = float(sys.argv[6])

    if threshold < 0 or threshold > 1:
        print("Threshold must be between 0 and 1")
        sys.exit(1)

    query = sys.argv[7]
    k_tuples = int(sys.argv[8])


    main(method, api_key, engine_id, gemini, relation, threshold, query, k_tuples)

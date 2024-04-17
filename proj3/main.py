import pandas as pd
import sys
from itertools import combinations

# Function to one-hot encode the dataset
def one_hot_encode(df):
    # Create a new dataframe for the one-hot encoding
    ohe_df = pd.get_dummies(df.select_dtypes(include='object'))
    return ohe_df

# Function to discretize the concentration into categories
def discretize_concentration(concentration):
    if concentration <= 3:
        return 'Low Concentration'
    elif concentration <= 10:
        return 'Medium Concentration'
    else:
        return 'High Concentration'

# Function to generate the candidate itemsets of size k
def apriori_gen(Lk_minus_1, k):
    Ck = set([i.union(j) for i in Lk_minus_1 for j in Lk_minus_1 if len(i.union(j)) == k])
    Ck_pruned = set()
    for itemset in Ck:
        subsets = set(frozenset([item]) for item in itemset)
        if subsets.issubset(Lk_minus_1):
            Ck_pruned.add(itemset)
    return Ck_pruned

# Function to calculate the support of itemsets in the DataFrame
def calculate_support(df, itemsets):
    support_data = {}
    for itemset in itemsets:
        support_data[itemset] = df[list(itemset)].all(axis=1).mean()
    return support_data

def scan_D(df, Ck, min_support):
    # Calculate the support for each candidate itemset
    support_data = {itemset: df[list(itemset)].all(axis=1).mean() for itemset in Ck}
    # Filter out itemsets with support less than min_support
    Lk = [itemset for itemset, support in support_data.items() if support >= min_support]
    return Lk, support_data

def generate_rules(L, support_data, min_confidence):
    rules = []
    for large_itemset in L:
        for itemset in large_itemset:
            subsets = list(combinations(itemset, len(itemset) - 1))
            for antecedent in subsets:
                antecedent = frozenset(antecedent)
                consequent = itemset.difference(antecedent)
                if antecedent in support_data and itemset in support_data:
                    ant_support = support_data[antecedent]
                    itemset_support = support_data[itemset]
                    if isinstance(ant_support, pd.Series):
                        ant_support = ant_support.mean()
                    if ant_support > 0:
                        confidence = itemset_support / ant_support
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, confidence, itemset_support))
    return rules

# Main Apriori algorithm function
def apriori(df, min_support, min_confidence):
    # Generate L1 and initial support data
    L1 = set(frozenset([item]) for item in df.columns if df[item].mean() >= min_support)
    Lk = L1
    L = []
    support_data = {item: df[list(item)].mean() for item in L1}
    while Lk:
        L.append(Lk)
        Ck = apriori_gen(Lk, k=len(list(Lk)[0]) + 1)
        Lk, new_support_data = scan_D(df, Ck, min_support)  # Scan dataset
        support_data.update(new_support_data)  # Update support data with the support of the new candidates
    # Generate high-confidence rules
    rules = generate_rules(L, support_data, min_confidence)
    return L, support_data, rules

def main():
    # Parse command line arguments
    filename = sys.argv[1]
    min_sup = float(sys.argv[2])
    min_conf = float(sys.argv[3])

    # Load the dataset
    df = pd.read_csv(filename)

    # List of values to drop
    values_to_drop = ['UNKNOWN OR NOT STATED']

    # Drop rows that contain any of the values in `values_to_drop`
    df = df[~df.isin(values_to_drop).any(axis=1)]

    # print("Number of rows after dropping: ", df.shape[0])

    # Discretize the 'CONCENTRATION' column and one-hot encode the dataframe
    df['CONCENTRATION_CATEGORY'] = df['CONCENTRATION'].apply(discretize_concentration)
    ohe_df = one_hot_encode(df)

    # Combine the one-hot encoded dataframe with the discretized concentration
    ohe_df = pd.concat([ohe_df, pd.get_dummies(df['CONCENTRATION_CATEGORY'])], axis=1)

    # Apply the apriori algorithm
    frequent_itemsets, support_data, rules = apriori(ohe_df, min_sup, min_conf)

    # Output results to a file
    with open('output.txt', 'w') as f:
        f.write("Frequent Itemsets:\n")
        for itemset in frequent_itemsets:
            for it in itemset:
                f.write("[{}] => {:.4%}\n".format(', '.join(it), support_data[it].mean()))
        f.write("\nHigh-Confidence Rules:\n")
        for rule in sorted(rules, key=lambda x: -x[2]):
            f.write("[{}] => [{}] (Conf: {:.4%}, Supp: {:.4%})\n".format(
                ', '.join(rule[0]),
                ', '.join(rule[1]),
                rule[2],
                rule[3]))

if __name__ == '__main__':
    main()

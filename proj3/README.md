## COMS 6111E
## Sharmista Shastry (ss6950) and Cindy Ruan (cxr2000)
## Association Rules

## Submitted Files

- proj3.tar.gz
    - main.py
    - requirements.txt
- INTEGRATED_DATASET.csv
- example-run.txt
- README.md

## Running the Program

To run the program, follow these steps:

1. Unzip the proj3.tar.gz file by running:
   ```
    tar -xvzf proj3.tar.gz
   ```

2. Navigate to the proj3 directory and install all necessary packages with this command:
   ```
    pip install -r requirements.txt
   ```

3. Execute the command below:
   ```
    python3 proj3.py INTEGRATED_DATASET.csv <min_sup> <min_conf>
   ```
   min_sup is the minimum support and min_conf is the minimum confidence of the associations. Both should be entered as decimals between 0 and 1.

## Chosen Dataset

a) NYC Open Data dataset: Metal Content of Consumer Products Tested by the NYC Health Department

b) Our integrated dataset is essentially unchanged from the original dataset. However, we did change the dataset within the main.py code itself. 

    - Any rows with 'UNKNOWN OR NOT STATED' as a value were dropped.

    - The 'CONCENTRATION' row was modified to instead use levels of Low, Medium, and High Concentration.

        - Low is <= 3 ppm or MG/CM^2 of metal
        - Medium is 4 - 10 ppm or MG/CM^2 of metal
        - High is > 10 ppm or MG/CM^2 of metal

c) We chose this dataset because finding dangerous metals like lead and mercury in our food and household products can be very dangerous for our health. We want to know which products are more likely to have higher concentrations and where they may be coming from. Realistically however, no amount of lead or mercury in any product is safe no matter the concentration.

## Apriori Algorithm

1) Finding Frequent Itemsets: The algorithm iteratively generates candidate itemsets of increasing size based on the previous set of frequent itemsets. It then scans the dataset to count the support of each candidate itemset. Itemsets are kept if they meet the minimum support.

2) Generating Association Rules: For each frequent itemset, the algorithm generates association rules by exploring all possible combinations of values. The confidence of each rule is calculated based on the support of the itemset and its subsets. Rules that meet the minimum confidence are kept.

## Functions

- one_hot_encode: Function to one-hot encode categorical variables in the dataset.

- discretize_concentration: Function to discretize the concentration variable into categories.

- apriori_gen: Function to generate candidate itemsets of size k from the previous set of frequent itemsets.

- calculate_support: Function to calculate the support of itemsets in the DataFrame.

- scan_D: Function to scan the dataset and filter out itemsets with support less than the minimum support threshold.

- generate_rules: Function to generate association rules from frequent itemsets based on the minimum confidence threshold.

- apriori: Main Apriori algorithm function that executes the process of finding frequent itemsets and generating association rules.

- main: Function to parse command line arguments, load the dataset, apply the Apriori algorithm, and output the results to a file.

## Example Run

```
python3 main.py INTEGRATED_DATASET.csv 0.05 0.8
```

There is some miscellaneous information that is irrelevant, but there are some very interesting things to note.

There is a 7.43% support for high concentrations of lead in medications/supplements, which is incredibly dangerous as a necessity for people to take when they need it. India also appears to be a country that exports the highest percentage of high concentration dangerous metal products, with a support of 7.43% as well. Lead seems to be the most abundant in the products, with a support of ~66%.

Lead seems to especially be shown in candy and toy products, with a confidence of 100% and supports of 5.39% and 5.49% respectively. This is incredibly dangerous for kids, who these products typically will cater to. Spice also has a 99.5% confidence and 22.14% support for lead. 

In general, the dataset seems to have many associations for countries like India, Bangladesh, Mexico, China, and the USA, which is perhaps not surprising considering these are the countries that tend to export the most products to the USA.

Of course, most rows had low concentration with a support of 83%, but as mentioned any amount of lead, mercury, arsenic, etc. in products is very dangerous. The mentioned products are food, medicine, toys, and cosmetics.



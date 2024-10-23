# Store_recommendation_system

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
data = pd.read_csv("store_data.csv", header=None)

#Convert data set rows into list. 
data_list = data.values.tolist()

#Handelling the NaN values
cleaned_list = [[item for item in transaction if pd.notna(item)] for transaction in data_list]

# Print the list
print(cleaned_list)
# Let's convert the dataset to a DataFrame with bool type
# Each column represents an item, and each row represents a transaction
# We'll one-hot encode the data
te = TransactionEncoder()
te_ary = te.fit(cleaned_list).transform(cleaned_list)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.head()
df.info()
# Count the frequency of items in all transactions and print top 20 frequently purchased items.

item_counts = df.sum()
sorted_items = item_counts.sort_values(ascending=False)
print("Items with the most number of 'true' occurrences:")
print(sorted_items.head(20))

# Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.013, use_colnames=True)

frequent_itemsets
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
# Task 1: Calculate frequent item set for iteration
frequent_itemsets_3 = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x)) == 3]
frequent_itemsets_3
# Step2. Find out the support and confidence

items = df.columns[:]

# Calculate support for the first 20 items
support = {}
total_transactions = len(df)

for item in items:
    item_support = df[item].sum() / total_transactions
    support[item] = item_support

# Sort support values in ascending order
sorted_support = sorted(support.items(), key=lambda x: x[1])

# Print the support values in ascending order
print("Support values in ascending order:")
for item, item_support in sorted_support:
    print(f"{item}: {item_support}")
# Step 3. How many relations you can derive
# Step 4. Generate Association Rules From Frequent Itemsets.

# Generate association rules from frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

# Print the number of derived relations
print("Number of derived relations:", len(rules))

# Print the generated association rules
print("Generated association rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])

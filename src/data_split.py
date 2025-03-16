"""
@Authors: Ilyes DJERFAF, Nazim KESKES

The following code is designed to create three dataframes: train, dev, and test. 
The train dataframe will contain 80% of the data, which corresponds to 329,879 samples. 
The dev dataframe is further split into dev1 and dev2, each containing 5% of the data, 
equivalent to 20,617 samples each. The test dataframe will contain 10% of the data, 
which corresponds to 41,234 samples. 

In each dataframe, the distribution of the labels ('neutral', 'entailment', and 'contradiction') 
is maintained to be equal, ensuring a balanced representation of 33.33% for each label.
"""

import pandas as pd
import time

# Start the timer
start = time.time()

# Load the dataframes
df_dev_matched = pd.read_json("data/multinli_1.0_dev_matched.jsonl", lines=True)
df_dev_mismatched = pd.read_json("data/multinli_1.0_dev_mismatched.jsonl", lines=True)
df_train = pd.read_json("data/multinli_1.0_train.jsonl", lines=True)

# Keep only the columns that are useful
cols_to_keep = ['gold_label', 'sentence1','sentence2']
df_dev_matched = df_dev_matched[cols_to_keep]
df_dev_mismatched = df_dev_mismatched[cols_to_keep]
df_train = df_train[cols_to_keep]

# Rename the columns
df_dev_matched.columns = ['label', 'text','assertion']
df_dev_mismatched.columns = ['label', 'text','assertion']
df_train.columns = ['label', 'text','assertion']

# Check the unique labels
print("*" * 50)
print("Check the unique labels")
print("\tUnique labels in dev_matched:", df_dev_matched["label"].unique())
print("\tUnique labels in dev_mismatched:", df_dev_mismatched["label"].unique())
print("\tUnique labels in train:", df_train["label"].unique())
print("\n")

# Drop the rows with '-' label
df_dev_matched = df_dev_matched[df_dev_matched['label']!='-']
df_dev_mismatched = df_dev_mismatched[df_dev_mismatched['label']!='-']

# Shape of the dataframes
size_df = df_dev_matched.shape[0] + df_dev_mismatched.shape[0] + df_train.shape[0]
print("*" * 50)
print("Shape of the dataframes")
print("\tShape of dev_matched dataframe:", df_dev_matched.shape)
print("\tShape of dev_mismatched dataframe:", df_dev_mismatched.shape)
print("\tShape of train dataframe:", df_train.shape)
print("\tTotal size of the dataframes:", size_df)
print("\n")

# Check the na values
print("*" * 50)
print("Check the na values")
print("\tNa values in dev_matched:")
print(df_dev_matched.isna().sum())
print("\n")
print("\tNa values in dev_mismatched:")
print(df_dev_mismatched.isna().sum())
print("\n")
print("\tNa values in train:")
print(df_train.isna().sum())
print("\n")

# Step 1 : shuffle the dataframes
df_dev_matched = df_dev_matched.sample(frac=1)
df_dev_mismatched = df_dev_mismatched.sample(frac=1)
df_train = df_train.sample(frac=1)

# Step 2 : split the dataframes into neutral, entailment and contradiction dataframes
neutral_df_train = df_train[df_train["label"] == "neutral"]
neutral_df_dev_matched = df_dev_matched[df_dev_matched["label"] == "neutral"]
neutral_df_dev_mismatched = df_dev_mismatched[df_dev_mismatched["label"] == "neutral"]
neutral_df = pd.concat([neutral_df_train, neutral_df_dev_matched, neutral_df_dev_mismatched])

entailment_df_train = df_train[df_train["label"] == "entailment"]
entailment_df_dev_matched = df_dev_matched[df_dev_matched["label"] == "entailment"]
entailment_df_dev_mismatched = df_dev_mismatched[df_dev_mismatched["label"] == "entailment"]
entailment_df = pd.concat([entailment_df_train, entailment_df_dev_matched, entailment_df_dev_mismatched])

contradiction_df_train = df_train[df_train["label"] == "contradiction"]
contradiction_df_dev_matched = df_dev_matched[df_dev_matched["label"] == "contradiction"]
contradiction_df_dev_mismatched = df_dev_mismatched[df_dev_mismatched["label"] == "contradiction"]
contradiction_df = pd.concat([contradiction_df_train, contradiction_df_dev_matched, contradiction_df_dev_mismatched])

# Step 3 : Reshuffle the dataframes
neutral_df = neutral_df.sample(frac=1)
entailment_df = entailment_df.sample(frac=1)
contradiction_df = contradiction_df.sample(frac=1)

# Step 4 : Check the size of each dataframe
print("*" * 50)
print("Check the size of each dataframe")
print("\tSize of neutral dataframe:", neutral_df.shape[0])
print("\tSize of entailment dataframe:", entailment_df.shape[0])
print("\tSize of contradiction dataframe:", contradiction_df.shape[0])
print("\n")

# Step 5 : Check the pourcentage of each dataframe size compared to the total size
print("*" * 50)
print("Check the pourcentage of each dataframe size compared to the total size")
print("\tPourcentage of neutral dataframe:", neutral_df.shape[0] / size_df)
print("\tPourcentage of entailment dataframe:", entailment_df.shape[0] / size_df)
print("\tPourcentage of contradiction dataframe:", contradiction_df.shape[0] / size_df)
print("\n")

# Step 6 : Split the dataframes into train, dev and test dataframes (train = 80%, dev = 10%, test = 10%) and (train = 80% of each label, dev1 = 5% of each label, dev2 = 5% of each label, test = 10% of each label)
neutral_train = neutral_df.iloc[:int(neutral_df.shape[0] * 0.8)]
neutral_dev1 = neutral_df.iloc[int(neutral_df.shape[0] * 0.8):int(neutral_df.shape[0] * 0.85)]
neutral_dev2 = neutral_df.iloc[int(neutral_df.shape[0] * 0.85):int(neutral_df.shape[0] * 0.9)]
neutral_test = neutral_df.iloc[int(neutral_df.shape[0] * 0.9):]

entailment_train = entailment_df.iloc[:int(entailment_df.shape[0] * 0.8)]
entailment_dev1 = entailment_df.iloc[int(entailment_df.shape[0] * 0.8):int(entailment_df.shape[0] * 0.85)]
entailment_dev2 = entailment_df.iloc[int(entailment_df.shape[0] * 0.85):int(entailment_df.shape[0] * 0.9)]
entailment_test = entailment_df.iloc[int(entailment_df.shape[0] * 0.9):]

contradiction_train = contradiction_df.iloc[:int(contradiction_df.shape[0] * 0.8)]
contradiction_dev1 = contradiction_df.iloc[int(contradiction_df.shape[0] * 0.8):int(contradiction_df.shape[0] * 0.85)]
contradiction_dev2 = contradiction_df.iloc[int(contradiction_df.shape[0] * 0.85):int(contradiction_df.shape[0] * 0.9)]
contradiction_test = contradiction_df.iloc[int(contradiction_df.shape[0] * 0.9):]

train_df = pd.concat([neutral_train, entailment_train, contradiction_train])
dev1_df = pd.concat([neutral_dev1, entailment_dev1, contradiction_dev1])
dev2_df = pd.concat([neutral_dev2, entailment_dev2, contradiction_dev2])
test_df = pd.concat([neutral_test, entailment_test, contradiction_test])

# Step 7 : Check the size of each dataframe
print("*" * 50)
print("Check the size of each dataframe")
print("\tSize of train dataframe:", train_df.shape[0])
print("\tSize of dev1 dataframe:", dev1_df.shape[0])
print("\tSize of dev2 dataframe:", dev2_df.shape[0])
print("\tSize of test dataframe:", test_df.shape[0])
print("\n")


# Step 8 : Check the pourcentage of each dataframe size compared to the total size
print("*" * 50)
print("Check the pourcentage of each dataframe size compared to the total size")
print("\tPourcentage of train dataframe:", train_df.shape[0] / size_df)
print("\tPourcentage of dev1 dataframe:", dev1_df.shape[0] / size_df)
print("\tPourcentage of dev2 dataframe:", dev2_df.shape[0] / size_df)
print("\tPourcentage of test dataframe:", test_df.shape[0] / size_df)
print("\n")

# check the distribution of labels in each dataframe
print("*" * 50)
print("Check the distribution of labels in each dataframe")
print("\tDistribution of labels in train:")
print(train_df["label"].value_counts(normalize=True))
print("\n")
print("\tDistribution of labels in dev1:")
print(dev1_df["label"].value_counts(normalize=True))
print("\n")
print("\tDistribution of labels in dev2:")
print(dev2_df["label"].value_counts(normalize=True))
print("\n")
print("\tDistribution of labels in test:")
print(test_df["label"].value_counts(normalize=True))

# Step 9 : Shuffle the dataframes
train_df = train_df.sample(frac=1)
dev1_df = dev1_df.sample(frac=1)
dev2_df = dev2_df.sample(frac=1)
test_df = test_df.sample(frac=1)

# Drop na values
train_df.dropna(inplace=True)
dev1_df.dropna(inplace=True)
dev2_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Step 10 : Save the dataframes
train_df.to_csv("data/train.csv", index=False)
dev1_df.to_csv("data/dev1.csv", index=False)
dev2_df.to_csv("data/dev2.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Dataframes saved successfully")

# Stop the timer
end = time.time()

print('*' * 50)
print('Data Memory Usage')
print(train_df.info())
print(dev1_df.info())
print(dev2_df.info())
print(test_df.info())
print('*' * 50)

print(f"Execution time: {end - start} seconds")

"""
CS 171 - Assignment 1 (Fall 2026)
Starter Code

The data generation and train/test split are done for you.
Fill in the sections marked TODO.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ─────────────────────────────────────────────
#  DATA GENERATION (do not modify)
# ─────────────────────────────────────────────
np.random.seed(0)

# Class 0 -- cluster centered around (2, 2)
X_class0 = np.random.randn(50, 2) * 1.25 + [2, 2]
y_class0 = np.zeros(50, dtype=int)

# Class 1 -- cluster centered around (4.4, 4.4)
X_class1 = np.random.randn(50, 2) * 1.25 + [4.4, 4.4]
y_class1 = np.ones(50, dtype=int)

# Combine into one dataset
X = np.vstack([X_class0, X_class1])
y = np.concatenate([y_class0, y_class1])

# ─────────────────────────────────────────────
#  TRAIN / TEST SPLIT (do not modify)
#
#  sklearn's train_test_split randomly shuffles
#  the data and splits it into two parts.
#  We use 70% for training and 30% for testing.
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=3
)

print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)
print(f"Total samples:    {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")
print(f"Features:         {X.shape[1]}")
print(f"Classes:          0 and 1")
print()


# ═══════════════════════════════════════════════════════════
#  STEP 2: Train a KNN model with k = 3 (15 points)
# ═══════════════════════════════════════════════════════════
print("=" * 50)
print("STEP 2: KNN with k = 3")
print("=" * 50)

# TODO: Create a KNeighborsClassifier with n_neighbors=3
#       and fit it on X_train, y_train.
#
trained_model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

# TODO: Use the model to predict labels for X_test.
#
predictions3 = trained_model.predict(X_test)
# print(predictions3) # the return type is nd.array

# Print predicted vs true (this code is provided -- just uncomment it
# after you have 'predictions3' ready):
#


print()


# ═══════════════════════════════════════════════════════════
#  STEP 3: Count the mistakes (10 points)
# ═══════════════════════════════════════════════════════════
print("=" * 50)
print("STEP 3: Counting mistakes for k = 3")
print("=" * 50)

# TODO: Count how many predictions were wrong.
#       Hint: (predictions3 != y_test) gives True for each mistake.
#             Use .sum() to count them.
#
incorrect_labels = (predictions3 != y_test).astype(int) # type nd.array
#incorrect_labels = [int(l) for l in labels]
print(f'Number of incorrect labels for k = 3: ' + str(sum(incorrect_labels)))

print()


# ═══════════════════════════════════════════════════════════
#  STEP 4: Repeat with k = 1 and k = 7 (15 points)
# ═══════════════════════════════════════════════════════════
print("=" * 50)
print("STEP 4: KNN with k = 1")
print("=" * 50)

# TODO: Create KNN with k=1, fit, predict, print predicted vs true,
#       count mistakes. Same steps as above.

predictions1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train).predict(X_test)
incorrect_labels = (predictions1 != y_test).astype(int) # type nd.array
print(f'Number of incorrect labels for k = 1: ' + str(sum(incorrect_labels)))

print()

print("=" * 50)
print("STEP 4: KNN with k = 7")
print("=" * 50)

# TODO: Create KNN with k=7, fit, predict, print predicted vs true,
#       count mistakes. Same steps as above.

predictions7 = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train).predict(X_test)
incorrect_labels = (predictions7 != y_test).astype(int) # type nd.array
print(f'Number of incorrect labels for k = 7: ' + str(sum(incorrect_labels)))

print()


# ═══════════════════════════════════════════════════════════
#  STEP 5: Observations (10 points)
# ═══════════════════════════════════════════════════════════
print("=" * 50)
print("STEP 5: Observations")
print("=" * 50)

# TODO: Answer the following as print statements or comments.
#
# (a) Which value of k (1, 3, or 7) made the fewest mistakes
#     on the test set?

# ANSWER: k = 7 made the fewest number of mistakes (2 mistakes) compared to k = 1 and k = 3

# (b) When k = 1, the model gets every training example correct
#     by definition. Does it also make the fewest mistakes on
#     the test set? What does this tell you?

# ANSWER: No, k = 1 model doesn't make the fewest number of mistakes on the test set. In fact, it makes the largest number of mistakes, ie 5.
# This tells us that for this particular situation given the training data, the more neighbors we take into consideration, the better/more accurate are our results.

# (c) In 1-2 sentences, what is the trade-off in choosing
#     a small k vs a large k?

# ANSWER: Small k values: Results are less accurate, but computation is faster.
# Large k value: Results are more accurate, however, computation is slower. Choosing large values for k could also lead to overfitting?

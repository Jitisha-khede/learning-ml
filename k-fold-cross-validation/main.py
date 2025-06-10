from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 1. Load dataset
data = load_iris()
X = data.data
y = data.target

# 2. Create model
model = DecisionTreeClassifier()

# 3. Set up K-Fold (e.g., 5 folds)
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 4. Evaluate using cross_val_score
scores = cross_val_score(model, X, y, cv=kf)

# 5. Print results
print(f"Cross-validation scores: {scores}")
print(f"Average accuracy: {np.mean(scores):.2f}")

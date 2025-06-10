import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

# Simulated dataset of 20 samples
n_samples = 20
X = np.arange(n_samples)

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Plot
fig, ax = plt.subplots()
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    y = np.full(n_samples, np.nan)
    y[test_index] = fold  # Assign fold number to test data
    
    ax.scatter(X, y, label=f'Fold {fold+1}', s=100)

ax.set_yticks(np.arange(5))
ax.set_yticklabels([f'Fold {i+1}' for i in range(5)])
ax.set_xlabel("Sample Index")
ax.set_title("K-Fold Cross Validation Split Visualization")
ax.legend()
plt.show()

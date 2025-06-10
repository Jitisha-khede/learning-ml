from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Visualize
plt.figure(figsize=(15, 10))
plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    class_names=iris.target_names,
    rounded=True
)
plt.title("Decision Tree Visualization")
plt.show()
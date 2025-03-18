import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset from CSV
df = pd.read_csv("iris.csv")  # Update with your file path if needed

# Encode species column (if it's categorical)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])


# Split into features (X) and target (y)
X = df.drop(columns=['species'])
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

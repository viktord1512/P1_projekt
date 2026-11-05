import numpy as np # til numeriske operationer og array-manipulationer
import pandas as pd # til håndtering af datasæt som DataFrame
import seaborn as sns # til udregning af heatmaps, fx confusion matrix
import matplotlib.pyplot as plt # til visualisering af diagrammer, plots og confusion matrixer
from sklearn.ensemble import RandomForestClassifier # til Random Forest-klassifikation
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV # til opdeling af datasæt i trænings- og testdata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder # til one-hot encoding og standardisering af features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # til evaluering af modelens præstation og beregning af confusion matrix
import time


# lav en timer der viser tiden det tager at køre koden
start_time = time.time()

# Indlæs heart-datasættet
heart = pd.read_csv("heart.csv")

# Onehot encoding for kategoriske features
# nonnumeric_features = [1, 2, 6, 8, 10]
# numeric_features = [0, 3, 4, 5, 7, 9]

nonnumeric_features = [1, 2, 8]
numeric_features = [0, 3, 4, 5]

encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(heart.iloc[:, nonnumeric_features]).toarray()

X_numerical = heart.iloc[:, numeric_features].values

# Lægger de nye one-hot encoded kategoriske features sammen med de numeriske features
X = np.hstack((X_numerical, X_categorical))
y = heart.iloc[:, -1].values

# Opdel data i træn- og testdata (80% træning, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # 0.2 svarer til 20% af dataene som testes
)

# Pipeline for Random Forest med StandardScaler og SimpleImputer
Rf_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy="median", add_indicator=True)),
    ('rbs', StandardScaler()),
    ('rf', RandomForestClassifier())
])

params = [{
        'rf__n_estimators': list(range(100, 1001, 100)),
        'rf__min_samples_leaf': list(range(1, 21, 1)),
        'rf__min_samples_split': list(range(2, 41, 1)),
        "rf__criterion": ["gini", "entropy"]
}]

# RandomizedSearchCV for at finde de bedste hyperparametre for Random Forest
grid_search = RandomizedSearchCV(Rf_pipe, params, cv=5, scoring='accuracy', n_jobs=-1, n_iter=10)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.5f}")

# Evaluering på testdata med de bedste parametre
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Test accuracy with best parameters: {accuracy_best:.5f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred_best)}")
print(f"Classification report:\n{classification_report(y_test, y_pred_best)}")



class_report = classification_report(y_test, y_pred_best, output_dict=True)

# Confusion matrix med visualization
cm = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix:")
print(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Koden der skal måle tiden går her (hele scriptet)
end_time = time.time()
print(f"Tiden det tog at køre koden: {end_time - start_time:.2f} sekunder")

# https://medium.com/@Doug-Creates/plotting-scikit-learn-classification-report-for-analysis-0229447fe232
def plot_classification_report_with_support(report):
    labels = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    metrics = ['precision', 'recall', 'f1-score', 'support']
    data = np.array([[report[label][metric] for metric in metrics] for label in labels])
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.matshow(data, cmap='coolwarm')
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(cax)
    # Adding the text
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.title('Classification Report with Support')
    plt.show()

plot_classification_report_with_support(class_report)
from time import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt # til visualisering af diagrammer, plots og confusion matrixer
import seaborn as sns # til udregning af heatmaps, fx confusion matrix
import pandas as pd # til håndtering af datasæt som DataFrame, hvor dataframes gør det nemt at manipulere og analysere data
import numpy as np # til numeriske operationer og array-manipulationer, hvor arrays bruges til at gemme og manipulere data effektivt
from sklearn.neighbors import KNeighborsClassifier # til kNN-klassifikation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA # hjælper med at visualisere vores 11 dimensioner til 2D
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

# lav en timer der viser tiden det tager at køre koden
start_time = time()

# Indlæs heart-datasættet
heart = pd.read_csv(Path(__file__).resolve().parent / "heart.csv")

# Tager alle nuller i cholesterol (og i resting BP, men der er legit kun et sted hvor der er 0) og laver dem om til NaN'er.
# de skal være NaN og ikke 0 fordi 0-værdier kan faktisk være legit i nogle kolonner (som FastingBS og OldPeak), og dem vil vi ikke impute på
heart["Cholesterol"] = heart["Cholesterol"].replace(0,np.nan)
heart["RestingBP"] = heart["RestingBP"].replace(0,np.nan)

# nonnumeric_features = [1, 2, 6, 8, 10]
# numeric_features = [0, 3, 4, 5, 7, 9]

nonnumeric_features = [1, 2, 8]
numeric_features = [0, 3, 4, 5]



#onehot encoding for kategoriske features
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(heart.iloc[:, nonnumeric_features]).toarray()
X_numerical = heart.iloc[:, numeric_features].values



# Gemmer vores matrix, som er en liste af en liste
X = np.hstack((X_numerical, X_categorical))
y = heart.iloc[:, -1].values



# Opdel data i trænings- og testdata
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4,
    stratify=y
)
# https://www.geeksforgeeks.org/machine-learning/stratified-sampling-in-machine-learning/
# Vores data er rimelig lige, men stratify gør en lille forskel

# Pipeline for kNN med RobustScaler og GridSearchCV
knn_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy="median", add_indicator=True)),
    ("imputer", IterativeImputer()),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
last_k = 50
params = [{'knn__n_neighbors': list(range(1, last_k+1, 2)),
         'knn__weights': ['uniform', 'distance'],
         'knn__p': [1, 2],
         'scaler': [StandardScaler(), RobustScaler()],
         "imputer": [SimpleImputer(strategy="median", add_indicator=True),
                     IterativeImputer(initial_strategy="median", add_indicator=True,
                                      random_state=43)]
}]


# Udfør grid search for at finde de bedste hyperparametre for kNN
print("Starting grid search...")
grid_search = GridSearchCV(knn_pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)


# Evaluering på testdata med de bedste parametre
best_model = grid_search.best_estimator_

# PR-kurve for kNN
print("Valgte kNN-hyperparametre:", grid_search.best_params_)

score_traening = best_model.predict_proba(X_train)[:, 1]
score_test = best_model.predict_proba(X_test)[:, 1]

precision_traening, recall_traening, _ = precision_recall_curve(y_train, score_traening)
precision_test, recall_test, _ = precision_recall_curve(y_test, score_test)

ap_traening = average_precision_score(y_train, score_traening)
ap_test = average_precision_score(y_test, score_test)

plt.figure(figsize=(9, 6))
plt.step(recall_traening, precision_traening, where="post",
         linewidth=2, label=f"Træning (AP = {ap_traening:.3f})")
plt.step(recall_test, precision_test, where="post",
         linewidth=2, label=f"Test (AP = {ap_test:.3f})")
plt.xlabel("Recall (sensitivitet)")
plt.ylabel("Precision")
plt.title("PR-kurve for kNN - GridSearchCV")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.tight_layout()

billedmappe = Path.home() / "Downloads" / "P1_kurver"
billedmappe.mkdir(parents=True, exist_ok=True)
filnavn = f"PR_kNN_original_GridSearch_{datetime.now():%Y%m%d_%H%M%S}.png"
billedsti = billedmappe / filnavn
plt.savefig(str(billedsti), dpi=250, bbox_inches="tight")
print(f"PR-kurven er gemt: {billedsti}")
print(f"AP træning: {ap_traening:.3f}; AP test: {ap_test:.3f}")
plt.show()

y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
cm = confusion_matrix(y_test, y_pred_best)
print(f"Test accuracy with best parameters: {accuracy_best:.5f}")
print(f"Confusion matrix:\n{cm}")
print(f"Classification report:\n{classification_report(y_test, y_pred_best)}")


# Her tager vi og laver en liste af vores numeric_features' navne, så det samme på nonnumeric_features og så sætter vi vores indicator fra imputeren
# Beregn confusion matrix på træningsdata til visualiseringen
y_pred_train = best_model.predict(X_train)
cm = confusion_matrix(y_train, y_pred_train)

kolonne_navne = list(heart.columns[numeric_features]) + list(encoder.get_feature_names_out(heart.columns[nonnumeric_features])) + ["Cholesterol_missing"]
# print(kolonne_navne)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_percent = cm_normalized * 100


# Confusion matrix: Vis antal med navnene TN, FP, FN og TP
if cm.shape == (2, 2):
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])
    count_annotations = np.array([
        [f"{cell_names[i, j]}\n{cm[i, j]}" for j in range(2)]
        for i in range(2)
    ])
else:
    count_annotations = cm.astype(str)

# Vis begge confusion matrixer i samme vindue
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=count_annotations, fmt="", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title(f"Confusion Matrix (antal) - kNN, k={grid_search.best_params_['knn__n_neighbors']}")
axes[0].set_title(f"Confusion Matrix træning (antal) - kNN, k={grid_search.best_params_['knn__n_neighbors']}")

# Vis de samme værdier som procenter af hver faktisk klasse
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
            cbar_kws={"label": "%"}, ax=axes[1])
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].set_title(f"Confusion Matrix (procent) - kNN, k={grid_search.best_params_['knn__n_neighbors']}")
axes[1].set_title(f"Confusion Matrix træning (procent) - kNN, k={grid_search.best_params_['knn__n_neighbors']}")
plt.tight_layout()
plt.show()


# Hent de behandlede data fra pipeline'en
preprocessor = best_model[: -1]
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)


# Lav kolonnenavne dynamisk, så de matcher de behandlede data fra pipeline'en
base_names = (list(heart.columns[numeric_features])
              + list(encoder.get_feature_names_out(heart.columns[nonnumeric_features])))
missing_idx = best_model.named_steps['imputer'].indicator_.features_
kolonne_navne = base_names + [f"{base_names[i]}_missing" for i in missing_idx]


# Her tager vi vores two behandlet datasæt (x_train og x_test) som vi splittede tidligere for så at sætte dem sammen i en stor csv fil
train_data = pd.DataFrame(X_train_proc, columns=kolonne_navne)
train_data["HeartDisease"] = y_train
train_data["split"] = "train"

test_data = pd.DataFrame(X_test_proc, columns=kolonne_navne)
test_data["HeartDisease"] = y_test
test_data["split"] = "test"


# Her sætter vi dem sammen og exporter det som en csv fil
big_data = pd.concat([train_data,test_data], ignore_index=True)
big_data.to_csv("heart_behandlet_gridSearch.csv", index=False)


# print de bedste parametre og score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.5f}")
print(f"Køretid: {time() - start_time:.2f} sekunder")
cv_results = pd.DataFrame(
    grid_search.cv_results_)
cv_results.sort_values(by='mean_test_score', ascending=False, inplace=True)
print(cv_results)
print("Classification Report:")
class_report = classification_report(y_test, y_pred_best, output_dict=True)


# print classification report
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


# Confusion matrix: Vis antal med navnene TN, FP, FN og TP
if cm.shape == (2, 2):
    cell_names = np.array([["TN", "FP"], ["FN", "TP"]])
    count_annotations = np.array([
        [f"{cell_names[i, j]}\n{cm[i, j]}" for j in range(2)]
        for i in range(2)
    ])
else:
    count_annotations = cm.astype(str)
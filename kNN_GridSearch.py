from time import time
import matplotlib.pyplot as plt # til visualisering af diagrammer, plots og confusion matrixer
import seaborn as sns # til udregning af heatmaps, fx confusion matrix
import pandas as pd # til håndtering af datasæt som DataFrame, hvor dataframes gør det nemt at manipulere og analysere data
import numpy as np # til numeriske operationer og array-manipulationer, hvor arrays bruges til at gemme og manipulere data effektivt
from collections import OrderedDict # bedre dictionary
from sklearn.model_selection import train_test_split # til opdeling af datasæt i trænings- og testdata
from sklearn.neighbors import KNeighborsClassifier # til kNN-klassifikation
from sklearn.metrics import accuracy_score, roc_curve, auc # til beregning af nøjagtighed
from sklearn.decomposition import PCA # hjælper med at visualisere vores 11 dimensioner til 2D
from sklearn.metrics import confusion_matrix # til beregning af confusion matrix
from sklearn.preprocessing import OneHotEncoder # til one-hot encoding af kategoriske features
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# lav en timer der viser tiden det tager at køre koden
start_time = time()

# Indlæs heart-datasættet
heart = pd.read_csv("heart.csv")

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
    X, y, test_size=0.20, random_state=42,
    stratify=y
)
# https://www.geeksforgeeks.org/machine-learning/stratified-sampling-in-machine-learning/
# Vores data er rimelig lige, men stratify gør en lille forskel

# Pipeline for kNN med RobustScaler og GridSearchCV
knn_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy="median", add_indicator=True)),
    ('rbs', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
last_k = 51
params = [{'knn__n_neighbors': list(range(1, last_k+1, 2)),
         'knn__weights': ['uniform', 'distance'],
         'knn__p': [1, 2]}]

# Udfør grid search for at finde de bedste hyperparametre for kNN
print("Starting grid search...")
grid_search = GridSearchCV(knn_pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluering på testdata med de bedste parametre
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
cm = confusion_matrix(y_test, y_pred_best)
print(f"Test accuracy with best parameters: {accuracy_best:.5f}")
print(f"Confusion matrix:\n{cm}")
print(f"Classification report:\n{classification_report(y_test, y_pred_best)}")


# her tager vi og laver en liste af vores numeric_features' navne, så det samme på nonnumeric_features og så sætter vi vores indicator fra imputeren
kolonne_navne = list(heart.columns[numeric_features]) + list(encoder.get_feature_names_out(heart.columns[nonnumeric_features])) + ["Cholesterol_missing"]
# print(kolonne_navne)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# visualisering af confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
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


### VIRKER IKKE!!! Skal fikse på et eller andet tidspunkt idk
# from mlxtend.plotting import plot_decision_regions
# from sklearn.decomposition import PCA
# X_2d = PCA(n_components=2).fit_transform(X)
#
# # Træn en kNN på de 2D-data (kun til plottet)
# viz_model = KNeighborsClassifier(n_neighbors=last_k).fit(X_2d, y)
#
# plot_decision_regions(X=X_2d, y=y, clf=viz_model, legend=2)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title(f"KNN med k={last_k}")
# plt.show()

# TODO
# Pipeline?
# CVGrid
# plottet oven over


# # Funktion for at plotte ROC curve
# def plot_roc_curve(y_true, y_scores):
#     fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2,
#              label='kNN with RobustScaler (AUC = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve - kNN with RobustScaler')
#     plt.legend(loc="lower right")
#     plt.show()

# print de bedste parametre og score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.5f}")
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.sort_values(by='mean_test_score', ascending=False, inplace=True)
print(cv_results)

print(f"Køretid: {time() - start_time:.2f} sekunder")
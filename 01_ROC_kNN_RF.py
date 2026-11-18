from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)

# Indlæsning af dataene
mappe = Path(__file__).resolve().parent
fil = next((p for p in (mappe / "heart.csv", mappe / "heart(1).csv") if p.is_file()), None)
if fil is None:
    raise FileNotFoundError("heart.csv skal ligge i samme mappe som Python-filen.")
heart = pd.read_csv(fil)
heart["Cholesterol"] = heart["Cholesterol"].replace(0, np.nan)
heart["RestingBP"] = heart["RestingBP"].replace(0, np.nan)

# Valg af features
numeriske = [0, 3, 4, 5]
kategoriske = [1, 2, 8]
encoder = OneHotEncoder(handle_unknown="ignore")
X_numeriske = heart.iloc[:, numeriske].to_numpy()
X_kategoriske = encoder.fit_transform(heart.iloc[:, kategoriske]).toarray()
X = np.hstack([X_numeriske, X_kategoriske])
y = heart["HeartDisease"].to_numpy()

# Fælles opdeling af dataene
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Træning: {len(y_train)} | Test: {len(y_test)}", flush=True)

# Modelindstillinger fra den tidligere fælles optimering
# Disse parametre erstatter den langsomme GridSearch/RandomizedSearch.
# kNN: GridSearchCV gav k=79, distance, p=2, RobustScaler, IterativeImputer.
# RF: RandomizedSearchCV + GridSearchCV gav 400 træer, leaf=12, split=13, entropy.

def opret_knn():
    return Pipeline([
        ("imputer", IterativeImputer(initial_strategy="median", add_indicator=True,
                                      random_state=43)),
        ("scaler", RobustScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=79, weights="distance", p=2))
    ])

def opret_rf():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=12,
            min_samples_split=13,
            criterion="entropy",
            random_state=42
        ))
    ])

# Visning af billedet

def vis_billede(sti):
    print(f"PNG-fil gemt: {sti}", flush=True)
    if os.name == "nt":
        os.startfile(str(sti))  # Windows: åbning i standardbilledprogrammet
    else:
        plt.show()

# Træning af modellerne
knn = opret_knn()
rf = opret_rf()
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Beregning af sandsynlighederne
knn_test = knn.predict_proba(X_test)[:, 1]
rf_test = rf.predict_proba(X_test)[:, 1]

# ROC-kurve for kNN og Random Forest
plt.figure(figsize=(9, 6.5))
for navn, score in (("kNN", knn_test), ("Random Forest", rf_test)):
    fpr, tpr, _ = roc_curve(y_test, score)
    roc_auc = roc_auc_score(y_test, score)
    plt.plot(fpr, tpr, lw=2.2, label=f"{navn} (AUC = {roc_auc:.3f})")
    print(f"{navn}: test ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Tilfældig klassifikation")
plt.title("ROC-kurve – kNN og Random Forest (fælles testdata)")
plt.xlabel("Andel falsk positive (1 − specificitet)")
plt.ylabel("Sensitivitet")
plt.xlim(0, 1)
plt.ylim(0, 1.02)
plt.grid(alpha=0.25)
plt.legend(loc="lower right")
plt.tight_layout()
billede = mappe / "ROC_kNN_RF_rapport.png"
plt.savefig(billede, dpi=300, bbox_inches="tight")

# Testresultater
for navn, model in (("kNN", knn), ("Random Forest", rf)):
    print(f"\n{navn} – confusion matrix")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print(classification_report(y_test, model.predict(X_test), digits=3))

vis_billede(billede)

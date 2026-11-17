import numpy as np # til numeriske operationer og array-manipulationer
import pandas as pd # til håndtering af datasæt som DataFrame
import seaborn as sns # til udregning af heatmaps, fx confusion matrix
import matplotlib.pyplot as plt # til visualisering af diagrammer, plots og confusion matrixer
from sklearn.ensemble import RandomForestClassifier # til Random Forest-klassifikation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV # til opdeling af datasæt i trænings- og testdata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler # til one-hot encoding og standardisering af features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # til evaluering af modelens præstation og beregning af confusion matrix
import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

heart = pd.read_csv("heart.csv")

heart["Cholesterol"] = heart["Cholesterol"].replace(0,np.nan)
heart["RestingBP"] = heart["RestingBP"].replace(0,np.nan)

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
imputer = IterativeImputer(initial_strategy="median", add_indicator=True,
                                 random_state=43)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Define model
model = LogisticRegression(max_iter=1000)

# Forward selection
sfs = SequentialFeatureSelector(model, n_features_to_select=4, direction='forward', n_jobs=-1)
sfs.fit(X_train, y_train)
feature_names = (
    [heart.columns[i] for i in numeric_features]
    + list(encoder.get_feature_names_out(heart.columns[nonnumeric_features]))
)

# Add the missing-indicator names the imputer appended
feature_names += [
    f"missing_{feature_names[i]}" for i in imputer.indicator_.features_
]

feature_names = np.array(feature_names)

features = sfs.get_support()
print("Selected features:", list(feature_names[features]))

# # ændrer vores 0ler til not a number så imputeren kan se hvor den skal impute
# heart["Cholesterol"] = heart["Cholesterol"].replace(0,np.nan)
#
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer()
#
#
# heart.to_csv("heart_behandlet.csv", index=False, header=True)
# print((heart["Cholesterol"] == 0).sum())

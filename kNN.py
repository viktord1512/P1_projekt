import matplotlib.pyplot as plt # til visualisering af diagrammer, plots og confusion matrixer
import seaborn as sns # til udregning af heatmaps, fx confusion matrix
import pandas as pd # til håndtering af datasæt som DataFrame, hvor dataframes gør det nemt at manipulere og analysere data
import numpy as np # til numeriske operationer og array-manipulationer, hvor arrays bruges til at gemme og manipulere data effektivt
from collections import OrderedDict # bedre dictionary
from sklearn.model_selection import train_test_split # til opdeling af datasæt i trænings- og testdata
from sklearn.neighbors import KNeighborsClassifier # til kNN-klassifikation
from sklearn.metrics import accuracy_score # til beregning af nøjagtighed
from sklearn.decomposition import PCA # hjælper med at visualisere vores 11 dimensioner til 2D
from sklearn.metrics import confusion_matrix # til beregning af confusion matrix
from sklearn.preprocessing import OneHotEncoder # til one-hot encoding af kategoriske features



# Indlæs heart-datasættet
heart = pd.read_csv("heart.csv")

# Tager alle nuller i cholesterol (og i resting BP, men der er legit kun et sted hvor der er 0) og laver dem om til NaN'er.
# de skal være NaN og ikke 0 fordi 0-værdier kan faktisk være legit i nogle kolonner (som FastingBS og OldPeak), og dem vil vi ikke impute på
heart["Cholesterol"] = heart["Cholesterol"].replace(0,np.nan)
heart["RestingBP"] = heart["RestingBP"].replace(0,np.nan)

# til noget debugging kode jeg lavede, skal fjernes for CVgrid senere så bar ignore
results = OrderedDict()

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
    X, y, test_size=0.30, random_state=42,
    stratify=y
)
# https://www.geeksforgeeks.org/machine-learning/stratified-sampling-in-machine-learning/
# Vores data er rimelig lige, men stratify gør en lille forskel

# laver en imputer der går ud fra medianen af kolonnen (strategy="median) og laver en ny kolonne der siger om data er blevet imputed på den række (add_indicator=True)
# vigtig note om de nye kolonner den laver er at de bliver ikke automatisk sat ind, det sker længere nede
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median", add_indicator=True)

#impute, derefter scale
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# vi initializer 3 scalers her, men vi bruge kun en som er standard scaleren. Den giver de bedste resultater *** skal måske ændres til robust***
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
standard_scaler = StandardScaler()
norm_scaler = Normalizer()
min_max_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

# X_train = standard_scaler.fit_transform(X_train)
# X_test = standard_scaler.transform(X_test)
# X_train = norm_scaler.fit_transform(X_train)
# X_test = norm_scaler.transform(X_test)
# X_train = min_max_scaler.fit_transform(X_train)
# X_test = min_max_scaler.transform(X_test)
X_train = robust_scaler.fit_transform(X_train)
X_test = robust_scaler.transform(X_test)

# her tager vi og laver en liste af vores numeric_features' navne, så det samme på nonnumeric_features og så sætter vi vores indicator fra imputeren
kolonne_navne = list(heart.columns[numeric_features]) + list(encoder.get_feature_names_out(heart.columns[nonnumeric_features])) + ["Cholesterol_missing"]
# print(kolonne_navne)

last_k = 21
# Opret kNN-modellen og træn den for forskellige værdier af n_neighbors igennem et loop
# Bliver sikkert skiftet ud med CVgrid
for n_neighbors in range(1, last_k+1, 2):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Træn modellen
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)

    # Beregn nøjagtighed
    accuracy = accuracy_score(y_test, y_pred)
    results[n_neighbors] = accuracy
    print(f"K = {n_neighbors}: {accuracy:.5f}")


# Beregning af confusion matrix
cm = confusion_matrix(y_test, y_pred)
# visualisering i terminal
print("Confusion Matrix:")
print(cm)

# caller bare functions som sklearn har for at få vores precision og recall. Accuracy caller vi igen for shits and giggles
from sklearn.metrics import precision_score, recall_score
print(f"Precision: {precision_score(y_test,y_pred)*100:.5}%")
print(f"Recall: {recall_score(y_test, y_pred)*100:.5}%")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.5}%")

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# visualisering af confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# # For at teste k værdier. Viser kun den bedste med dens accuracy, hvis man vil have cm og precision/recall så skal man manuelt sætte k
sorted_results = dict(sorted(results.items(), key=lambda item:item[1]))
print(f"Best K = {next(reversed(sorted_results))}\n"
      f"Accuracy = {next(reversed(sorted_results.values()))}")

# her tager vi vores two behandlet datasæt (x_train og x_test) som vi splittede tidligere for så at sætte dem sammen i en stor csv fil
train_data = pd.DataFrame(X_train, columns=kolonne_navne)
train_data["HeartDisease"] = y_train
train_data["split"] = "train"

test_data = pd.DataFrame(X_test, columns=kolonne_navne)
test_data["HeartDisease"] = y_test
test_data["split"] = "test"

# her sætter vi dem sammen og exporter det som en csv fil
big_data = pd.concat([train_data,test_data], ignore_index=True)
big_data.to_csv("heart_behandlet.csv", index=False)
#hvilken idiot lavede index arguments anderledes i samme fucking library


### VIRKER IKKE!!! skal fikse på et eller andet tidspunkt idk
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

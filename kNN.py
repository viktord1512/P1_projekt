import matplotlib.pyplot as plt # til visualisering af diagrammer, plots og confusion matrixer
import seaborn as sns # til udregning af heatmaps, fx confusion matrix
import pandas as pd # til håndtering af datasæt som DataFrame, hvor dataframes gør det nemt at manipulere og analysere data
import numpy as np # til numeriske operationer og array-manipulationer, hvor arrays bruges til at gemme og manipulere data effektivt
from sklearn.model_selection import train_test_split # til opdeling af datasæt i trænings- og testdata
from sklearn.neighbors import KNeighborsClassifier # til kNN-klassifikation
from sklearn.metrics import accuracy_score # til beregning af nøjagtighed
from sklearn.decomposition import PCA # hjælper med at visualisere vorees 11 dimentioner til 2D
from sklearn.metrics import confusion_matrix # til beregning af confusion matrix
from sklearn.preprocessing import OneHotEncoder # til one-hot encoding af kategoriske features


# Indlæs heart-datasættet
heart = pd.read_csv("heart.csv")

# standardiser numeriske features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# nonnumeric_features = [1, 2, 6, 8, 10]
# numeric_features = [0, 3, 4, 5, 7, 9]

nonnumeric_features = [1, 8, 10]
numeric_features = [3, 5, 9]

#onehot encoding for kategoriske features
encoder = OneHotEncoder()
X_categorical = encoder.fit_transform(heart.iloc[:, nonnumeric_features]).toarray()
X_numerical = heart.iloc[:, numeric_features].values

# Gemmer vores matrix, som er en liste af en liste
X = np.hstack((X_numerical, X_categorical))
y = heart.iloc[:, -1].values

# Opdel data i trænings- og testdata
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


last_k = 25
# Opret kNN-modellen og træn den for forskellige værdier af n_neighbors igennem et loop
for n_neighbors in range(1, last_k+1, 2):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Træn modellen
    model.fit(X_train, y_train)

    # Forudsigelser på testdata
    y_pred = model.predict(X_test)

    # Beregn nøjagtighed
    accuracy = accuracy_score(y_test, y_pred)

    print(f"K = {n_neighbors}: {accuracy:.5f}")

# Beregning af confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
# visualisering i terminal
print(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# visualisering af confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
X_2d = PCA(n_components=2).fit_transform(X)

# Træn en kNN på de 2D-data (kun til plottet)
viz_model = KNeighborsClassifier(n_neighbors=last_k).fit(X_2d, y)

plot_decision_regions(X=X_2d, y=y, clf=viz_model, legend=2)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"KNN med k={last_k}")
plt.show()

# ny_patient = [[54,1,0,120,188,0,1,113,0,1.4,1,1,3]]
# resultat = model.predict(ny_patient)

# print("Forudsagt klasse:", resultat[0])

# # Klassificér en ny patient

# ny_patient = [[54,1,0,120,188,0,1,113,0,1.4,1,1,3]]
# resultat = model.predict(ny_patient)

# print("Forudsagt klasse:", resultat[0])

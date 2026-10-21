from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Indlæs Iris-datasættet
iris = load_iris()
X = iris.data
y = iris.target

# Opdel data i trænings- og testdata
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Opret kNN-modellen
model = KNeighborsClassifier(n_neighbors=3)

# Træn modellen
model.fit(X_train, y_train)

# Forudsigelser på testdata
y_pred = model.predict(X_test)

# Beregn nøjagtighed
accuracy = accuracy_score(y_test, y_pred)

print(f"Nøjagtighed: {accuracy:.2f}")

# Klassificér en ny blomst
ny_blomst = [[5.1, 3.5, 1.4, 0.2]]
resultat = model.predict(ny_blomst)

print("Forudsagt klasse:", iris.target_names[resultat[0]])
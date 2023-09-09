from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar CIFAR-10
cifar10 = fetch_openml("CIFAR_10_small")
X, y = cifar10.data, cifar10.target

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Crear modelo
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Evaluar modelo
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc}')

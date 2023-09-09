from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Cargar 20 newsgroups
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')
X_train, y_train = data_train.data, data_train.target
X_test, y_test = data_test.data, data_test.target

# Crear modelo
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc}')

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
classifier = model.fit(X, y)
joblib.dump(classifier, 'iris_model.pkl')


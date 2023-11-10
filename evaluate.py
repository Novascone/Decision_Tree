import decision_tree as dt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()


X = iris['data']

y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = dt.DecisionTree()
model.fit(X_train, y_train)
pred = model.new_predict(X_test)

print(accuracy_score(y_test, pred))


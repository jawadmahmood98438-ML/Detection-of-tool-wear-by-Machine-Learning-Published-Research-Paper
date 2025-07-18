from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def apply_l2_regularization(X_train, y_train):
    parameters = {'alpha': [1e-8, 1e-7, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]}
    model = Ridge()
    clf = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5, verbose=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf.best_estimator_
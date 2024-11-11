import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.base import clone

# Define the regression models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'SVR': SVR(),
    'KNeighbors Regressor': KNeighborsRegressor()
}

# Set up parameters for RandomizedSearchCV
param_grids = {
    'Linear Regression': {},
    'Ridge': {'alpha': [0.01, 0.1, 1, 10, 100]},
    'Lasso': {'alpha': [0.01, 0.1, 1, 10, 100]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'SVR': {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2, 0.5]},
    'KNeighbors Regressor': {'n_neighbors': [3, 5, 7, 9]}
}

# 1. Training and evaluating models with k-fold cross-validation
def evaluate_models(xtrain, ytrain):
    cross_val_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring='neg_mean_squared_error')
        cross_val_results[name] = -np.mean(scores)
        print(f'{name} MSE: {-np.mean(scores)}')
    return cross_val_results

# 2. Hyperparameter tuning function with RandomizedSearchCV
def tune_with_random_search(xtrain, ytrain, model_name, model, param_grid):
    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    search.fit(xtrain, ytrain)
    print(f"Best parameters for {model_name}: {search.best_params_}")
    print(f"Best cross-validation MSE for {model_name}: {-search.best_score_}")
    return search.best_estimator_

# 3. Hyperparameter tuning function with Hyperopt
def tune_with_hyperopt(xtrain, ytrain, model_name, model):
    def objective(params):
        model_clone = clone(model)
        model_clone.set_params(**params)
        score = -np.mean(cross_val_score(model_clone, xtrain, ytrain, cv=5, scoring='neg_mean_squared_error'))
        return {'loss': score, 'status': STATUS_OK}

    # Define parameter search space
    search_spaces = {
        'Ridge': {'alpha': hp.loguniform('alpha', np.log(0.01), np.log(100))},
        'Lasso': {'alpha': hp.loguniform('alpha', np.log(0.01), np.log(100))},
        'Random Forest': {'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)), 'max_depth': hp.choice('max_depth', [None, 10, 20, 30])},
        'Gradient Boosting': {'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)), 'learning_rate': hp.uniform('learning_rate', 0.01, 0.2)},
        'SVR': {'C': hp.loguniform('C', np.log(0.1), np.log(10)), 'epsilon': hp.uniform('epsilon', 0.1, 0.5)},
        'KNeighbors Regressor': {'n_neighbors': scope.int(hp.quniform('n_neighbors', 3, 9, 1))}
    }

    trials = Trials()
    best_params = fmin(fn=objective, space=search_spaces[model_name], algo=tpe.suggest, max_evals=10, trials=trials)
    print(f"Best parameters for {model_name} (Hyperopt): {best_params}")
    model.set_params(**best_params)
    return model

# Execute training, tuning, and testing
def train_and_evaluate(xtrain, ytrain, xval, yval, xtest, ytest):
    cross_val_results = evaluate_models(xtrain, ytrain)

    # Select top 3 models based on cross-validation results
    top_models = sorted(cross_val_results, key=cross_val_results.get)[:3]

    best_models = {}
    for model_name in top_models:
        model = models[model_name]
        print(f"\nTuning {model_name}...")

        # Random Search Tuning
        if param_grids[model_name]:
            best_model_random = tune_with_random_search(xtrain, ytrain, model_name, model, param_grids[model_name])
            best_models[f"{model_name} - RandomSearchCV"] = best_model_random

        # Hyperopt Tuning
        best_model_hyperopt = tune_with_hyperopt(xtrain, ytrain, model_name, model)
        best_models[f"{model_name} - Hyperopt"] = best_model_hyperopt

    # Evaluate each best model on validation and test set
    for name, model in best_models.items():
        model.fit(xtrain, ytrain)
        val_preds = model.predict(xval)
        test_preds = model.predict(xtest)
        
        val_mse = mean_squared_error(yval, val_preds)
        test_mse = mean_squared_error(ytest, test_preds)
        
        print(f"\n{name} - Validation MSE: {val_mse}")
        print(f"{name} - Test MSE: {test_mse}")

# Run the entire pipeline
train_and_evaluate(xtrain, ytrain, xval, yval, xtest, ytest)

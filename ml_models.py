import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from error_handler import check_model_data_compatibility, ModelCompatibilityError

def train_model(X_train, y_train, model_type, algorithm, params=None):
    """
    Train a machine learning model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target variable
    model_type : str
        Type of model - "regression" or "classification"
    algorithm : str
        The algorithm to use
    params : dict, default=None
        Parameters for the algorithm
    
    Returns:
    --------
    tuple
        (trained_model, feature_importance)
    
    Raises:
    -------
    ModelCompatibilityError
        Si hay problemas de compatibilidad entre el modelo y los datos
    """
    if params is None:
        params = {}
    
    # Para depuración - imprimir el valor exacto del algoritmo
    print(f"Modelo recibido: '{algorithm}', tipo: '{model_type}'")
        
    # Verificar la compatibilidad entre el modelo y los datos
    check_model_data_compatibility(X_train, y_train, model_type, algorithm)
    
    # Initialize the model based on algorithm and model type
    # Normalizamos el nombre del algoritmo para hacer la comparación más robusta
    algorithm_lower = algorithm.lower().strip()
    
    if model_type == "regression":
        if "regresión lineal" in algorithm_lower or "regresion lineal" in algorithm_lower or algorithm == "Regresión Lineal":
            model = LinearRegression(**params)
        elif "random forest" in algorithm_lower:
            model = RandomForestRegressor(**params)
        elif "vectores de soporte" in algorithm_lower or "svm" in algorithm_lower or "support vector" in algorithm_lower:
            model = SVR(**params)
        elif "gradient boosting" in algorithm_lower:
            model = GradientBoostingRegressor(**params)
        elif "árbol" in algorithm_lower or "arbol" in algorithm_lower or "decision tree" in algorithm_lower:
            model = DecisionTreeRegressor(**params)
        else:
            raise ValueError(f"Algoritmo de regresión desconocido: {algorithm}")
    
    elif model_type == "classification":
        if "regresión logística" in algorithm_lower or "regresion logistica" in algorithm_lower:
            model = LogisticRegression(**params)
        elif "random forest" in algorithm_lower:
            model = RandomForestClassifier(**params)
        elif "vectores de soporte" in algorithm_lower or "svm" in algorithm_lower or "support vector" in algorithm_lower:
            model = SVC(**params)
        elif "árbol" in algorithm_lower or "arbol" in algorithm_lower or "decision tree" in algorithm_lower:
            model = DecisionTreeClassifier(**params)
        elif "k-vecinos" in algorithm_lower or "vecinos" in algorithm_lower or "knn" in algorithm_lower or "nearest neighbors" in algorithm_lower:
            model = KNeighborsClassifier(**params)
        else:
            raise ValueError(f"Algoritmo de clasificación desconocido: {algorithm}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'regression' or 'classification'.")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Get feature importance if available
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        if model_type == "regression" or (model_type == "classification" and len(model.classes_) == 2):
            feature_importance = np.abs(model.coef_)
            if feature_importance.ndim > 1:
                feature_importance = feature_importance[0]
        else:  # Multiclass logistic regression
            feature_importance = np.mean(np.abs(model.coef_), axis=0)
    
    return model, feature_importance

def predict(model, X):
    """
    Generate predictions using a trained model.
    
    Parameters:
    -----------
    model : object
        Trained model
    X : pandas.DataFrame
        Features to make predictions on
    
    Returns:
    --------
    numpy.ndarray
        Predictions
    """
    # Check if model has predict_proba (classification) or just predict (regression)
    if hasattr(model, 'predict_proba') and not isinstance(model, (SVR, SVC)):
        # For classification, return class predictions
        return model.predict(X)
    else:
        # For regression or SVM classification
        return model.predict(X)

def get_model_metrics(y_true, y_pred, model_type):
    """
    Calculate model performance metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_type : str
        Type of model - "regression" or "classification"
    
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    if model_type == "regression":
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2
        }
    
    elif model_type == "classification":
        # Check if binary or multiclass
        unique_classes = np.unique(y_true)
        is_binary = len(unique_classes) == 2
        
        accuracy = accuracy_score(y_true, y_pred)
        
        if is_binary:
            precision = precision_score(y_true, y_pred, average='binary')
            recall = recall_score(y_true, y_pred, average='binary')
            f1 = f1_score(y_true, y_pred, average='binary')
        else:
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'regression' or 'classification'.")

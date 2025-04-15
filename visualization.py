import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_predictions(y_true, y_pred, model_type):
    """
    Create a plot comparing actual vs predicted values.
    
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
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if model_type == "regression":
        # Scatter plot of actual vs predicted values
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel('Valores reales')
        ax.set_ylabel('Valores predichos')
        ax.set_title('Valores Reales vs Predichos')
        
        # Add R² to the plot
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
    else:  # classification
        try:
            # Get unique classes
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            
            # Create a confusion matrix manually
            conf_matrix = np.zeros((len(unique_classes), len(unique_classes)))
            for i in range(len(y_true)):
                true_idx = np.where(unique_classes == y_true[i])[0][0]
                pred_idx = np.where(unique_classes == y_pred[i])[0][0]
                conf_matrix[true_idx, pred_idx] += 1
            
            # Plot the matrix as a heatmap
            sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                        xticklabels=unique_classes, yticklabels=unique_classes, ax=ax)
            
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Valor real')
            ax.set_title('Matriz de Confusión')
        except Exception as e:
            # Fallback si hay algún error
            ax.text(0.5, 0.5, f"Error al generar la matriz de confusión: {str(e)}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, wrap=True)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_importance, feature_names):
    """
    Create a bar plot of feature importance.
    
    Parameters:
    -----------
    feature_importance : array-like
        Feature importance values
    feature_names : list
        Names of the features
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    if feature_importance is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Importancia de características no disponible para este modelo",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Asegurarse de que feature_names sea una lista para evitar errores
    if feature_names is None:
        feature_names = [f"Característica {i+1}" for i in range(len(feature_importance))]
    
    # Si los tamaños no coinciden, ajustar según sea necesario
    if len(feature_importance) != len(feature_names):
        # Usar los nombres disponibles o generar nuevos según sea necesario
        if len(feature_importance) > len(feature_names):
            additional_names = [f"Característica {i+1}" for i in range(len(feature_names), len(feature_importance))]
            feature_names = list(feature_names) + additional_names
        else:  # len(feature_importance) < len(feature_names)
            feature_names = list(feature_names)[:len(feature_importance)]
    
    # Create a DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Take top 15 features if there are many
    if len(importance_df) > 15:
        importance_df = importance_df.head(15)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bars
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
    
    # Customize appearance
    ax.set_xlabel('Importancia')
    ax.set_title('Importancia de las Características')
    ax.invert_yaxis()  # Display the most important feature at the top
    
    # Add values to the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                ha='left', va='center')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(conf_matrix, class_names):
    """
    Create a visualization of a confusion matrix.
    
    Parameters:
    -----------
    conf_matrix : array-like
        The confusion matrix
    class_names : array-like
        Names of the classes
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Asegurarse de que class_names tenga la longitud correcta
    if class_names is None:
        class_count = conf_matrix.shape[0]
        class_names = [f"Clase {i+1}" for i in range(class_count)]
    else:
        # Si las dimensiones no coinciden, usar solo las clases necesarias o generar nuevas
        if len(class_names) != conf_matrix.shape[0]:
            if len(class_names) > conf_matrix.shape[0]:
                class_names = class_names[:conf_matrix.shape[0]]
            else:
                additional_classes = [f"Clase {i+1}" for i in range(len(class_names), conf_matrix.shape[0])]
                class_names = list(class_names) + additional_classes
    
    # Plot the matrix as a heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor real')
    ax.set_title('Matriz de Confusión')
    
    plt.tight_layout()
    return fig

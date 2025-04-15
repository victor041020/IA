
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
import inspect

class ModelCompatibilityError(Exception):
    """Error personalizado para problemas de compatibilidad entre modelos y datos."""
    pass

def check_model_data_compatibility(X, y, model_type, algorithm):
    """
    Verifica la compatibilidad entre los datos y el modelo seleccionado.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Características de entrada
    y : pandas.Series
        Variable objetivo
    model_type : str
        Tipo de modelo - "regression" o "classification"
    algorithm : str
        El algoritmo seleccionado
    
    Raises:
    -------
    ModelCompatibilityError
        Si se detecta incompatibilidad entre los datos y el modelo
    """
    # Verificar si es un problema de regresión pero los datos parecen ser categoriales
    if model_type == "regression":
        # Verificar si y contiene valores categóricos (pocos valores únicos)
        unique_values = y.nunique()
        if unique_values <= 5 or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
            raise ModelCompatibilityError(
                f"Has seleccionado un modelo de regresión ({algorithm}), pero la variable objetivo "
                f"parece ser categórica con {unique_values} valores únicos. "
                f"Considera usar un modelo de clasificación en su lugar."
            )
    
    # Verificar si es un problema de clasificación pero los datos parecen ser continuos
    elif model_type == "classification":
        unique_values = y.nunique()
        # Si hay demasiados valores únicos para clasificación
        if unique_values > 50 and pd.api.types.is_numeric_dtype(y):
            raise ModelCompatibilityError(
                f"Has seleccionado un modelo de clasificación ({algorithm}), pero la variable objetivo "
                f"parece ser continua con {unique_values} valores únicos. "
                f"Considera usar un modelo de regresión en su lugar."
            )
        
        # Verificar si hay suficientes muestras por clase
        value_counts = y.value_counts()
        min_samples = value_counts.min()
        if min_samples < 5:
            rare_classes = value_counts[value_counts < 5].index.tolist()
            raise ModelCompatibilityError(
                f"Hay clases con muy pocas muestras: {rare_classes} "
                f"(mínimo: {min_samples}). Esto puede causar problemas al entrenar "
                f"modelos de clasificación. Considera aumentar los datos para estas clases o "
                f"usar técnicas de balanceo de clases."
            )
    
    # Verificar características para problemas comunes
    if X.shape[1] == 0:
        raise ModelCompatibilityError("No hay características seleccionadas para el entrenamiento.")
    
    # Verificar si hay suficientes muestras para entrenar
    if X.shape[0] < 10:
        raise ModelCompatibilityError(
            f"El conjunto de datos es demasiado pequeño ({X.shape[0]} muestras). "
            f"Se necesitan más datos para entrenar un modelo confiable."
        )
    
    # Verificar algoritmos específicos
    if algorithm == "Máquina de Vectores de Soporte" and X.shape[0] > 10000:
        raise ModelCompatibilityError(
            f"El conjunto de datos es demasiado grande ({X.shape[0]} muestras) para "
            f"una Máquina de Vectores de Soporte. Este algoritmo es más adecuado para "
            f"conjuntos de datos pequeños o medianos (< 10,000 muestras). "
            f"Considera usar Random Forest o Gradient Boosting en su lugar."
        )
    
    # Verificar problema de alta dimensionalidad
    if X.shape[1] > 100 and algorithm in ["Regresión Lineal", "Regresión Logística"]:
        raise ModelCompatibilityError(
            f"El conjunto de datos tiene demasiadas características ({X.shape[1]}) para "
            f"{algorithm}. Considera usar técnicas de reducción de dimensionalidad "
            f"o algoritmos más robustos como Random Forest."
        )

def display_error_modal(title, message, suggestions=None):
    """
    Muestra un error en una ventana emergente tipo modal en Streamlit.
    
    Parameters:
    -----------
    title : str
        Título del error
    message : str
        Mensaje de error detallado
    suggestions : list, optional
        Lista de sugerencias para resolver el problema
    """
    # Crear un contenedor con estilo de error
    error_container = st.container()
    with error_container:
        # Usar columnas para crear un efecto de modal
        col1, col2, col3 = st.columns([1, 10, 1])
        with col2:
            # Encabezado con icono
            st.markdown(f"<h3 style='color: #ff4b4b;'>⚠️ {title}</h3>", unsafe_allow_html=True)
            
            # Mensaje de error en un recuadro con estilo
            st.markdown(
                f"""
                <div style="background-color: #ffe5e5; padding: 15px; border-radius: 5px; border-left: 5px solid #ff4b4b;">
                <p style="font-weight: bold; color: #5f0000;">{message}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Sugerencias en una lista con viñetas
            if suggestions:
                st.markdown("<p style='font-weight: bold; margin-top: 15px;'>Sugerencias para resolver este problema:</p>", unsafe_allow_html=True)
                for suggestion in suggestions:
                    st.markdown(f"<p style='margin-left: 20px;'>• {suggestion}</p>", unsafe_allow_html=True)
            
            # Botón para cerrar
            st.button("Entendido", key="close_error_modal", use_container_width=True)

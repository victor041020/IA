import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import traceback
import sqlite3
from data_processor import preprocess_data, split_data
from ml_models import train_model, predict, get_model_metrics
from visualization import plot_predictions, plot_feature_importance, plot_confusion_matrix
from error_handler import ModelCompatibilityError, display_error_modal
from database import DatabaseHandler

# Initialize database
db = DatabaseHandler()

st.set_page_config(page_title="Aplicación de Predicción ML",
                   page_icon="📊",
                   layout="wide")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'confusion_matrix' not in st.session_state:
    st.session_state.confusion_matrix = None
if 'unique_classes' not in st.session_state:
    st.session_state.unique_classes = None


def reset_state():
    """Reset all session state variables"""
    st.session_state.data = None
    st.session_state.processed_data = None
    st.session_state.model = None
    st.session_state.predictions = None
    st.session_state.metrics = None
    st.session_state.target_column = None
    st.session_state.features = None
    st.session_state.model_type = None
    st.session_state.feature_importance = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.confusion_matrix = None
    st.session_state.unique_classes = None


# Main title
st.title("📊 Aplicación de Predicción con Machine Learning")

# Sidebar
with st.sidebar:
    st.header("Navegación")
    page = st.radio("Ir a", [
        "1. Cargar Datos", "2. Preprocesar Datos", "3. Entrenar Modelo",
        "4. Ver Predicciones", "5. Historial"
    ])

    st.header("Acerca de")
    st.markdown("""
    Esta aplicación te permite:
    - Cargar archivos CSV
    - Preprocesar tus datos
    - Entrenar modelos de machine learning
    - Generar y visualizar predicciones
    """)

    if st.button("Reiniciar Aplicación"):
        reset_state()
        st.rerun()

# 1. Upload Data Page
if page == "1. Cargar Datos":
    st.header("1. Carga tus Datos CSV")

    st.markdown("""
    Por favor, carga un archivo CSV que contenga tus datos.
    - El archivo debe tener una fila de encabezado.
    - Asegúrate de que tus datos sean limpios y consistentes.
    """)

    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data

            st.success(
                f"Archivo CSV cargado exitosamente con {data.shape[0]} filas y {data.shape[1]} columnas."
            )

            st.subheader("Vista previa de datos")
            st.dataframe(data.head())

            st.subheader("Información de datos")
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())

            st.subheader("Resumen estadístico")
            st.dataframe(data.describe())

            st.info(
                "👈 Continúa en '2. Preprocesar Datos' en el menú de navegación."
            )

        except Exception as e:
            st.error(f"Error al cargar el archivo CSV: {str(e)}")

# 2. Preprocess Data Page
elif page == "2. Preprocesar Datos":
    st.header("2. Preprocesa tus Datos")

    if st.session_state.data is None:
        st.warning(
            "Por favor, carga los datos primero en la sección 'Cargar Datos'.")
    else:
        data = st.session_state.data

        st.subheader("Selecciona la Variable Objetivo")
        target_column = st.selectbox(
            "Selecciona la columna que quieres predecir:", data.columns)
        st.session_state.target_column = target_column

        st.subheader("Selecciona las Características")
        feature_options = [col for col in data.columns if col != target_column]
        selected_features = st.multiselect(
            "Selecciona las características a usar para la predicción:",
            feature_options,
            default=feature_options)

        if len(selected_features) == 0:
            st.warning("Por favor, selecciona al menos una característica.")
        else:
            st.session_state.features = selected_features

            # Determine if classification or regression
            target_values = data[target_column].unique()
            if len(target_values
                   ) <= 10 or data[target_column].dtype == 'object' or data[
                       target_column].dtype == 'bool':
                model_type = "classification"
                st.info(
                    f"Basado en la variable objetivo, usaremos modelos de clasificación (detectados {len(target_values)} valores únicos)."
                )
            else:
                model_type = "regression"
                st.info(
                    "Basado en la variable objetivo, usaremos modelos de regresión."
                )
            st.session_state.model_type = model_type

            st.subheader("Opciones de Preprocesamiento")
            handle_missing = st.checkbox("Manejar valores faltantes",
                                         value=True)
            scale_data = st.checkbox("Escalar características numéricas",
                                     value=True)
            encode_categorical = st.checkbox("Codificar variables categóricas",
                                             value=True)
            test_size = st.slider("Tamaño del conjunto de prueba (%)",
                                  min_value=10,
                                  max_value=50,
                                  value=20,
                                  step=5) / 100

            if st.button("Preprocesar Datos"):
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Show progress steps
                    status_text.text(
                        "Extrayendo características y objetivo...")
                    progress_bar.progress(10)
                    time.sleep(0.5)

                    X = data[selected_features]
                    y = data[target_column]

                    status_text.text(
                        "Preprocesando datos (esto puede tomar un momento)...")
                    progress_bar.progress(30)

                    # Use smaller chunk of data for very large datasets to avoid memory issues
                    row_count = len(X)
                    sample_size = min(10000, row_count)
                    if row_count > 20000:
                        status_text.text(
                            f"Conjunto de datos grande detectado ({row_count} filas). Procesando una muestra primero..."
                        )
                        X_sample = X.sample(sample_size, random_state=42)

                        # Test preprocessing on sample to check for errors
                        try:
                            _ = preprocess_data(X_sample,
                                                handle_missing=handle_missing,
                                                scale_features=scale_data,
                                                encode_cat=encode_categorical)
                        except Exception as e:
                            raise Exception(
                                f"Error al probar con la muestra: {str(e)}")

                    # Preprocess the full data
                    status_text.text("Preprocesando todos los datos...")
                    progress_bar.progress(50)

                    X_processed = preprocess_data(
                        X,
                        handle_missing=handle_missing,
                        scale_features=scale_data,
                        encode_cat=encode_categorical)

                    status_text.text(
                        "Dividiendo los datos en conjuntos de entrenamiento y prueba..."
                    )
                    progress_bar.progress(75)

                    # Split the data
                    X_train, X_test, y_train, y_test = split_data(
                        X_processed, y, test_size=test_size)

                    status_text.text("Guardando los datos procesados...")
                    progress_bar.progress(90)

                    # Save the preprocessed data and splits
                    st.session_state.processed_data = X_processed
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test

                    if model_type == "classification":
                        st.session_state.unique_classes = np.unique(y)

                    # Clear progress indicators
                    status_text.empty()
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    progress_placeholder.empty()
                    progress_bar.empty()

                    st.success(
                        "¡Preprocesamiento de datos completado exitosamente!")

                    st.subheader("Vista previa de datos procesados")
                    st.dataframe(X_processed.head())

                    st.info(
                        "👈 Continúa en '3. Entrenar Modelo' en el menú de navegación."
                    )

                except Exception as e:
                    # Clear progress indicators
                    status_text.empty()
                    progress_placeholder.empty()
                    progress_bar.empty()

                    error_details = traceback.format_exc()
                    st.error(
                        f"Error durante el preprocesamiento de datos: {str(e)}"
                    )
                    with st.expander("Detalles del error"):
                        st.code(error_details)

# 3. Train Model Page
elif page == "3. Entrenar Modelo":
    st.header("3. Entrena el Modelo de Machine Learning")

    if st.session_state.processed_data is None:
        st.warning(
            "Por favor, completa el paso de preprocesamiento de datos primero."
        )
    else:
        model_type = st.session_state.model_type

        st.subheader("Selecciona un Modelo")

        if model_type == "classification":
            model_options = [
                "Regresión Logística", "Random Forest",
                "Máquina de Vectores de Soporte", "K-Vecinos Más Cercanos",
                "Árbol de Decisión"
            ]
        else:  # regression
            model_options = [
                "Regresión Lineal", "Random Forest",
                "Máquina de Vectores de Soporte", "Gradient Boosting",
                "Árbol de Decisión"
            ]

        selected_model = st.selectbox("Elige un modelo:", model_options)

        st.subheader("Parámetros del Modelo")

        # Common parameters
        random_state = 42

        # Model-specific parameters
        if "Random Forest" in selected_model:
            n_estimators = st.slider("Número de árboles",
                                     min_value=10,
                                     max_value=500,
                                     value=100,
                                     step=10)
            max_depth = st.slider("Profundidad máxima del árbol",
                                  min_value=1,
                                  max_value=30,
                                  value=10,
                                  step=1)
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state
            }
        elif "Regresión Lineal" in selected_model:
            params = {}
        elif "Regresión Logística" in selected_model:
            C = st.slider("Fuerza de regularización (C)",
                          min_value=0.01,
                          max_value=10.0,
                          value=1.0,
                          step=0.01)
            params = {"C": C, "random_state": random_state}
        elif "Máquina de Vectores de Soporte" in selected_model:
            C = st.slider("Parámetro de regularización (C)",
                          min_value=0.1,
                          max_value=10.0,
                          value=1.0,
                          step=0.1)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            params = {
                "C": C,
                "kernel": kernel,
                "random_state": random_state if kernel != "linear" else None
            }
        elif "Gradient Boosting" in selected_model:
            n_estimators = st.slider("Número de etapas de boosting",
                                     min_value=10,
                                     max_value=500,
                                     value=100,
                                     step=10)
            learning_rate = st.slider("Tasa de aprendizaje",
                                      min_value=0.01,
                                      max_value=1.0,
                                      value=0.1,
                                      step=0.01)
            params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "random_state": random_state
            }
        elif "Árbol de Decisión" in selected_model:
            max_depth = st.slider("Profundidad máxima del árbol",
                                  min_value=1,
                                  max_value=30,
                                  value=10,
                                  step=1)
            params = {"max_depth": max_depth, "random_state": random_state}
        elif "K-Vecinos Más Cercanos" in selected_model:
            n_neighbors = st.slider("Número de vecinos",
                                    min_value=1,
                                    max_value=20,
                                    value=5,
                                    step=1)
            params = {"n_neighbors": n_neighbors}

        if st.button("Entrenar Modelo"):
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            error_container = st.empty()

            try:
                # Show progress steps
                status_text.text("Preparando los datos...")
                progress_bar.progress(10)
                time.sleep(0.5)

                # Get the training and testing data
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test

                status_text.text(
                    f"Verificando compatibilidad entre datos y modelo...")
                progress_bar.progress(20)

                # Verificar compatibilidad datos-modelo
                try:
                    status_text.text(f"Entrenando modelo {selected_model}...")
                    progress_bar.progress(40)

                    # Train the model - Esto ya hace la verificación de compatibilidad
                    model, feature_importance = train_model(
                        X_train,
                        y_train,
                        model_type=model_type,
                        algorithm=selected_model,
                        params=params)
                except ModelCompatibilityError as e:
                    # Limpiar barras de progreso
                    status_text.empty()
                    progress_placeholder.empty()
                    progress_bar.empty()

                    # Mostrar ventana emergente de error con sugerencias
                    with error_container.container():
                        display_error_modal(
                            "Incompatibilidad entre Datos y Modelo",
                            str(e),
                            suggestions=[
                                "Verifica que el tipo de modelo (clasificación/regresión) sea apropiado para tu variable objetivo.",
                                "Para variables categóricas (pocos valores únicos), usa modelos de clasificación.",
                                "Para variables continuas (muchos valores únicos), usa modelos de regresión.",
                                "Asegúrate de tener suficientes datos para cada clase en problemas de clasificación.",
                                "Considera usar técnicas de balanceo de clases si hay clases minoritarias."
                            ])
                    # En lugar de 'return' usamos una bandera para detectar errores
                    raise Exception("Problema de compatibilidad detectado")

                status_text.text("Generando predicciones...")
                progress_bar.progress(70)

                # Make predictions and calculate metrics
                predictions = predict(model, X_test)
                metrics = get_model_metrics(y_test, predictions, model_type)

                # For classification, calculate confusion matrix
                if model_type == "classification":
                    status_text.text("Calculando la matriz de confusión...")
                    from sklearn.metrics import confusion_matrix
                    conf_matrix = confusion_matrix(y_test, predictions)
                    st.session_state.confusion_matrix = conf_matrix

                status_text.text("Guardando el modelo y resultados...")
                progress_bar.progress(90)

                # Save the model to database
                model_id = db.save_model(selected_model, model_type, model,
                                         metrics)
                db.save_predictions(model_id, y_test, predictions)

                # Save to session state
                st.session_state.model = model
                st.session_state.predictions = predictions
                st.session_state.metrics = metrics
                st.session_state.feature_importance = feature_importance
                st.session_state.model_id = model_id

                # Clear progress indicators
                status_text.empty()
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_placeholder.empty()
                progress_bar.empty()

                st.success(
                    "¡El entrenamiento del modelo se completó correctamente!")

                # Display metrics
                st.subheader("Métricas de rendimiento del modelo")
                if metrics:
                    metrics_df = pd.DataFrame([metrics]).round(4)
                    st.dataframe(metrics_df)

                st.info(
                    "👈 Proceda a '4. Ver Predicciones' en el menú de navegación para ver los resultados."
                )

            except Exception as e:
                # Clear progress indicators
                status_text.empty()
                progress_placeholder.empty()
                progress_bar.empty()

                error_details = traceback.format_exc()
                st.error(
                    f"Error durante el entrenamiento del modelo: {str(e)}")
                with st.expander("Detalles del error"):
                    st.code(error_details)

# 4. View Predictions Page
elif page == "5. Historial":
    st.header("5. Historial de Modelos y Predicciones")

    # Obtener todos los modelos
    models = db.get_all_models()

    if not models:
        st.info("No hay modelos guardados en la base de datos.")
    else:
        # Crear DataFrame con la información de los modelos
        models_df = pd.DataFrame(
            models,
            columns=['ID', 'Nombre', 'Tipo', 'Métricas', 'Fecha de Creación'])
        st.subheader("Modelos Guardados")
        st.dataframe(models_df)

        # Permitir seleccionar un modelo para ver sus predicciones
        selected_model_id = st.selectbox(
            "Selecciona un modelo para ver sus predicciones:", models_df['ID'])

        if selected_model_id:
            # Consultar predicciones del modelo seleccionado
            with sqlite3.connect(db.db_name) as conn:
                predictions_df = pd.read_sql_query(
                    "SELECT actual_value as 'Valor Real', predicted_value as 'Predicción', created_at as 'Fecha' FROM predictions WHERE model_id = ?",
                    conn,
                    params=(selected_model_id, ))

            if not predictions_df.empty:
                st.subheader(f"Predicciones del Modelo {selected_model_id}")
                st.dataframe(predictions_df)

                # Botón para descargar predicciones
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Descargar Predicciones como CSV",
                    data=csv,
                    file_name=f"predicciones_modelo_{selected_model_id}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No hay predicciones guardadas para este modelo.")

elif page == "4. Ver Predicciones":
    st.header("4. Ver Predicciones y Resultados")

    if st.session_state.model is None or st.session_state.predictions is None:
        st.warning(
            "Por favor, completa el paso de entrenamiento del modelo primero.")
    else:
        # Get data from session state
        model = st.session_state.model
        predictions = st.session_state.predictions
        metrics = st.session_state.metrics
        model_type = st.session_state.model_type
        feature_importance = st.session_state.feature_importance
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # Create a DataFrame with actual vs predicted values
        results_df = pd.DataFrame({
            'Valor Real': y_test,
            'Predicción': predictions
        })

        # Display prediction metrics
        st.subheader("Métricas de Rendimiento del Modelo")
        metrics_df = pd.DataFrame([metrics]).round(4)
        st.dataframe(metrics_df)

        # Visualizations
        st.subheader("Visualizaciones")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Valores Reales vs Predichos")
            fig = plot_predictions(y_test, predictions, model_type)
            st.pyplot(fig)

        with col2:
            if feature_importance is not None:
                st.write("Importancia de Características")
                fig = plot_feature_importance(feature_importance,
                                              st.session_state.features)
                st.pyplot(fig)

            if model_type == "classification" and st.session_state.confusion_matrix is not None:
                st.write("Matriz de Confusión")
                fig = plot_confusion_matrix(st.session_state.confusion_matrix,
                                            st.session_state.unique_classes)
                st.pyplot(fig)

        # Display results table
        st.subheader("Resultados de Predicción")
        st.dataframe(results_df)

        # Add download button for results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Descargar Predicciones como CSV",
            data=csv,
            file_name="resultados_prediccion.csv",
            mime="text/csv",
        )

        # Make new predictions with user input
        st.subheader("Realizar Nuevas Predicciones")
        st.markdown(
            "Carga un nuevo archivo CSV con las mismas características para generar predicciones."
        )

        new_file = st.file_uploader("Elige un archivo CSV",
                                    type="csv",
                                    key="new_prediction_file")

        if new_file is not None:
            try:
                new_data = pd.read_csv(new_file)

                # Check if the new data has the required features
                required_features = st.session_state.features
                missing_features = [
                    f for f in required_features if f not in new_data.columns
                ]

                if missing_features:
                    st.error(
                        f"Al archivo cargado le faltan las siguientes características requeridas: {', '.join(missing_features)}"
                    )
                else:
                    # Preprocess the new data
                    new_data_processed = preprocess_data(
                        new_data[required_features],
                        handle_missing=True,
                        scale_features=True,
                        encode_cat=True)

                    # Generate predictions
                    new_predictions = predict(model, new_data_processed)

                    # Create results DataFrame
                    new_results_df = new_data.copy()
                    new_results_df['Predicción'] = new_predictions

                    st.subheader("Nuevas Predicciones")
                    st.dataframe(new_results_df)

                    # Add download button for new predictions
                    new_csv = new_results_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar Nuevas Predicciones como CSV",
                        data=new_csv,
                        file_name="nuevas_predicciones.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Error al procesar los nuevos datos: {str(e)}")


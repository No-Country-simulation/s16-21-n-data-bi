import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from PIL import Image

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data_paises_ML.csv", usecols=range(5))
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Preparar datos
def prepare_data(df, X_days=8):
    df['day_of_week'] = df['date'].dt.day_name()
    df = df.sort_values(by=['country_code', 'date'])

    for i in range(1, X_days + 1):
        df[f'new_confirmed_lag_{i}'] = df.groupby('country_code')['new_confirmed'].shift(i)

    df = df.dropna()
    df = pd.get_dummies(df, columns=['country_code', 'day_of_week'], drop_first=True)
    
    return df

# Entrenar modelo
def train_model(df):
    feature_cols = [col for col in df.columns if col not in ['location_key', 'date', 'country_name', 'new_confirmed']]
    X = df[feature_cols]
    y = df['new_confirmed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression(positive=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    return model, X_test, y_test, y_pred, r2, mae, mse, rmse

# Función principal de la app
def main():
    st.set_page_config(page_title="Predicción de Nuevos Casos y Decesos", layout="wide")
    
    # Cargar imagen del logo
    logo = Image.open("logo_empresa.png")
    st.image(logo, use_column_width=True)

    # Cargar CSS personalizado
    try:
        with open("style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al cargar el CSS: {e}")

    st.sidebar.title("Opciones")
    page = st.sidebar.radio("Seleccione una vista", ["Nuevos Casos", "Nuevos Decesos", "Pre-Vacunación", "Post-Vacunación", "Comparativo"])

    df = load_data()
    if df is not None:
        if page == "Nuevos Casos":
            st.header("Predicción de Nuevos Casos")
            
            df_cases = prepare_data(df)
            model, X_test, y_test, y_pred, r2, mae, mse, rmse = train_model(df_cases)

            st.subheader("Métricas de Evaluación")
            st.markdown(f"""
            - **R² (R cuadrado)**: {r2:.2f} - Indica qué tan bien se ajusta el modelo a los datos observados.
            - **MAE (Error Absoluto Medio)**: {mae:.2f} - Promedio de los errores absolutos.
            - **MSE (Error Cuadrático Medio)**: {mse:.2f} - Promedio de los errores al cuadrado.
            - **RMSE (Raíz del Error Cuadrático Medio)**: {rmse:.2f} - Raíz cuadrada del MSE.
            """)

            st.subheader("Seleccionar País")
            selected_country = st.selectbox("Selecciona el país", ["Todos", "Perú", "Brasil", "Chile"])

            # Mostrar banderas debajo del selector de país
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.image("banderas/peru.png", width=50)
            with col2:
                st.image("banderas/brasil.png", width=50)
            with col3:
                st.image("banderas/chile.png", width=50)
            
            if selected_country != "Todos":
                country_code_map = {
                    "Perú": "PE",
                    "Brasil": "BR",
                    "Chile": "CL"
                }
                country_code = country_code_map[selected_country]
                df_cases = df_cases[df_cases[f'country_code_{country_code}'] == 1]
            
            st.subheader("Gráfico de Predicciones vs. Valores Reales")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.title('Predicciones vs. Valores Reales')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
            st.pyplot(plt)

        elif page == "Nuevos Decesos":
            st.header("Predicción de Nuevos Decesos")
            st.write("Esta sección está en desarrollo.")

        elif page == "Pre-Vacunación":
            st.header("Datos Pre-Vacunación")
            st.write("Esta sección está en desarrollo.")

        elif page == "Post-Vacunación":
            st.header("Datos Post-Vacunación")
            st.write("Esta sección está en desarrollo.")

        elif page == "Comparativo":
            st.header("Comparativo entre Países")
            st.write("Esta sección está en desarrollo.")
            # Aquí podrías agregar la lógica para mostrar gráficos comparativos entre países

if __name__ == "__main__":
    main()

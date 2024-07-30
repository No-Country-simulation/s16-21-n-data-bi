import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("data_paises_ML.csv", usecols=range(5))
    df['date'] = pd.to_datetime(df['date'])
    return df

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
    st.title("Predicción de Nuevos Casos y Decesos")

    # Cargar CSS personalizado
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.sidebar.title("Opciones")
    option = st.sidebar.selectbox("Selecciona la vista", ["Nuevos Casos", "Decesos"])
    
    df = load_data()
    
    if option == "Nuevos Casos":
        st.header("Predicción de Nuevos Casos")
        
        df_cases = prepare_data(df)
        model, X_test, y_test, y_pred, r2, mae, mse, rmse = train_model(df_cases)

        st.subheader("Métricas de Evaluación")
        st.write(f"R²: {r2}")
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")
        st.write(f"RMSE: {rmse}")

        st.subheader("Gráfico de Predicciones vs. Valores Reales")
        plt.scatter(y_test, y_pred)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs. Valores Reales')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        st.pyplot(plt)

    elif option == "Decesos":
        st.header("Predicción de Decesos")
        st.write("Esta sección está en desarrollo.")
        # Aquí se puede añadir el código similar para la predicción de decesos

if __name__ == "__main__":
    main()

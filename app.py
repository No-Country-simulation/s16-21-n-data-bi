import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

# Cargar los archivos CSV
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Cargar los datos
deceased_data = load_data('data/processed/prediccion_ML_deceased.csv')
cases_data = load_data('data/processed/prediccion_ML_casos.csv')

# Convertir las columnas de fecha a datetime
deceased_data['date'] = pd.to_datetime(deceased_data['date'])
cases_data['date'] = pd.to_datetime(cases_data['date'])

# Función para mostrar las predicciones
def mostrar_predicciones(data, title):
    st.title(title)

    # Selector de país
    countries = data['location_key'].unique()
    selected_country = st.selectbox('Selecciona un país:', countries)

    # Selector de fecha, con fecha predeterminada 2021-01-01
    selected_date = st.date_input('Selecciona una fecha:', value=datetime(2021, 1, 1), min_value=datetime(2021, 1, 1), max_value=datetime.now().date())

    # Filtrar los datos para el país y la fecha seleccionados
    filtered_data = data[(data['location_key'] == selected_country) & (data['date'] == pd.to_datetime(selected_date))]

    if not filtered_data.empty:
        # Obtener los datos de los próximos 14 días
        future_dates = [selected_date + timedelta(days=i) for i in range(15)]
        future_data = data[(data['location_key'] == selected_country) & (data['date'].isin(future_dates))]

        # Formatear la columna de fechas para mostrar solo la fecha sin la hora
        future_data['date'] = future_data['date'].dt.strftime('%Y-%m-%d')

        st.write('Predicciones para los próximos 14 días:')
        st.write(future_data[['date', 'location_key', 'Prediccion']].to_html(index=False), unsafe_allow_html=True)
    else:
        st.write('No hay datos disponibles para la fecha y el país seleccionados.')

# Configuración de la barra de navegación con pestañas
with st.sidebar:
    selected = option_menu(
        menu_title="Navegación",
        options=["Nosotros", "Predicción de decesos", "Prediccion de Casos", "Reporte", "Gráficos ML"],
        icons=["people", "activity", "bar-chart", "table", "bar-chart"],
        menu_icon="cast",
        default_index=0,
    )

# Nosotros
if selected == "Nosotros":
    st.title("Presentación del Equipo de Trabajo")
    st.write("""
    **Equipo de Predicción de COVID-19**
    
    Nuestro equipo está compuesto por los siguientes miembros:
    
    - **Angel Jaramillo Sulca** (Data Engineer): [LinkedIn](https://www.linkedin.com/in/angeljarads/) | [GitHub](https://github.com/Angeljs094)
    - **Carolina Romero** (Machine Learning Engineer): [LinkedIn](https://www.linkedin.com/in/carolina-romerou/) | [GitHub](https://github.com/caromerou)
    - **Fabricio Diego Angulo Luna** (Data Engineer): [LinkedIn](https://www.linkedin.com/in/fabricio-diego-angulo-luna-0a8b46259/) | [GitHub](https://github.com/FabricioAngulo)
    - **Fabrizio Flamini** (Data Analyst): [LinkedIn](https://www.linkedin.com/in/fabrizioflamini/) | [GitHub](https://github.com/FlamInIFabrIzIo)
    - **Gabriel Valdez** (Data Analyst): [LinkedIn](https://www.linkedin.com/in/gabdez/) | [GitHub](https://github.com/GabooV2)
    - **Gonzalo Raffo** (Data Engineer): [LinkedIn](https://www.linkedin.com/in/gonzaloraffo/) | [GitHub](https://github.com/goraffo)
    - **Marcelo Ortiz** (Machine Learning Engineer): [LinkedIn](https://www.linkedin.com/in/marceloortizz/) | [GitHub](https://github.com/MarceloOrtizz)
    - **Yesica Milagros Leon** (Data Analyst): [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/yesicamilagros)
    
    Nuestro objetivo es proporcionar predicciones precisas y útiles para ayudar en la lucha contra la pandemia.
    """)

# Predicción de decesos
elif selected == "Predicción de decesos":
    mostrar_predicciones(deceased_data, "Predicción de Decesos")

# Prediccion de Casos
elif selected == "Prediccion de Casos":
    mostrar_predicciones(cases_data, "Predicción de Casos Confirmados")

# Dashboard Power BI
elif selected == "Reporte":
    embed_url = "https://app.powerbi.com/view"
    st.components.v1.iframe(src=embed_url, height=800, width=1100)

# Gráficos ML
elif selected == "Gráficos ML":
    st.title("Gráficos de Predicción vs Realidad")

    # Selector de tipo de predicción
    prediccion_tipo = st.selectbox('Selecciona el tipo de predicción:', ['Decesos', 'Casos Confirmados'])

    # Selector de país
    if prediccion_tipo == 'Decesos':
        data = deceased_data
        y_label_real = 'new_deceased'
        y_label_pred = 'Prediccion'
        y_title = 'Decesos'
    else:
        data = cases_data
        y_label_real = 'new_confirmed'  
        y_label_pred = 'Prediccion'
        y_title = 'Casos Confirmados'

    countries = data['location_key'].unique()
    selected_country = st.selectbox('Selecciona un país:', countries)

    # Filtrar el DataFrame para el país seleccionado
    df_selected = data[data['location_key'] == selected_country]

    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_selected['date'], df_selected[y_label_real], label=y_title)
    ax.plot(df_selected['date'], df_selected[y_label_pred], label='Predicción')

    # Añadir títulos y etiquetas
    ax.set_title(f'{y_title} vs Predicción en {selected_country}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel(y_title)
    ax.legend()

    # Rotar las etiquetas del eje x para mejor visualización
    plt.xticks(rotation=45)

    # Mostrar el gráfico
    st.pyplot(fig)
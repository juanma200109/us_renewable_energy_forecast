import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# Función para cargar datos desde un archivo CSV a un DataFrame de pandas
# ==============================================================================

def cargar_datos(ruta_archivo):
    """
    Carga un archivo CSV y devuelve un DataFrame de pandas.

    Entradas:
        - ruta_archivo: str, ruta del archivo CSV a cargar
    Salidas:
        - datos: DataFrame, datos cargados desde el archivo CSV    
    """
    try:
        # Cargar datos desde archivo CSV
        datos = pd.read_csv(ruta_archivo)
        return datos
    except FileNotFoundError:
        print("Error: El archivo no existe")
    except pd.errors.EmptyDataError:
        print("Error: El archivo está vacío")
    except pd.errors.ParserError:
        print("Error: Error al analizar el archivo")
    except Exception as e:
        print(f"Error inesperado: {e}")
    return None

# ==============================================================================
# Función para fusionar las columnas 'Year' y 'Month' en una columna de fecha
# ==============================================================================

def crear_indice_fecha(df):
    """
    Crea una columna de fecha a partir de las columnas 'Year' y 'Month' y la establece como índice.

    Esta función agrega una columna temporal 'Day' con valor 1, combina las columnas 'Year', 'Month' y 'Day'
    para crear una columna 'datetime' en formato de fecha, elimina las columnas temporales ('Year', 'Month', 'Day')
    y establece 'datetime' como el índice del DataFrame. Si la columna 'datetime' ya existe, no realiza cambios.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame de entrada que debe contener las columnas 'Year' y 'Month' si 'datetime' no está presente.

    Retorna:
    --------
    pandas.DataFrame
        DataFrame con la columna 'datetime' como índice y las columnas 'Year', 'Month' y 'Day' eliminadas
        si se crearon.
    """
    # Crear una copia del DataFrame para no modificar el original
    df_modificado = df.copy()

    # Verificar si 'datetime' no está en las columnas
    if 'datetime' not in df_modificado.columns:
        # Verificar si 'Year' y 'Month' están presentes
        if 'Year' in df_modificado.columns and 'Month' in df_modificado.columns:
            # Asignar el día 1 a una nueva columna 'Day'
            df_modificado['Day'] = 1
            # Crear la columna 'datetime' combinando 'Year', 'Month' y 'Day'
            df_modificado['datetime'] = pd.to_datetime(
                df_modificado[['Year', 'Month', 'Day']], format='%Y-%m'
            )
            # Eliminar las columnas 'Year', 'Month' y 'Day'
            df_modificado = df_modificado.drop(columns=['Year', 'Month', 'Day'])
            # Establecer 'datetime' como índice
            df_modificado.set_index('datetime', inplace=True)

    return df_modificado

# ==============================================================================
# Función para imputar datos de consumo en un DataFrame
# ==============================================================================

def imputar_datos_consumo(df, columnas_consumo):
    """
    Imputa valores faltantes en columnas específicas de un DataFrame.

    Esta función procesa un DataFrame creando una copia para no modificar el original,
    y para cada columna especificada:
        - Establece en 0 todos los valores antes de la primera observación válida (no NaN).
        - Interpola linealmente los valores faltantes a partir de la primera observación válida.
        - Si una columna está completamente vacía (solo NaN), la rellena con 0.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame de entrada con los datos a imputar. Generalmente indexado por fechas u otro índice secuencial.
    columnas_consumo : list de str
        Lista de nombres de columnas del DataFrame donde se aplicará el proceso de imputación.

    Retorna:
    --------
    pandas.DataFrame
        Un nuevo DataFrame con los valores imputados en las columnas especificadas.
    """
    # Crear una copia para no modificar el DataFrame original
    df_imputado = df.copy()

    # Iterar por cada columna de consumo
    for columna in columnas_consumo:
        # Encontrar el índice de la primera observación válida (no NaN)
        primer_indice_valido = df_imputado[columna].first_valid_index()

        if primer_indice_valido is not None:
            # Si hay al menos un valor válido en la columna:
            # a) Establecer en 0 todos los valores ANTES de la primera observación válida
            df_imputado.loc[:primer_indice_valido, columna] = df_imputado.loc[:primer_indice_valido, columna].fillna(0)

            # b) Interpolar los valores faltantes a partir de la primera observación válida
            df_imputado.loc[primer_indice_valido:, columna] = df_imputado.loc[primer_indice_valido:, columna].interpolate(
                method='linear', limit_direction='both'
            )
        else:
            # Si toda la columna es NaN, rellenar con 0
            df_imputado[columna] = df_imputado[columna].fillna(0)

    return df_imputado


def guardar_datos(df, ruta_archivo):
    """
    Guarda un DataFrame de pandas en un archivo CSV.

    Entradas:
        - df: DataFrame, datos a guardar
        - ruta_archivo: str, ruta del archivo CSV donde se guardarán los datos
    Salidas:
        - None
    """
    try:
        df.to_csv(ruta_archivo, index=True)
        print(f"Datos guardados correctamente en {ruta_archivo}")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
        
def mix_heatmap(df, columnas_consumo):
    """
    Crea un heatmap de consumo de energía.

    Esta función toma un DataFrame y una lista de columnas de consumo, y genera un heatmap
    que muestra el consumo de energía a lo largo del tiempo.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos de consumo de energía.
    columnas_consumo : list de str
        Lista de nombres de columnas que representan diferentes tipos de consumo.

    Retorna:
    --------
    matplotlib.figure.Figure
        Un objeto Figure que contiene el heatmap generado.
    """

    # Obtenemos los años del índice
    years = df['datetime'].dt.year

    # Creamos un DataFrame agrupado por año
    df_annual = df.groupby(years)[columnas_consumo].sum()

    # Calculamos los porcentajes por año
    for year in df_annual.index:
        total = df_annual.loc[year].sum()
        if total > 0:  # Evitamos divisiones por cero
            df_annual.loc[year] = df_annual.loc[year] / total * 100

    # Transponemos para tener fuentes en filas y años en columnas
    df_annual_transposed = df_annual.T

    # Visualización: Heatmap por Año (con selección de años para mayor claridad)

    plt.figure(figsize=(16, 10))

    # Seleccionamos un subconjunto de años para evitar sobrecarga visual, por ejemplo, uno de cada 5 años
    all_years = sorted(list(set(years)))
    selected_years = all_years[::5]  # Toma uno cada 5 años
    if all_years[-1] not in selected_years:  # Asegura incluir el año más reciente
        selected_years.append(all_years[-1])

    # Filtramos el DataFrame para mostrar solo los años seleccionados
    df_annual_selected = df_annual_transposed[selected_years]

    # Creamos el heatmap
    sns.heatmap(df_annual_selected, cmap='viridis', 
                cbar_kws={'label': 'Porcentaje de Consumo (%)'}, 
                linewidths=0.3,
                annot=True,  # Mostramos los valores
                fmt='.1f')   # Con un decimal

    plt.title('Evolución del Consumo Energético por Fuente (Años Seleccionados)', fontsize=16, fontweight='bold')
    plt.xlabel('Año', fontsize=14)
    plt.ylabel('Fuente de Energía', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_sector_consumo(df, columnas_consumo):
    """
    Crea un gráfico de lineas del consumo por sector.

    Esta función toma un DataFrame y una lista de columnas de consumo, y genera un gráfico
    de lineas que muestra el consumo de energías renovables a lo largo de los años para cada sector.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos de consumo de energía.
    columnas_consumo : list de str
        Lista de nombres de columnas que representan diferentes tipos de consumo.

    Retorna:
    --------
    matplotlib.figure.Figure
        Un objeto Figure que contiene el gráfico generado.
    """
   # Se calcula el total de energía renovable consumida por todos los sectores
    df_processed_4 = df.groupby(['datetime', 'Sector'])[columnas_consumo].sum()
    df_processed_4['Total Renewable Energy'] = df_processed_4.sum(axis=1)
    df_processed_4 = df_processed_4.reset_index('Sector')

    # Grafico de líneas por sector
    plt.figure(figsize=(14, 7))
    df_processed_1 = df_processed_4['Total Renewable Energy']
    df_processed_1[df_processed_4['Sector'] == 'Commercial'].plot(kind = 'line', linewidth=2, label='Commercial')
    df_processed_1[df_processed_4['Sector'] == 'Residential'].plot(kind = 'line', linewidth=2, label='Residential')
    df_processed_1[df_processed_4['Sector'] == 'Industrial'].plot(kind = 'line', linewidth=2, label='Industrial')
    df_processed_1[df_processed_4['Sector'] == 'Transportation'].plot(kind = 'line', linewidth=2, label='Transportation')
    df_processed_1[df_processed_4['Sector'] == 'Electric Power'].plot(kind = 'line', linewidth=2, label='Electric Power')
    plt.title('Consumo Total por mes de Energía Renovable en EE.UU por Sector. (1973-2024)', fontsize=16)
    plt.xlabel('Año', fontsize=12)
    plt.ylabel('Consumo (Trillion BTU)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Sector', fontsize=10)
    plt.tight_layout()
    plt.show()

def bar_sector_consumo(df, columnas_consumo):
    """
    Crea un gráfico de barras del consumo por sector.

    Esta función toma un DataFrame y una lista de columnas de consumo, y genera un gráfico
    de barras que muestra el consumo de energías renovables a lo largo de los años para cada sector.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos de consumo de energía.
    columnas_consumo : list de str
        Lista de nombres de columnas que representan diferentes tipos de consumo.

    Retorna:
    --------
    matplotlib.figure.Figure
        Un objeto Figure que contiene el gráfico generado.
    """
    # Asegurarse de que las columnas de consumo sean numéricas
    df[columnas_consumo] = df[columnas_consumo].apply(pd.to_numeric, errors='coerce')

    # Asegurarse de que la columna datetime sea de tipo datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Calcular el total de energía renovable consumida por todos los sectores
    df_processed_5 = df.groupby(['datetime', 'Sector'])[columnas_consumo].sum()
    df_processed_5['Total Renewable Energy'] = df_processed_5.sum(axis=1)
    df_processed_5 = df_processed_5.reset_index('Sector')

    # Extraer los años del índice datetime
    df_processed_5['year'] = df_processed_5.index.year

    # Agrupar por año y sector, y pivotar
    df_processed_5 = df_processed_5.groupby(['year', 'Sector'])['Total Renewable Energy'].sum().unstack()

    # Llenar valores NaN con 0
    df_processed_5 = df_processed_5.fillna(0)

    # Asegurarse de que los datos sean numéricos
    df_processed_5 = df_processed_5.astype(float)

    # Verificar el DataFrame antes de graficar
    print("DataFrame procesado:")
    print(df_processed_5)
    print("Tipos de datos:")
    print(df_processed_5.dtypes)

    # Crear el gráfico de barras apiladas
    plt.figure(figsize=(16, 10))
    df_processed_5.plot(kind='bar', stacked=True, colormap='viridis', edgecolor='black')
    plt.title('Consumo Total por año de Energía Renovable por Sector (1973-2024)', fontsize=16)
    plt.xlabel('Año', fontsize=12)
    plt.ylabel('Consumo (Trillion BTU)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Sector', fontsize=10)
    plt.tight_layout()
    plt.show()

    return plt.gcf()
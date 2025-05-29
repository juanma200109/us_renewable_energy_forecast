import pandas as pd
import numpy as np

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
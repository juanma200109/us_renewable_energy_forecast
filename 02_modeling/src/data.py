import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
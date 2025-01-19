import boto3
from botocore.exceptions import NoCredentialsError
import streamlit as st
import json
from datetime import datetime

class AWSStorage:
    """
    Clase para guardar diccionarios como archivos JSON en un bucket de AWS S3.
    """

    def __init__(self,
                    aws_access_key_id = st.secrets['aws']['aws_access_key_id'],
                    aws_secret_access_key = st.secrets['aws']['aws_secret_access_key'],
                    region_name = st.secrets['aws']['region_name'],  # Puede cambiar la región si es necesario
                 ):
        """
        Inicializa una instancia de AWSStorage con las credenciales de AWS proporcionadas.

        Parámetros
        ----------
        aws_access_key_id : str
            Clave de acceso de AWS.
        aws_secret_access_key : str
            Clave secreta de acceso de AWS.
        region_name : str
            Nombre de la región de AWS (por ejemplo, 'us-west-2').
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name

        # Crear un cliente de S3 utilizando las credenciales proporcionadas
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )

    def guardar_diccionario(self, diccionario_str: str, bucket_name: str = 'potatochallengelogs', s3_key_prefix: str = ''):
        """
        Guarda un diccionario en un archivo JSON en un bucket de AWS S3.

        Parámetros
        ----------
        diccionario : dict
            El diccionario que se desea guardar.
        bucket_name : str
            El nombre del bucket de S3 donde se guardará el archivo.
        s3_key_prefix : str, opcional
            Prefijo de la clave S3 (ruta dentro del bucket), por defecto es una cadena vacía.

        Ejemplos
        --------
        >>> storage = AWSStorage('ACCESS_KEY', 'SECRET_KEY', 'us-west-2')
        >>> mi_diccionario = {'nombre': 'Juan', 'edad': 30}
        >>> storage.guardar_diccionario(mi_diccionario, 'mi-bucket', 'ruta/en/bucket/')
        """
        # Obtener la fecha y hora actual en el formato YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Crear el nombre del archivo usando la fecha y hora
        nombre_archivo = f'02_App_reasoning_{timestamp}.json'
        # Convertir el diccionario a una cadena JSON
        contenido_json = json.dumps({'messages':diccionario_str}, indent=4)
        
        # Crear la clave S3 (ruta dentro del bucket)
        s3_key = f"{s3_key_prefix}{nombre_archivo}"

        try:
            # Subir el archivo al bucket S3
            self.s3.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=contenido_json,
                ContentType='application/json'
            )
            print(f"Archivo {nombre_archivo} guardado exitosamente en el bucket '{bucket_name}' en la ruta '{s3_key}'.")
        except NoCredentialsError:
            print("Error: No se encontraron las credenciales de AWS.")
        except Exception as e:
            print(f"Ocurrió un error al subir el archivo a S3: {e}")
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import re
# Cargar variables de entorno
load_dotenv()

# Configurar el motor de OpenAI 
engine = "gpt-3.5-turbo"
client = OpenAI(api_key=os.getenv("OPENAI.API_KEY"))
print(os.getenv("OPENAI.API_KEY"))


def fill_column_with_chatgpt(row, column_name):
    try:
        # Formatea la fila como texto para enviar a ChatGPT
        input_text = " ".join([f"{column}: {str(value)}" for column, value in row.items() if not pd.isna(value)])
        print(input_text)

        # Realiza la llamada a ChatGPT para obtener el relleno
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", 
                    "content": "Hola"}, 
                    {"role": "user", 
                    "content": f"Podrias por favor completar los registros que faltan: {input_text}"}]
        )
        
        # Mostrar solo la información clave de la respuesta
        print("API Response:")
        print(response)
        #print(response["choices"][0]["text"].strip())
        #print("Response time:")
        #print(response["usage"]["total_tokens"])
        # Extraer el texto relevante de la respuesta
        # Obtener el valor directamente desde la respuesta según la columna
         # Obtener el valor directamente desde la respuesta
       
        
    except Exception as e:
            # Manejar cualquier error que pueda ocurrir durante la llamada a la API
            print(f"Error en la llamada a la API de OpenAI: {e}")
            return None
    

def extract_release_date(choices_text):
    # Buscar el patrón de release_date en el texto
    match = re.search(r'release_date: (\d{4}-\d{2}-\d{2})', choices_text)

    # Verificar si se encontró el patrón
    if match:
        release_date = match.group(1)
        return release_date
    else:
        print("No se pudo encontrar release_date en el texto.")
        return None

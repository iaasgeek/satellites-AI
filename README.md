# Satellite AI Explorer

Satellite AI Explorer es una aplicación interactiva de visualización de datos de satélites construida con Streamlit y potenciada por Amazon Bedrock. Permite a los usuarios explorar una extensa base de datos de satélites, visualizar sus órbitas y obtener información detallada sobre cada uno de ellos.

## Características

- Visualización de estadísticas globales de satélites
- Búsqueda y visualización de información detallada de satélites individuales
- Visualización 3D de órbitas de satélites
- Chat IA para responder preguntas sobre satélites usando Amazon Bedrock

## Requisitos

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Boto3
- Cuenta de AWS con acceso a Amazon Bedrock

## Instalación

1. Clona este repositorio: \
git clone https://github.com/iaasgeek/satellites-AI.git \
cd satellites-AI

2. Crea un entorno virtual y actívalo: \
python -m venv venv \
source venv/bin/activate

3. Instala las dependencias: \
pip install -r requirements.txt

4. Configura tus credenciales de AWS para acceder a Amazon Bedrock.

## Uso

1. Navega al directorio `src`:
cd src

2. Ejecuta la aplicación Streamlit:
streamlit run satellites.py

3. Abre tu navegador y ve a `http://localhost:8501` para ver la aplicación.

## Estructura del Proyecto

satellite-explorer/\
│\
├── venv/\
├── data/\
│ └── UCS-Satellite-Database-1-1-2023.csv\
├── src/\
│ └── satellites.py\
├── requirements.txt\
└── README.md\

## Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de crear un pull request.

## Licencia

Este proyecto está licenciado bajo la Licencia Apache 2.0 - ver el archivo [LICENSE]([LICENSE](https://github.com/apache/.github/blob/main/LICENSE)) para más detalles.

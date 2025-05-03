# Gen-AI Music Project - Backend

Este es el componente backend del sistema de generación y procesamiento de música utilizando inteligencia artificial para armar playlists. El sistema permite cargar archivos de música como contexto, extraer características, almacenarlas en una base de datos de contexto, y generar playlists basadas en descripciones y solicitudes del usuario.

---

## Estructura del Proyecto


### Archivos y Módulos

#### 1. **`init.py`**
Este archivo contiene funciones para cargar datos de música, extraer características (como MFCCs), y configurar la base de datos de contexto de informacion.

- **Funciones:**
  - `load_music_data(file_paths)`: Carga archivos de música, extrae características MFCC y los convierte en objetos `Document` para su almacenamiento.
  - `summarize_text_and_tables(text, tables)`: Resume la informacion de contexto utilizando un modelo de lenguaje.
  - `initialize_retriever()`: Inicializa y devuelve un `MultiVectorRetriever` para la recuperación de documentos. Funciona como estructura de datos para poder utilizar en la base de datos de contexto

---

#### 2. **`chat.py`**
Este módulo maneja la interacción con el modelo de lenguaje para generar playlists basadas en descripciones y solicitudes.

- **Funciones:**
  - `chat_with_llm(retriever)`: Genera una playlist basada en descripciones de canciones y una solicitud del usuario. Utiliza un modelo de lenguaje para procesar el contexto y la pregunta.

---

#### 3. **`main.py`**
Entrypoint del backend. Define la estructura API REST utilizando Flask para interactuar con el sistema.

- **Rutas:**
  - `/playlist` (POST): Recibe una solicitud de playlist con descripciones de canciones y genera una respuesta utilizando el modelo de lenguaje.

- **Funciones:**
  - `create_playlist()`: Procesa la solicitud del usuario, carga los datos de música, y genera una playlist.

---

#### 4. **`RAG_with_streamlit.py`**
Este archivo combina técnicas de Recuperación-Augmentada-Generación (RAG) con Streamlit para procesar archivos de contexto de música y almacenarlos en la base de datos de contexto para luego utilizarlo combinado con el prompt.

- **Funciones:**
  - `_get_file_path(file_upload)`: Obtiene la ruta de un archivo cargado.
  - `generate_music_hash(music_path)`: Genera un hash único para un archivo de música.
  - `process_music(file_upload)`: Procesa un archivo de música, extrae características, y las almacena en una base de datos de contexto.

---

#### 5. **`utils.py`**
Este módulo contiene funciones auxiliares para manejar documentos, almacenar datos en la base de datos, y procesar la salida del recuperador.

- **Funciones:**
  - `store_docs_in_retriever(text, text_summary, table, table_summary, retriever)`: Almacena documentos y sus resúmenes en la base de datos.
  - `parse_retriver_output(data)`: Procesa la salida del recuperador para convertirla en un formato legible.

---

### Diagrama de Relaciones

```mermaid
graph TD
    A[main.py] -->|Llama a| B[chat.py]
    A -->|Llama a| C[utils.py]
    A -->|Llama a| D[init.py]
    B -->|Llama a| C[utils.py]
    D -->|Usa| E[utils.py]
    D -->|Usa| F[RAG_with_streamlit.py]
    F -->|Usa| C[utils.py]
    F -->|Usa| D[init.py]
    C -->|Usa| G[RedisStore y PGVector]
    D -->|Usa| G[RedisStore y PGVector]
```

![alt text](image-1.png)


### Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-repositorio/gen-ai-music-project.git
   cd gen-ai-music-project/backend
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar variables de entorno:**
   Crear un archivo `.env` con las siguientes variables:
   ```
   REDIS_URL=redis://localhost:6379
   COLLECTION_NAME=music_collection
   CONNECTION_STRING=postgresql://user:password@localhost/dbname
   PORT=8000
   ```

4. **Ejecutar el servidor:**
   ```bash
   python main.py
   ```

---

### Uso

- **Endpoint `/playlist`:**
  - Método: `POST`
  - Parámetros:
    - `prompt`: Solicitud del usuario para la playlist.
    - `music_files`: Lista de archivos de música.
  - Ejemplo de solicitud:
    ```json
    {
      "prompt": "Crea una playlist relajante para estudiar.",
      "music_files": ["path/to/song1.mp3", "path/to/song2.mp3"]
    }
    ```


## Base de datos vectorial para LLM

Para configurar la base de datos vectorial, puedes usar el siguiente comando Docker:
```
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 5432:5432 -d pgvector/pgvector:pg16
```

```
docker run --name gen-ai-music-backend-container -p 8080:8080 gen-ai-music-backend
```
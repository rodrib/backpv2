from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# Configurar CORS para permitir todas las solicitudes durante el desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI()

# Cargar el DataFrame desde el archivo CSV existente
df = pd.read_csv("bp2\df_completo.csv")  # Ajusta el nombre del archivo según sea necesario

# Ruta para la raíz
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Ruta para obtener datos del DataFrame
@app.get("/api/data")
def get_data():
    return df.to_dict(orient="records")

# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



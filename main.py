from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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



# Cargar el DataFrame desde el archivo CSV existente
df = pd.read_csv("df_completo1.csv")  # Ajusta el nombre del archivo según sea necesario

# Ruta para la raíz
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Ruta para obtener datos del DataFrame
@app.get("/api/data")
def get_data():
    return df.to_dict(orient="records")

###
#### Predicciones
from fastapi import FastAPI, Query
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# Crear DataFrame de ejemplo
data = {
    'RECIDIVA': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'DX1': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'EDAD': [45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 55, 60, 65, 70, 75],
    'GRADO1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    'HER21': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df_ejemplo = pd.DataFrame(data)

# Definir las características (X) y las etiquetas (y)
X = df_ejemplo[['EDAD', 'GRADO1']]
y_recidiva = df_ejemplo['RECIDIVA']
y_dx1 = df_ejemplo['DX1']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_recidiva, X_test_recidiva, y_train_recidiva, y_test_recidiva = train_test_split(X, y_recidiva, test_size=0.2, random_state=42)
X_train_dx1, X_test_dx1, y_train_dx1, y_test_dx1 = train_test_split(X, y_dx1, test_size=0.2, random_state=42)

# Ruta de la API para el modelo de recurrencia del cáncer
@app.get("/api/recidiva")
def predict_recidiva():
    # Modelo de Regresión Logística para predecir la recurrencia del cáncer
    model_recidiva = LogisticRegression()
    model_recidiva.fit(X_train_recidiva, y_train_recidiva)
    y_pred_recidiva = model_recidiva.predict(X_test_recidiva)
    report_recidiva = classification_report(y_test_recidiva, y_pred_recidiva, output_dict=True)
    accuracy_recidiva = model_recidiva.score(X_test_recidiva, y_test_recidiva)
    return {"report": report_recidiva, "accuracy": accuracy_recidiva}

# Ruta de la API para el modelo de diagnóstico del paciente
@app.get("/api/dx1")
def predict_dx1():
    # Modelo de Regresión Logística para predecir el diagnóstico del paciente
    model_dx1 = LogisticRegression()
    model_dx1.fit(X_train_dx1, y_train_dx1)
    y_pred_dx1 = model_dx1.predict(X_test_dx1)
    report_dx1 = classification_report(y_test_dx1, y_pred_dx1, output_dict=True)
    accuracy_dx1 = model_dx1.score(X_test_dx1, y_test_dx1)
    return {"report": report_dx1, "accuracy": accuracy_dx1}

# Ruta de la API para hacer predicciones con los modelos
from fastapi.encoders import jsonable_encoder

# Ruta de la API para hacer predicciones con los modelos
@app.get("/api/prediccion")
def hacer_prediccion(edad: int = Query(...), grado: int = Query(...)):
    # Aquí puedes utilizar los modelos entrenados para hacer las predicciones
    # Por ejemplo, podrías utilizar los modelos de recurrencia del cáncer y diagnóstico del paciente

    # Predicción de recurrencia del cáncer
    model_recidiva = LogisticRegression()
    model_recidiva.fit(X_train_recidiva, y_train_recidiva)
    recidiva_prediccion = model_recidiva.predict([[edad, grado]])
    
    # Predicción de diagnóstico del paciente
    model_dx1 = LogisticRegression()
    model_dx1.fit(X_train_dx1, y_train_dx1)
    dx1_prediccion = model_dx1.predict([[edad, grado]])

    # Formatear el resultado de la predicción
    resultado_prediccion = {
        "edad": int(edad),  # Convertir a int
        "grado": int(grado),  # Convertir a int
        "recidiva_prediccion": int(recidiva_prediccion[0]),  # Convertir a int
        "dx1_prediccion": int(dx1_prediccion[0])  # Convertir a int
    }

    # Convertir el resultado a JSON
    resultado_json = jsonable_encoder(resultado_prediccion)

    return resultado_json





# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



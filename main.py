import io
import sqlite3
import uvicorn
import numpy as np
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tf_keras as tfk

app = FastAPI(title="API PKLot - Vision Transformer + SQLite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- INICIALIZACIÓN DE BASE DE DATOS (SQLite) ---
def init_db():
    conn = sqlite3.connect("estacionamiento.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reportes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            archivo TEXT,
            total INTEGER,
            ocupados INTEGER,
            disponibles INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db() # Se ejecuta al arrancar el servidor

print("Cargando el Vision Transformer (ViT-B/16)...")
model = tfk.models.load_model('modelo_vit_oficial')
print("¡Modelo cargado en memoria y Base de Datos lista!")

@app.post("/predict_grid")
async def predecir_estacionamiento_grid(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    filas = 16
    columnas = 16
    ancho_celda = image.width // columnas
    alto_celda = image.height // filas
    
    parches = []
    coordenadas = []
    
    for f in range(filas):
        for c in range(columnas):
            left = c * ancho_celda
            top = f * alto_celda
            right = (c + 1) * ancho_celda
            bottom = (f + 1) * alto_celda
            
            parche = image.crop((left, top, right, bottom))
            parche = parche.resize((224, 224))
            parches.append(np.array(parche) / 255.0)
            coordenadas.append({"fila": f, "columna": c})
            
    batch_tensor = np.array(parches)
    predicciones = model.predict(batch_tensor, batch_size=64)
    
    resultados = []
    ocupados = 0
    disponibles = 0
    
    for idx, pred in enumerate(predicciones):
        probabilidad = float(pred[0])
        estado = "Ocupado" if probabilidad > 0.5 else "Disponible"
        
        if estado == "Ocupado": ocupados += 1
        else: disponibles += 1
            
        resultados.append({
            "fila": coordenadas[idx]["fila"],
            "columna": coordenadas[idx]["columna"],
            "estado": estado,
            "confianza": round(probabilidad if estado == "Ocupado" else (1 - probabilidad), 2)
        })
    
    # --- GUARDAR EN BASE DE DATOS ---
    conn = sqlite3.connect("estacionamiento.db")
    cursor = conn.cursor()
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO reportes (fecha, archivo, total, ocupados, disponibles) VALUES (?, ?, ?, ?, ?)",
                   (fecha_actual, file.filename, filas*columnas, ocupados, disponibles))
    conn.commit()
    conn.close()
        
    return {
        "grid_size": {"filas": filas, "columnas": columnas},
        "stats": {"total": filas * columnas, "ocupados": ocupados, "disponibles": disponibles},
        "celdas": resultados
    }

# --- NUEVO ENDPOINT PARA CONSULTAR REPORTES ---
@app.get("/reportes")
def obtener_reportes():
    conn = sqlite3.connect("estacionamiento.db")
    cursor = conn.cursor()
    # Traemos los últimos 10 registros
    cursor.execute("SELECT fecha, archivo, ocupados, disponibles FROM reportes ORDER BY id DESC LIMIT 10")
    filas = cursor.fetchall()
    conn.close()
    
    reportes = []
    for f in filas:
        reportes.append({
            "fecha": f[0],
            "archivo": f[1],
            "ocupados": f[2],
            "disponibles": f[3]
        })
    return reportes

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
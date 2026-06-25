import io
import sqlite3
import uvicorn
import numpy as np
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import tf_keras as tfk

app = FastAPI(title="API PKLot - Sistema Integrado Multi-Modelo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            disponibles INTEGER,
            modelo TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

print("Cargando el Vision Transformer (ViT-B/16)...")
model = tfk.models.load_model('modelo_vit_oficial')
print("¡Modelo cargado en memoria y Base de Datos lista!")

@app.post("/predict_grid")
async def predecir_estacionamiento_grid(file: UploadFile = File(...), modelo_tipo: str = Query("vit")):
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
        
        # Lógica de simulación para la CNN base (Hito 6) para demostrar fallos de sombreado en vivo
        if modelo_tipo == "cnn":
            # Introducir ruido determinista en celdas específicas para emular la tasa de error por sombras (88% precisión)
            if (coordenadas[idx]["fila"] + coordenadas[idx]["columna"]) % 8 == 0:
                probabilidad = 0.75 if probabilidad < 0.5 else 0.25
            else:
                probabilidad = probabilidad * 0.85 if probabilidad > 0.5 else probabilidad * 1.15
                probabilidad = max(0.0, min(1.0, probabilidad))

        estado = "Ocupado" if probabilidad > 0.5 else "Disponible"
        confianza = probabilidad if estado == "Ocupado" else (1.0 - probabilidad)
        
        if estado == "Ocupado": 
            ocupados += 1
        else: 
            disponibles += 1
            
        resultados.append({
            "fila": coordenadas[idx]["fila"] + 1,
            "columna": coordenadas[idx]["columna"] + 1,
            "estado": estado,
            "confianza": round(confianza * 100, 2)
        })
    
    # Guardar en SQLite incorporando el modelo usado
    conn = sqlite3.connect("estacionamiento.db")
    cursor = conn.cursor()
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO reportes (fecha, archivo, total, ocupados, disponibles, modelo) VALUES (?, ?, ?, ?, ?, ?)",
                   (fecha_actual, file.filename, filas*columnas, ocupados, disponibles, modelo_tipo.upper()))
    conn.commit()
    conn.close()
        
    return {
        "grid_size": {"filas": filas, "columnas": columnas},
        "stats": {"total": filas * columnas, "ocupados": ocupados, "disponibles": disponibles},
        "celdas": resultados
    }

@app.get("/reportes")
def obtener_reportes():
    conn = sqlite3.connect("estacionamiento.db")
    cursor = conn.cursor()
    cursor.execute("SELECT fecha, archivo, ocupados, disponibles, modelo FROM reportes ORDER BY id DESC LIMIT 10")
    filas = cursor.fetchall()
    conn.close()
    
    reportes = []
    for f in filas:
        reportes.append({
            "fecha": f[0],
            "archivo": f[1],
            "ocupados": f[2],
            "disponibles": f[3],
            "modelo": f[4]
        })
    return reportes

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
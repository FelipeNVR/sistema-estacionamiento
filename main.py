import io
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tf_keras as tfk

app = FastAPI(title="API PKLot - Vision Transformer High-Density Grid")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando el Vision Transformer (ViT-B/16)...")
model = tfk.models.load_model('modelo_vit_oficial')
print("¡Modelo cargado en memoria!")

@app.post("/predict_grid")
async def predecir_estacionamiento_grid(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Alta densidad: 16 filas x 16 columnas = 256 micro-zonas
    filas = 16
    columnas = 16
    ancho_celda = image.width // columnas
    alto_celda = image.height // filas
    
    parches = []
    coordenadas = []
    
    # 1. Recorte de alta resolución
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
            
    # 2. Inferencia masiva vectorizada (256 tensores procesados en un solo bloque)
    batch_tensor = np.array(parches)
    predicciones = model.predict(batch_tensor, batch_size=64)
    
    resultados = []
    ocupados = 0
    disponibles = 0
    
    for idx, pred in enumerate(predicciones):
        probabilidad = float(pred[0])
        estado = "Ocupado" if probabilidad > 0.5 else "Disponible"
        
        if estado == "Ocupado":
            ocupados += 1
        else:
            disponibles += 1
            
        resultados.append({
            "fila": coordenadas[idx]["fila"],
            "columna": coordenadas[idx]["columna"],
            "estado": estado,
            "confianza": round(probabilidad if estado == "Ocupado" else (1 - probabilidad), 2)
        })
        
    return {
        "grid_size": {"filas": filas, "columnas": columnas},
        "stats": {"total": filas * columnas, "ocupados": ocupados, "disponibles": disponibles},
        "celdas": resultados
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
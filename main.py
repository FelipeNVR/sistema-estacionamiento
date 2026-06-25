import io
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tf_keras as tfk
import tensorflow_hub as hub

app = FastAPI(title="API PKLot - Vision Transformer")

# Permitir que el frontend (Vercel) se comunique con este backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando el Vision Transformer (ViT-B/16)... Esto tomará unos segundos.")
# Cargamos la carpeta del modelo SavedModel
model = tfk.models.load_model('modelo_vit_oficial')
print("¡Modelo cargado en memoria!")

def preprocesar_imagen(image_bytes):
    # Convertir bytes a imagen, asegurar que sea RGB, redimensionar a 224x224 y normalizar
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0) # Añadir dimensión de lote (1, 224, 224, 3)

@app.post("/predict")
async def predecir_estacionamiento(file: UploadFile = File(...)):
    contents = await file.read()
    img_tensor = preprocesar_imagen(contents)
    
    # Inferencia
    prediccion = model.predict(img_tensor)[0][0]
    
    # Lógica de negocio (Umbral > 0.5 es Ocupado, modelo Sigmoide)
    estado = "Ocupado" if prediccion > 0.5 else "Disponible"
    confianza = float(prediccion) if estado == "Ocupado" else float(1 - prediccion)
    
    return {
        "estado": estado,
        "confianza": round(confianza * 100, 2),
        "filename": file.filename
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
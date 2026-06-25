# 🅿️ Sistema Inteligente de Monitoreo de Estacionamientos

Este proyecto es una solución de Visión Artificial para la detección en tiempo real de la disponibilidad de espacios de estacionamiento. Evoluciona desde una arquitectura CNN base hacia un mecanismo de atención **Vision Transformer (ViT-B/16)** para superar las limitaciones geométricas y los falsos positivos causados por ilusiones ópticas.

## 🚀 Arquitectura del Sistema
El proyecto está dividido en microservicios:
* **Backend (API):** Construido con `FastAPI` y empaquetado en `Docker`. Procesa recortes de la imagen original en lotes masivos (matriz 16x16) utilizando un modelo ViT preentrenado.
* **Frontend (Interfaz):** Aplicación web estática (HTML/TailwindCSS) responsiva que se comunica de forma asíncrona con el contenedor Docker.

## 📊 Dataset y Rendimiento
El modelo fue entrenado con el dataset **PKLot** (1.1 millones de imágenes). Para evitar el *Data Leakage*, se implementó una **Validación Cruzada por Dominio**:
* **Entrenamiento:** Cámaras PUCPR y UFPR04.
* **Validación (Blind Test):** Cámara UFPR05.

**Comparativa Experimental (Dominio Cruzado):**
* **Accuracy:** CNN (92.19%) ➔ ViT (98.87%)
* **Precision:** CNN (88.64%) ➔ ViT (98.42%)
* **Recall:** CNN (99.37%) ➔ ViT (99.68%)
* **F1-Score:** CNN (93.70%) ➔ ViT (99.05%)

## ⚙️ Instrucciones de Despliegue
Para ejecutar este proyecto utilizando la contenedorización:

1. Clonar el repositorio:
   git clone https://github.com/FelipeNVR/sistema-estacionamiento.git
2. Entrar al directorio:
   cd sistema-estacionamiento
3. Construir la imagen Docker:
   docker build -t api-estacionamiento .
4. Levantar el microservicio:
   docker run -p 8000:8000 api-estacionamiento
5. Abrir la interfaz web (`index.html`) conectada al puerto 8000 local.
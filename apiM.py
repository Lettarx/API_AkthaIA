from fastapi import FastAPI, UploadFile, File, Form
from src.functions.procesar import procesarImg, evaluarSimilitud, procesarImgBase64
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:4321"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class ImageInput(BaseModel):
    imagen_base64: str

@app.post("/CriteriosEvaluacion/")
async def evaluar_imagen(file: UploadFile = File(...)):
    img = await file.read()
    return procesarImg(img)

@app.post('/similitud/')
async def evaluar_imagen(file1: UploadFile = File(None), file2: UploadFile = File(None)):
    img1 = await file1.read()
    img2 = await file2.read()
    return evaluarSimilitud(img1, img2)
    
@app.post("/CriteriosEvaluacionBase64/")
async def evaluar_imagen(image_input: ImageInput):
    return procesarImgBase64(image_input.imagen_base64)

if __name__ == "__main__":
    import uvicorn
    # Ejecuta el servidor usando Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
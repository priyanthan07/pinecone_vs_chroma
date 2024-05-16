from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os

from app import router as VB_router


from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

app.include_router(VB_router)

origins = [
    "http://localhost",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", host='localhost', port= 8000, reload=True)
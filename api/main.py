from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"AAAAAAAAAAAH !! L'elephant": f"barit, cours {os.environ.get('NAME')}, {os.environ.get('OTHERNAME')} te poursuit !!! 😱"}

@app.get("/healthcheck")
def is_alive():
    return {"status": "ok"}

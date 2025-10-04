
from fastapi import FastAPI

# 1. Create an instance of the FastAPI application
app = FastAPI()

# 2. Define a "path operation decorator"
@app.get("/")
async def root():
    # 3. Return a Python dictionary
    return {"message": "Hello, World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id, "description": f"This is item number {item_id}"}
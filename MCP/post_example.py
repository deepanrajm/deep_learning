# square_app.py

from fastapi import FastAPI
from pydantic import BaseModel

# 1. Define the structure of the incoming data.
#    We expect a JSON object with a single key "number".
class NumberInput(BaseModel):
    number: float  # Using float allows for integers (like 5) and decimals (like 2.5)

# 2. Create the FastAPI app
app = FastAPI()

# 3. Create the POST endpoint
@app.post("/calculate/square")
async def calculate_square(data: NumberInput):
    """
    Receives a JSON object with a number, calculates its square,
    and returns the result.
    """
    # Get the number from the input data
    received_number = data.number
    
    # Calculate the square
    squared_result = received_number ** 2
    
    # Return a JSON response with the original number and the result
    return {"original_number": received_number, "squared_result": squared_result}
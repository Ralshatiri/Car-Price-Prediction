from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
import pathlib
import pandas as pd
import numpy as np
from pydantic import BaseModel
import shap
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY")) 

parent_path = pathlib.Path(__file__).parent
model_path = f"{parent_path}/car_price_model/car_price_model.pkl"
preprocessor_path = f"{parent_path}/car_price_model/preprocessor.pkl"
background_path = f"{parent_path}/car_price_model/background_data.pkl"
feature_names_path = f"{parent_path}/car_price_model/feature_names.pkl"

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
background_data = joblib.load(background_path)
feature_names = joblib.load(feature_names_path)

explainer = shap.Explainer(model.estimator_, background_data)

class carinput(BaseModel):
    Make: str
    Type: str
    Region: str
    Gear_Type: int
    Origin: str
    Options: str
    Engine_Size: float
    Mileage: int
    Negotiable: int
    CarAge: int

app = FastAPI()

url_list = [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5500"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins = url_list,
  allow_methods = ["*"],
  allow_headers = ["*"]
)

def format_feature_name(name: str):
    name = name.replace("cat__", "").replace("num__", "")
    if "_" in name:
        field, value = name.split("_", 1)
        return f"{field} ({value})"
    return name

def generate_reasoning(shap_values, feature_names, top_k=4):
    values = shap_values.values[0]
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    reasoning = []
    used_fields = set()

    for feature, value in pairs:
        if abs(value) < 1e-8:
            continue

        readable = format_feature_name(feature)
        field = readable.split(" (")[0]
    
        if "(" in readable:
            readable = readable.split("(")[1].replace(")", "").strip()
        readable = readable.replace("CarAge", "car age").replace("Mileage", "mileage").replace("Engine", "engine size")

        if field in used_fields:
            continue

        direction = "increased" if value > 0 else "decreased"
        action = "adds value" if value > 0 else "reduces value"
        reasoning.append(f"{readable} {action}")
        
        used_fields.add(field)

        if len(reasoning) == top_k:
            break

    return reasoning

def generate_summary(price, reasoning):
    if not reasoning:
        return f"The estimated price is {price:,} SAR."
    prompt = f"""
            You are a car pricing expert. Write ONE short paragraph explaining a used car price estimate.

            STRICT RULES:
            - Maximum 2 sentences
            - Do NOT start with "Based on"
            - Do NOT use "we", "our", "I", "you"
            - Do NOT mention SHAP, machine learning, or feature engineering
            - Do NOT copy the factor phrases word for word
            - Write the car naturally (e.g. "Audi A4" not "Make (Audi)")
            - Sound like a market expert, not a chatbot

            EXAMPLE OF GOOD OUTPUT:
            "The Audi A4 is estimated at 55,977 SAR. Its premium brand value and older age push the price up, while the low mileage slightly offsets it."

            Predicted price: {price} SAR

            Factors:
            {chr(10).join(f"- {item}" for item in reasoning)}

            Now write the explanation:
            """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a car pricing expert. Never use 'we', 'our', 'I', or 'you'. Always write in third person about the car only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        summary = response.choices[0].message.content.strip()
        if summary:
            return summary
    except Exception:
        pass

    if len(reasoning) == 1:
        return f"The estimated price is {price:,} SAR. The main factor is that {reasoning[0].lower()}."

    text = ", ".join(r.lower() for r in reasoning[:-1]) + f", and {reasoning[-1].lower()}"
    return f"The estimated price is {price:,} SAR. The most important factors are that {text}."

@app.post("/predict")
def prediction(input_data: carinput):
    input_df = pd.DataFrame([input_data.model_dump()])
    transformed_data = preprocessor.transform(input_df)

    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    prediction_log = model.predict(transformed_data)
    predicted_price = round(float(np.exp(prediction_log[0])))

    selected_data = transformed_data[:, model.support_]
    shap_values = explainer(selected_data)

    reasoning = generate_reasoning(shap_values, feature_names)
    summary = generate_summary(predicted_price, reasoning)

    return {
        "predicted_price": predicted_price,
        "reasoning": reasoning,
        "summary": summary
    }

app.mount("/", StaticFiles(directory=f"{parent_path.parent}/frontend", html=True), name="frontend")
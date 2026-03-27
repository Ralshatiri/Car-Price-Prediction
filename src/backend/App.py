from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
import pathlib
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Literal
import shap
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

parent_path = pathlib.Path(__file__).parent

regression_model_path = f"{parent_path}/car_price_model/linear_regression_model/linear_model.pkl"
regression_preprocessor_path = f"{parent_path}/car_price_model/linear_regression_model/linear_preprocessor.pkl"
regression_background_path = f"{parent_path}/car_price_model/linear_regression_model/linear_background_data.pkl"
regression_feature_names_path= f"{parent_path}/car_price_model/linear_regression_model/linear_feature_names.pkl"

xgboost_model_path = f"{parent_path}/car_price_model/xgboost_model/xgboost_model.pkl"
xgboost_preprocessor_path = f"{parent_path}/car_price_model/xgboost_model/xgboost_preprocessor.pkl"
xgboost_background_path = f"{parent_path}/car_price_model/xgboost_model/xgboost_background_data.pkl"
xgboost_feature_names_path= f"{parent_path}/car_price_model/xgboost_model/xgboost_feature_names.pkl"

model_list = [joblib.load(regression_model_path),joblib.load(xgboost_model_path)]
preprocessor_list = [joblib.load(regression_preprocessor_path),joblib.load(xgboost_preprocessor_path)]
background_data_list = [joblib.load(regression_background_path),joblib.load(xgboost_background_path)]
feature_names_list = [joblib.load(regression_feature_names_path),joblib.load(xgboost_feature_names_path)]

print("=== LINEAR feature names (pkl) ===")
print(feature_names_list[0][:10])

print("=== XGBOOST feature names (pkl) ===")
print(feature_names_list[1][:10])

print("=== LINEAR preprocessor output names ===")
print(list(preprocessor_list[0].get_feature_names_out())[:10])

print("=== XGBOOST preprocessor output names ===")

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
    Listed_Price: float | None = None
    model: Literal[0, 1]

app = FastAPI()

url_list = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=url_list,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/predict")
def prediction(input_data: carinput):
    model_index = input_data.model
    input_df = pd.DataFrame([input_data.model_dump(exclude={'model'})])
    transformed_data = preprocessor_list[model_index].transform(input_df)

    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    prediction_log = model_list[model_index].predict(transformed_data)
    predicted_price = round(float(np.exp(prediction_log[0])))

    if model_index == 0:  # RFECV linear model
        model_for_shap = model_list[model_index].estimator_
        selected_data = transformed_data[:, model_list[model_index].support_]
    else:  # plain XGBRegressor
        model_for_shap = model_list[model_index]
        selected_data = transformed_data

    if model_index == 0:  # Linear RFECV
        explainer = shap.LinearExplainer(
            model_for_shap, 
            background_data_list[0]
        )
        shap_values = explainer(selected_data)

    else:  # XGBoost
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer(selected_data)



    def generate_reasoning(shap_values, feature_names, top_k=4, decoded_map=None):
        values = shap_values.values[0]
        pairs = list(zip(feature_names, values))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        human_names = {
            "Engine_Size": "engine size",
            "CarAge": "car age",
            "Mileage": "mileage",
            "Gear_Type": "transmission",
            "Negotiable": "negotiable",
            "Make": "make",
            "Type": "type",
            "Region": "region",
            "Origin": "origin",
            "Options": "options",
        }

        reasoning = []
        used_fields = set()

        for feature, value in pairs:
            if abs(value) < 1e-8 or len(reasoning) == top_k:
                break

            if decoded_map and feature in decoded_map:
                
                readable = decoded_map[feature]
                field = feature
            else:
              
                name = feature.replace("cat__", "").replace("num__", "")
                if "_" in name:
                    field, readable = name.split("_", 1)
                else:
                    field = name
                    readable = human_names.get(name, name.lower())
                readable = (readable
                    .replace("CarAge", "car age")
                    .replace("Mileage", "mileage")
                    .replace("Engine_Size", "engine size"))

            if field in used_fields:
                continue

            action = "adds value" if value > 0 else "reduces value"
            reasoning.append(f"{readable} {action}")
            used_fields.add(field)

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
    - Write the car naturally
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

    def get_recommendation(predicted_price, listed_price):
        diff_percentage = ((listed_price - predicted_price) / predicted_price) * 100

        if diff_percentage > 30:
            recommendation = "The car is significantly overpriced. It is not recommended to buy it."
        elif diff_percentage > 15:
            recommendation = "The car is overpriced. Negotiation is recommended before buying."
        elif diff_percentage >= -10:
            recommendation = "The listed price is fair and close to the expected market value."
        else:
            recommendation = "The car appears to be a good deal because it is priced below the expected value."

        return recommendation, round(diff_percentage, 2)

    def get_recommendation_reason(predicted_price, listed_price):
        diff_percentage = round(((listed_price - predicted_price) / predicted_price) * 100, 2)

        if diff_percentage > 0:
            return f"The listed price is {diff_percentage}% higher than the estimated market value."
        elif diff_percentage < 0:
            return f"The listed price is {abs(diff_percentage)}% lower than the estimated market value."
        else:
            return "The listed price matches the estimated market value exactly."

    if model_index == 1:
        ordinal_encoder = preprocessor_list[1].named_transformers_['cat']
        cat_cols = ["Make", "Type", "Region", "Origin", "Options", "Gear_Type"]
        decoded_map = {}
        for i, col in enumerate(cat_cols):
            encoded_val = transformed_data[0][3 + i]
            categories = ordinal_encoder.categories_[i]
            idx = int(round(encoded_val))
            decoded_map[col] = categories[idx] if 0 <= idx < len(categories) else col
        decoded_map["Engine_Size"] = f"{input_data.Engine_Size}L engine"
        decoded_map["Mileage"]     = f"{input_data.Mileage:,} km mileage"
        decoded_map["CarAge"]      = f"{input_data.CarAge} year old car" if input_data.CarAge > 0 else "brand new car"
        decoded_map["Negotiable"]  = "negotiable listing" if input_data.Negotiable else "non-negotiable listing"
        reasoning = generate_reasoning(shap_values, feature_names_list[1], decoded_map=decoded_map)
    else:
        reasoning = generate_reasoning(shap_values, feature_names_list[0])

    summary = generate_summary(predicted_price, reasoning)

    response = {
        "predicted_price": predicted_price,
        "reasoning": reasoning,
        "summary": summary,
        "recommendation": None,
        "recommendation_reason": None,
        "difference_percentage": None
    }

    if input_data.Listed_Price is not None:
        recommendation, diff_percentage = get_recommendation(predicted_price, input_data.Listed_Price)
        recommendation_reason = get_recommendation_reason(predicted_price, input_data.Listed_Price)

        response["recommendation"] = recommendation
        response["recommendation_reason"] = recommendation_reason
        response["difference_percentage"] = diff_percentage

    return response
    
frontend_path = pathlib.Path(__file__).parent.parent / "frontend"
print(f"Frontend exists: {frontend_path.exists()}")
print(f"Frontend path: {frontend_path}")

if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
else:
    print("WARNING: Frontend folder not found!")

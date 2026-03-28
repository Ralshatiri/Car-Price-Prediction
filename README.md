# Project Overview
This project aims to predict used car prices using Machine Learning techniques.
The dataset was collected, cleaned, explored, and preprocessed before training regression models.
The main objective is to build reliable models capable of estimating car prices based on features such as mileage, brand, and other attributes. Two models were developed and evaluated: a **Multi-Linear Regression** model and an **XGBoost Regression** model.

🌐 ** Website:** [Car Price Prediction](https://car-price-prediction-028b.onrender.com)

---

## Folder & File Description

### 📂 Dataset/
- **Processd_dataset.csv**  
  Intermediate dataset after processing using the `Main.py` file.
- **Final_processed_dataset.csv**  
  Fully cleaned and preprocessed dataset.

---

### 📂 Data Processing/
- **01_Data_Exploration_and_Preprocessing.ipynb**  
  This notebook performs preprocessing steps after visualization.
- **Main.py**  
  A Python script that processes and cleans the dataset outside the notebook environment.

---

### 📂 Model Development/
- **Linear_Model_Development_and_Evaluation.ipynb**  
  Develops and evaluates a Multi-Linear Regression model to predict car prices based on features such as mileage, brand, and other attributes.
- **Xgboost_Model_Development_and_Evaluation.ipynb**  
  Develops and evaluates an XGBoost Regression model.
---

### 📂 src/
- **backend/**  
  Server-side code handling API requests and model inference.
- **frontend/**  
  Client-side code for the web interface where users can input car features and receive price predictions.

---

## Tools Used
- Excel (initial preprocessing)
- Python
- Render
- Java script
- html and css
- Colab Notebook

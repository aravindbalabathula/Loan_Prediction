from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

# Load model and scaler
model = joblib.load('loan_prediction_model.pkl')
scaler = joblib.load('vector.pkl')

# Initialize app
app = FastAPI(
    title="Smart Loan Recovery System",
    description=(
        "🚀 This API predicts loan approval status based on applicant financial "
        "and demographic details using a trained machine learning model."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Numerical columns to scale
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Define request schema with field descriptions and example values
class LoanApproval(BaseModel):
    Gender: Literal["Male", "Female"] = Field(..., example="Male", description="Gender of the applicant")
    Married: Literal["Yes", "No"] = Field(..., example="Yes", description="Marital status of the applicant")
    Dependents: float = Field(..., example=2, description="Number of dependents")
    Education: Literal["Graduate", "Not Graduate"] = Field(..., example="Graduate", description="Education status")
    Self_Employed: Literal["Yes", "No"] = Field(..., example="No", description="Is the applicant self-employed?")
    ApplicantIncome: float = Field(..., example=5000, description="Applicant's monthly income")
    CoapplicantIncome: float = Field(..., example=2000, description="Co-applicant's monthly income")
    LoanAmount: float = Field(..., example=150, description="Total loan amount applied for")
    Loan_Amount_Term: float = Field(..., example=360, description="Loan repayment term (in months)")
    Credit_History: Literal[0, 1] = Field(..., example=1, description="Credit history (1 = good, 0 = poor)")
    Property_Area: Literal["Urban", "Semiurban", "Rural"] = Field(..., example="Urban", description="Property area")

# Category mapping dictionaries
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'Yes': 1, 'No': 0}
property_map = {'Urban': 1, 'Semiurban': 2, 'Rural': 0}

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Smart Loan Recovery Prediction API 🎯",
        "usage": "Use POST /predict to get loan approval predictions.",
        "docs": "Visit /docs for interactive documentation."
    }

@app.post("/predict", tags=["Prediction"])
async def predict_loan_status(application: LoanApproval):
    """
    Predict the loan approval status of an applicant.
    """
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([application.model_dump()])

        # Encode categorical variables
        input_data['Gender'] = input_data['Gender'].map(gender_map)
        input_data['Married'] = input_data['Married'].map(married_map)
        input_data['Education'] = input_data['Education'].map(education_map)
        input_data['Self_Employed'] = input_data['Self_Employed'].map(self_employed_map)
        input_data['Property_Area'] = input_data['Property_Area'].map(property_map)

        # Scale numerical columns
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        # Make prediction
        prediction = model.predict(input_data)[0]
        status = "Approved" if prediction == 1 else "Not Approved"

        return {
            "prediction": status,
            "details": {
                "message": f"The applicant's loan is likely to be **{status}**.",
                "model_version": "v1.0",
                "developer": "Aravind Balabathula"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

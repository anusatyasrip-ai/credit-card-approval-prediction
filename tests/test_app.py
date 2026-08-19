import pytest
import json
from app import app
from services.predictor import predictor
from services.database import get_history, get_dashboard_stats, init_db

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        init_db()
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Credit Card Approval Prediction System" in response.data

def test_predict_page_get(client):
    response = client.get('/predict')
    assert response.status_code == 200
    assert b"Applicant Evaluation Form" in response.data

def test_history_page(client):
    response = client.get('/history')
    assert response.status_code == 200
    assert b"Prediction History Log" in response.data

def test_about_page(client):
    response = client.get('/about')
    assert response.status_code == 200
    assert b"About System Architecture" in response.data

def test_valid_api_predict(client):
    payload = {
        "gender": "Female",
        "income_type": "Salary",
        "annual_income": 85000,
        "employment_years": 7.0,
        "education_level": "Higher education",
        "employment_status": "Employed",
        "existing_loan_balance": 4000,
        "credit_inquiries": 1,
        "credit_history_years": 10.0,
        "payment_status": "No Past Due"
    }

    response = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "success"
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["Approved", "Rejected"]
    assert 0.0 <= data["approval_probability"] <= 1.0
    assert "model" in data

def test_invalid_api_predict_negative_income(client):
    payload = {
        "gender": "Female",
        "income_type": "Salary",
        "annual_income": -5000,
        "employment_years": 5.0,
        "education_level": "Higher education",
        "employment_status": "Employed",
        "existing_loan_balance": 4000,
        "credit_inquiries": 1,
        "credit_history_years": 10.0,
        "payment_status": "No Past Due"
    }

    response = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 400

    data = response.get_json()
    assert data["status"] == "error"
    assert "Annual income cannot be negative" in data["message"]

def test_form_prediction_post(client):
    form_data = {
        "gender": "Male",
        "income_type": "Salary",
        "annual_income": "60000",
        "employment_years": "4.5",
        "education_level": "Higher education",
        "employment_status": "Employed",
        "existing_loan_balance": "8000",
        "credit_inquiries": "1",
        "credit_history_years": "6.0",
        "payment_status": "No Past Due"
    }

    response = client.post('/predict', data=form_data)
    assert response.status_code == 200
    assert b"APPLICATION APPROVED" in response.data or b"APPLICATION REJECTED" in response.data

def test_database_persistence():
    stats = get_dashboard_stats()
    assert stats["total"] >= 0
    assert stats["approved"] >= 0
    assert stats["rejected"] >= 0

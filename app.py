import json
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
import config
from services.predictor import predictor
from services.database import save_prediction, get_history, get_dashboard_stats, init_db

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "supersecretkey_change_in_production")

# Ensure SQLite DB table exists at app initialization
init_db()

@app.route('/')
def index():
    stats = get_dashboard_stats()
    stats["model_name"] = predictor.model_name
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict_post():
    try:
        raw_input = {
            "gender": request.form.get("gender"),
            "income_type": request.form.get("income_type"),
            "annual_income": request.form.get("annual_income"),
            "employment_years": request.form.get("employment_years"),
            "education_level": request.form.get("education_level"),
            "employment_status": request.form.get("employment_status"),
            "existing_loan_balance": request.form.get("existing_loan_balance"),
            "credit_inquiries": request.form.get("credit_inquiries"),
            "credit_history_years": request.form.get("credit_history_years"),
            "payment_status": request.form.get("payment_status")
        }

        result = predictor.predict(raw_input)
        
        # Store prediction audit trail in SQLite
        save_prediction(raw_input, result)

        return render_template('result.html', result=result, input_data=raw_input)

    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for('predict_page'))
    except Exception as e:
        flash(f"An unexpected error occurred during prediction: {str(e)}", "error")
        return redirect(url_for('predict_page'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not request.is_json:
        return jsonify({"error": "Request payload must be JSON."}), 400

    raw_input = request.get_json()

    try:
        result = predictor.predict(raw_input)
        
        # Save to database
        save_prediction(raw_input, result)

        return jsonify({
            "status": "success",
            "prediction": result["prediction"],
            "label": result["label"],
            "approval_probability": result["approval_probability"],
            "approval_percentage": result["approval_percentage"],
            "model": result["model"]
        }), 200

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server processing error: {str(e)}"}), 500

@app.route('/history', methods=['GET'])
def history():
    search_query = request.args.get('q', '').strip()
    records = get_history(limit=100, search_query=search_query if search_query else None)
    return render_template('history.html', records=records, search_query=search_query)

@app.route('/about', methods=['GET'])
def about():
    metrics_data = None
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

    return render_template('about.html', metrics=metrics_data)

@app.route('/visualizations/<path:filename>')
def get_visualization(filename):
    return send_from_directory(config.VISUALIZATIONS_DIR, filename)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='127.0.0.1', port=port, debug=True)

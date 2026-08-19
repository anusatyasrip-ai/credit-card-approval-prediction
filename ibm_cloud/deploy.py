import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

def deploy_to_ibm_watson():
    print("=" * 60)
    print(" IBM WATSON MACHINE LEARNING DEPLOYMENT SCRIPT ")
    print("=" * 60)

    api_key = os.getenv("IBM_WML_API_KEY")
    project_id = os.getenv("IBM_WML_PROJECT_ID")
    space_id = os.getenv("IBM_WML_SPACE_ID")
    wml_url = os.getenv("IBM_WML_URL", "https://us-south.ml.cloud.ibm.com")

    print("\n--- 1. CHECKING IBM CLOUD CREDENTIALS ---")
    print(f"IBM WML URL: {wml_url}")
    print(f"API Key Provided: {'Yes' if api_key and 'your_' not in api_key else 'No (Placeholder or missing)'}")
    print(f"Project ID:      {project_id if project_id else 'Not set'}")
    print(f"Space ID:        {space_id if space_id else 'Not set'}")

    if not api_key or "your_" in api_key:
        print("\n [WARNING] Real IBM Cloud API Key is missing in `.env` file.")
        print(" Please update `.env` with your actual IBM Cloud API Key and Space ID.")
        print(" Instructions:")
        print("  1. Copy `.env.example` to `.env`")
        print("  2. Set IBM_WML_API_KEY=your_actual_key")
        print("  3. Set IBM_WML_SPACE_ID=your_deployment_space_id")
        print("  4. Re-run: python ibm_cloud/deploy.py\n")
        return

    try:
        from ibm_watson_machine_learning import APIClient
    except ImportError:
        print("\n [INFO] `ibm-watson-machine-learning` SDK is not installed or requires Python <3.13.")
        print(" Production Deployment Logic (cURL / REST API Fallback):")
        print("  1. Authenticate with IAM Token endpoint: https://iam.cloud.ibm.com/identity/token")
        print("  2. POST payload to IBM Watson ML Model Storage endpoint")
        print("  3. Create online deployment with hardware spec `S` or `M`\n")
        return

    wml_credentials = {
        "url": wml_url,
        "apikey": api_key
    }

    client = APIClient(wml_credentials)

    if space_id:
        client.set.default_space(space_id)
        print(f"\nConnected to IBM Deployment Space: {space_id}")
    elif project_id:
        client.set.default_project(project_id)
        print(f"\nConnected to IBM Project: {project_id}")
    else:
        print("\n [ERROR] Either IBM_WML_SPACE_ID or IBM_WML_PROJECT_ID must be specified in `.env`.")
        return

    if not MODEL_PATH.exists():
        print(f"\n [ERROR] Model artifact missing at {MODEL_PATH}. Run `python train.py` first.")
        return

    print("\n--- 2. UPLOADING MODEL TO IBM WATSON ML ---")
    model_metadata = {
        client.repository.ModelMetaNames.NAME: "Credit_Card_Approval_Classifier",
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: client.software_specifications.get_id_by_name("runtime-22.2-py3.10")
    }

    model_details = client.repository.store_model(
        model=str(MODEL_PATH),
        meta_props=model_metadata
    )

    model_uid = client.repository.get_model_id(model_details)
    print(f" Successfully stored model in IBM Watson ML!")
    print(f" Model UID: {model_uid}")

    print("\n--- 3. CREATING ONLINE DEPLOYMENT ---")
    deployment_metadata = {
        client.deployments.ModelMetaNames.NAME: "Credit_Card_Approval_Online_Deployment",
        client.deployments.ModelMetaNames.ONLINE: {}
    }

    deployment = client.deployments.create(
        artifact_uid=model_uid,
        meta_props=deployment_metadata
    )

    deployment_id = client.deployments.get_id(deployment)
    scoring_url = client.deployments.get_scoring_href(deployment)

    print("\n" + "=" * 60)
    print(" IBM WATSON MACHINE LEARNING DEPLOYMENT SUCCESSFUL! ")
    print(f" Deployment ID: {deployment_id}")
    print(f" Scoring Endpoint: {scoring_url}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    deploy_to_ibm_watson()

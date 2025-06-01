clear

#!/bin/bash

# Step 1: Enable Cloud Run API
echo "Enabling Cloud Run API...${RESET}"
gcloud services enable run.googleapis.com

# Step 2: Download required files from GitHub
echo "Downloading required application files...${RESET}"

wget https://raw.githubusercontent.com/awadheshk/resturantapp/refs/heads/main/myresturantapp.py
wget https://raw.githubusercontent.com/awadheshk/resturantapp/refs/heads/main/Dockerfile
wget https://raw.githubusercontent.com/awadheshk/resturantapp/refs/heads/main/requirements.txt

# Step 3: Set project and region variables
echo "Setting GCP project and region variables...${RESET}"
GCP_PROJECT=$(gcloud config get-value project)
GCP_REGION=$(gcloud compute project-info describe \
--format="value(commonInstanceMetadata.items[google-compute-default-region])")

echo
echo "GCP Project ID: ${RESET}""$GCP_PROJECT"
echo

echo
echo "GCP Location: ${RESET}""$GCP_REGION"
echo

# Step 4: Create a virtual environment and install dependencies
echo "Setting up Python virtual environment...${RESET}"
python3 -m venv documentqna
source documentqna/bin/activate
python3 -m  pip install -r requirements.txt

# Step 5: Start Streamlit application
# echo "Running Kitchen Recipe Generator application in the background...${RESET}"
# nohup streamlit run chef.py \
#  --browser.serverAddress=localhost \
#  --server.enableCORS=false \
#  --server.enableXsrfProtection=false \
#  --server.port 8080 > kitchenrecipe.log 2>&1 &

# Step 6: Create Artifact Repository

echo "Creating Artifact Registry repository...${RESET}"
AR_REPO='documentqna-repo'
SERVICE_NAME='documentqna-app' 
gcloud artifacts repositories create "$AR_REPO" --location="$GCP_REGION" --repository-format=Docker

# Step 7: Submit Cloud Build
echo "Submitting Cloud Build...${RESET}"
gcloud builds submit --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"

# Step 8: Deploy Cloud Run Service
echo "Deploying Cloud Run service...${RESET}"
gcloud run deploy "$SERVICE_NAME" \
  --port=8080 \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --allow-unauthenticated \
  --region=$GCP_REGION \
  --platform=managed  \
  --project=$GCP_PROJECT \
  --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION

# Step 9: Get Cloud Run Service URL
echo "Fetching Cloud Run service URL...${RESET}"
CLOUD_RUN_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$GCP_REGION" --format='value(status.url)')

echo
echo "Cloud Run Service is available at: ${RESET}""$CLOUD_RUN_URL"
echo

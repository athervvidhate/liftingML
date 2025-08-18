#!/bin/bash

echo "🚀 Starting deployment to Google Cloud App Engine..."

# Set the project ID (replace with your actual project ID)
PROJECT_ID="liftingml"

# Set the region
REGION="us-central1"

echo "📋 Project ID: $PROJECT_ID"
echo "🌍 Region: $REGION"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "🔐 Please authenticate with gcloud first:"
    echo "   gcloud auth login"
    exit 1
fi

# Set the project
echo "🔧 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Deploy the app
echo "📦 Deploying to App Engine..."
gcloud app deploy --quiet

# Get the app URL
echo "🌐 Getting app URL..."
APP_URL=$(gcloud app browse --no-launch-browser)

echo "✅ Deployment completed!"
echo "🔗 Your app is available at: $APP_URL"
echo ""
echo "📊 To view logs: gcloud app logs tail -s default"
echo "🛠️  To manage your app: gcloud app browse"

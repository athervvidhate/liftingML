# Docker Deployment Guide for Google Cloud Run

This guide will help you deploy your Streamlit app to Google Cloud Run using Docker.

## 🐳 What's Included

The Docker setup includes only the essential files:
- ✅ `streamlit_app.py` - Main application
- ✅ `RobertaSentenceEmbedder.py` - Model wrapper
- ✅ `roberta_finetuned/` - Trained model files
- ✅ `data/cleaned_600k.csv` - Essential data files
- ✅ `data/program_features.csv` - Program features
- ✅ `data/final_features.csv` - Final features

## 🚫 What's Excluded

The following files are NOT included in the Docker image:
- ❌ Jupyter notebooks (`*.ipynb`)
- ❌ Large datasets (`data/600k+ dataset/`)
- ❌ Development files
- ❌ Git files
- ❌ IDE files
- ❌ Documentation files

## 🚀 Quick Deployment

### Prerequisites
1. Install [Docker](https://docs.docker.com/get-docker/)
2. Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
3. Authenticate with Google Cloud: `gcloud auth login`

### Deploy with Script
```bash
./deploy-cloudrun.sh
```

### Manual Deployment
```bash
# Set your project
gcloud config set project liftingml

# Configure Docker
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/liftingml/liftingml-app .
docker push gcr.io/liftingml/liftingml-app

# Deploy to Cloud Run
gcloud run deploy liftingml-app \
    --image gcr.io/liftingml/liftingml-app \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 1
```

## ⚙️ Configuration

### Resource Limits
- **Memory**: 2GB (sufficient for PyTorch model)
- **CPU**: 1 vCPU
- **Max Instances**: 1 (cost optimization)
- **Timeout**: 300 seconds

### Environment Variables
- `STREAMLIT_SERVER_PORT=8080`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
- `STREAMLIT_SERVER_HEADLESS=true`

## 📊 Monitoring

### View Logs
```bash
gcloud logs tail --service=liftingml-app
```

### Check Service Status
```bash
gcloud run services list
```

### Get Service URL
```bash
gcloud run services describe liftingml-app --region=us-central1 --format="value(status.url)"
```

## 🔧 Customization

### Change Resources
Edit the deployment command in `deploy-cloudrun.sh`:
```bash
--memory 4Gi \    # Increase memory
--cpu 2 \         # Increase CPU
--max-instances 5 # Increase max instances
```

### Add Environment Variables
Add to the Dockerfile:
```dockerfile
ENV MY_VAR=value
```

## 💰 Cost Optimization

- **Max Instances**: Set to 1 to prevent scaling costs
- **Memory**: Optimized for your model size
- **CPU**: Minimal required for your workload
- **Region**: Choose closest to your users

## 🐛 Troubleshooting

### Build Issues
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t gcr.io/liftingml/liftingml-app .
```

### Runtime Issues
```bash
# Check logs
gcloud logs tail --service=liftingml-app

# Check service status
gcloud run services describe liftingml-app --region=us-central1
```

### Permission Issues
```bash
# Ensure you have the right permissions
gcloud projects add-iam-policy-binding liftingml \
    --member="user:your-email@gmail.com" \
    --role="roles/run.admin"
```

## 🔄 Updates

To update your deployment:
1. Make changes to your code
2. Run `./deploy-cloudrun.sh` again
3. Cloud Run will automatically update the service

## 📈 Scaling

The service will automatically scale to 0 when not in use (cost optimization) and scale up to 1 instance when needed.

# Streamlit Deployment Guide

This guide covers how to run and deploy the multi-agent laptop chatbot as a Streamlit application.

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

Add your Azure OpenAI credentials to `.streamlit/secrets.toml`:

```toml
AZURE_OPENAI_API_KEY = "your-api-key-here"
AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment Options

### Option 1: Deploy on Streamlit Cloud

1. **Push to GitHub**
   - Commit all files to your GitHub repository

2. **Connect to Streamlit Cloud**
   - Go to [Streamlit Cloud](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo, branch, and file (`app.py`)

3. **Set Secrets**
   - In the app settings, add your credentials under "Secrets"
   - Add the same content from `.streamlit/secrets.toml`:
   ```
   AZURE_OPENAI_API_KEY = "your-api-key"
   AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
   ```

### Option 2: Deploy on Heroku

1. **Create a Procfile**
   ```
   web: streamlit run --logger.level=error --client.showErrorDetails=false --server.enableXsrfProtection=false app.py
   ```

2. **Create a setup.sh**
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableXsrfProtection = false
   
   [logger]
   level = \"error\"
   " > ~/.streamlit/config.toml
   ```

3. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   heroku config:set AZURE_OPENAI_API_KEY="your-api-key"
   heroku config:set AZURE_OPENAI_ENDPOINT="your-endpoint"
   git push heroku main
   ```

### Option 3: Deploy with Docker

1. **Create a Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   docker build -t laptop-chatbot .
   docker run -p 8501:8501 \
     -e AZURE_OPENAI_API_KEY="your-api-key" \
     -e AZURE_OPENAI_ENDPOINT="your-endpoint" \
     laptop-chatbot
   ```

### Option 4: Deploy on AWS/GCP/Azure

1. **Create a Cloud VM** with Python 3.11+
2. **Install dependencies**
   ```bash
   git clone your-repo
   cd your-repo
   pip install -r requirements.txt
   ```
3. **Set environment variables**
4. **Run behind a reverse proxy** (nginx/Apache)
5. **Set up SSL** with Let's Encrypt

## Features

- 🎯 **Intelligent Routing**: Automatically routes queries to the right agent
- 💬 **Product QnA**: Answer questions about laptop features and pricing
- 📦 **Order Management**: Check order status and update quantities
- 💾 **Conversation Memory**: Maintains context across messages
- 🧠 **Multi-Agent Architecture**: Three specialized agents working together

## Troubleshooting

### Issue: "API credentials not configured"
- Solution: Make sure `.streamlit/secrets.toml` exists with valid credentials

### Issue: "Module not found"
- Solution: Run `pip install -r requirements.txt` to install all dependencies

### Issue: "PDF file not found"
- Solution: Ensure `data/Laptop product descriptions.pdf` exists in the project directory

### Issue: Slow first load
- Solution: The first request caches agents and models. This is normal.

## Performance Tips

1. **Use caching**: The app uses `@st.cache_resource` for model and agent initialization
2. **Optimize vector search**: The retriever is configured with k=1 for speed
3. **Stream responses**: Consider using streaming for better UX with long responses

## Security Notes

- **Never commit secrets**: Always use `.streamlit/secrets.toml` locally and environment variables in production
- **API Key Protection**: Keep your Azure OpenAI API key secure
- **Rate Limiting**: Consider adding rate limiting in production deployments

# Quick Start Guide - Streamlit Chatbot Deployment

## 📋 Prerequisites
- Python 3.9+
- Azure OpenAI API Key & Endpoint
- Git (for deployment options)

## 🚀 Quick Local Run (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Add Credentials
Create `.streamlit/secrets.toml`:
```toml
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
```

### Step 3: Run the App
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser!

---

## 🐳 Docker Deployment (Easy)

### Option A: Docker Compose (Recommended)
```bash
# Set environment variables
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"

# Run with docker-compose
docker-compose up
```

Open http://localhost:8501

### Option B: Manual Docker
```bash
docker build -t laptop-chatbot .

docker run -p 8501:8501 \
  -e AZURE_OPENAI_API_KEY="your-api-key" \
  -e AZURE_OPENAI_ENDPOINT="your-endpoint" \
  laptop-chatbot
```

---

## ☁️ Cloud Deployment

### Streamlit Cloud (Free & Easiest)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Click "New app" → Select your repo
4. Add secrets in app settings
5. Done! Your app is live

### Heroku
```bash
# Make setup.sh executable
chmod +x setup.sh

# Deploy
heroku create your-app-name
heroku config:set AZURE_OPENAI_API_KEY="your-api-key" AZURE_OPENAI_ENDPOINT="your-endpoint"
git push heroku main
```

### AWS (EC2)
```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install Python 3.11
sudo apt update && sudo apt install python3.11 python3.11-venv -y

# Clone repo and run
git clone your-repo-url
cd your-repo
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set env vars
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"

# Run with nohup to keep running after disconnect
nohup streamlit run app.py --server.port=80 --server.address=0.0.0.0 &
```

---

## 🧪 Testing

Once deployed, try these sample queries:
- **Small Talk**: "Hello! How are you?"
- **Product Info**: "Tell me about the SpectraBook laptop"
- **Pricing**: "How much does the AlphaBook cost?"
- **Order Status**: "What is the status of order ORD-7311?"
- **Update Order**: "Can you add one more laptop to order ORD-7311?"

---

## 📊 Architecture

```
User Input
    ↓
Router Agent (What type of query?)
    ├→ SMALLTALK → Small Talk Handler
    ├→ PRODUCT → Product QnA Agent 
    │            ├→ get_laptop_price()
    │            └→ Get_Product_Features()
    ├→ ORDER → Orders Agent
    │          ├→ get_order_details()
    │          └→ update_quantity()
    └→ END → Generic Response

    ↓
Response to User
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "API credentials not configured" | Add `.streamlit/secrets.toml` with your keys |
| "Module not found" | Run `pip install -r requirements.txt` |
| "PDF not found" | Ensure `data/Laptop product descriptions.pdf` exists |
| "Connection timeout" | Check your API key and endpoint are valid |

---

## 📚 More Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [LangChain Docs](https://python.langchain.com)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)

---

## 💡 Tips for Production

1. **Rate Limiting**: Add middleware to prevent abuse
2. **Logging**: Enable logging for monitoring
3. **Caching**: Responses are cached automatically
4. **Error Handling**: Check logs in production dashboard
5. **Security**: Never expose API keys in code

---

**Ready to deploy?** Follow one of the options above!

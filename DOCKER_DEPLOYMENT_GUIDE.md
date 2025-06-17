# 🐳 Docker Deployment Guide

## 🎯 **Two Deployment Options**

### **Option 1: Clean Educational Deployment** (Recommended for Teaching)
- **Starts with empty database** - perfect for demonstrations
- **Students see the learning progression** from empty to populated
- **Educational value** - watch the system learn as documents are added

### **Option 2: Pre-loaded Development Deployment**
- **Starts with sample documents** already loaded
- **Immediate functionality** for testing and development
- **Quick demos** without setup time

---

## 🚀 **Quick Start Commands**

### **Clean Educational Deployment** ✨
```bash
# Start with empty database (perfect for teaching)
docker-compose -f docker-compose.clean.yml up -d

# View logs
docker-compose -f docker-compose.clean.yml logs -f

# Stop
docker-compose -f docker-compose.clean.yml down
```

### **Pre-loaded Development Deployment**
```bash
# Start with sample documents (quick testing)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📋 **Pre-Deployment Checklist**

### **1. Prerequisites**
- [ ] Docker installed and running
- [ ] Docker Compose available
- [ ] DeepSeek API key obtained
- [ ] OpenAI API key obtained
- [ ] At least 4GB RAM available
- [ ] Port 8501 available

### **2. Environment Setup**
```bash
# Clone repository
git clone <your-repo-url>
cd chatbot-learning-tool

# Set up environment variables
cp env.example .env
# Edit .env and add your API keys:
# DEEPSEEK_API_KEY=your_deepseek_key_here
# OPENAI_API_KEY=your_openai_key_here
```

### **3. Build the Application**
```bash
# Build Docker image (only needed once or after changes)
docker-compose build
```

---

## 🎓 **Educational Deployment Scenarios**

### **Scenario 1: Classroom Demonstration**
**Use: Clean deployment** to show students the complete learning process

```bash
# Start clean
docker-compose -f docker-compose.clean.yml up -d

# Open browser to http://localhost:8501
# Students will see:
# 1. Empty system with no documents
# 2. Pure LLM responses (no RAG)
# 3. Upload documents and watch RAG improve answers
# 4. Real-time cost tracking as they learn
```

### **Scenario 2: Individual Student Practice**
**Use: Clean deployment** for hands-on learning

```bash
# Each student runs their own instance
docker-compose -f docker-compose.clean.yml up -d
# Students learn by building their own knowledge base
```

### **Scenario 3: Quick Testing/Development**
**Use: Pre-loaded deployment** for immediate functionality

```bash
# Start with samples for quick testing
docker-compose up -d
# Immediate access to RAG functionality for testing
```

---

## 🔧 **Configuration Options**

### **Memory and Performance Tuning**
```yaml
# In docker-compose.yml, adjust resource limits:
deploy:
  resources:
    limits:
      memory: 2G      # Increase for better performance
    reservations:
      memory: 512M    # Minimum memory guarantee
```

### **Port Configuration**
```yaml
# Change port if 8501 is in use:
ports:
  - "8502:8501"  # Access via http://localhost:8502
```

### **API Key Management**
```bash
# Option 1: Environment variables
export DEEPSEEK_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"

# Option 2: .env file (recommended)
echo "DEEPSEEK_API_KEY=your_key_here" >> .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Port Already in Use**
```bash
# Check what's using port 8501
lsof -i :8501

# Kill the process or change port in docker-compose.yml
```

#### **API Keys Not Working**
```bash
# Verify keys are loaded
docker-compose exec rag-chatbot printenv | grep API_KEY

# Check .env file format (no spaces around =)
DEEPSEEK_API_KEY=sk-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

#### **Container Won't Start**
```bash
# Check logs for errors
docker-compose logs rag-chatbot

# Check Docker status
docker ps -a
```

#### **Database Issues**
```bash
# For clean deployment, reset database:
docker-compose -f docker-compose.clean.yml down -v
docker-compose -f docker-compose.clean.yml up -d

# For pre-loaded deployment, check volume mount:
ls -la ./data/chroma_database
```

### **Performance Issues**

#### **Slow Response Times**
- Increase memory allocation in docker-compose.yml
- Check system resources: `docker stats`
- Verify API key rate limits

#### **High Memory Usage**
- Reduce document size or chunk count
- Restart container: `docker-compose restart`
- Monitor with: `docker stats --no-stream`

---

## 📊 **Monitoring and Maintenance**

### **Health Checks**
```bash
# Built-in health check
curl http://localhost:8501/_stcore/health

# Application status
curl http://localhost:8501
```

### **Resource Monitoring**
```bash
# Real-time stats
docker stats

# Container logs
docker-compose logs -f --tail=100

# Disk usage
docker system df
```

### **Backup and Restore**
```bash
# Backup data (pre-loaded deployment)
cp -r ./data ./data_backup_$(date +%Y%m%d)

# Backup clean deployment database
docker run --rm -v chatbot-learning-tool_rag_data_clean:/data -v $(pwd):/backup alpine tar czf /backup/clean_backup_$(date +%Y%m%d).tar.gz /data

# Restore clean deployment
docker run --rm -v chatbot-learning-tool_rag_data_clean:/data -v $(pwd):/backup alpine tar xzf /backup/clean_backup_YYYYMMDD.tar.gz -C /
```

---

## 🎯 **Production Deployment**

### **Security Hardening**
- Use Docker secrets for API keys
- Run behind reverse proxy (nginx)
- Enable HTTPS/SSL
- Regular security updates

### **Scaling Considerations**
- Use external database for multi-instance deployment
- Load balancing for multiple users
- Resource monitoring and alerting
- Automated backup procedures

---

## 📚 **Educational Use Cases**

### **For Teachers**
1. **Token Education**: Show students how text becomes tokens and costs
2. **RAG Demonstration**: Compare with/without document context
3. **Cost Optimization**: Teach efficient LLM usage
4. **Real-time Learning**: Watch costs accumulate as students experiment

### **For Students**
1. **Hands-on Experience**: Upload documents and see immediate results
2. **Cost Awareness**: Learn the economics of AI usage
3. **Parameter Tuning**: Experiment with different settings
4. **Knowledge Building**: Build their own RAG system step by step

### **For Institutions**
1. **Multi-user Support**: Each class can have their own instance
2. **Cost Tracking**: Monitor usage across classes
3. **Curriculum Integration**: Structured learning paths
4. **Assessment Tools**: Measure understanding through practical exercises

---

## 🚀 **Next Steps**

1. **Test both deployment options** on your target platforms
2. **Customize for your specific needs** (API limits, resource constraints)
3. **Create student guides** for your specific deployment
4. **Set up monitoring** for production use
5. **Plan scaling strategy** for institution-wide deployment

---

## 📞 **Support**

- **Logs**: Always check `docker-compose logs` first
- **Documentation**: See `documents/deployment_testing_plan.md` for comprehensive testing
- **Testing Script**: Run `./test_docker_deployment.sh` for automated validation
- **Configuration**: See `env.example` for all available options

This deployment guide ensures you can successfully deploy the RAG Chatbot Learning Tool in any educational environment! 🎓 
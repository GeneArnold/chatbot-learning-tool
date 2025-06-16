# RAG Chatbot Learning Tool

## 🚀 **MAJOR UPDATE: Educational Token & Cost Tracking** 

This application has been **significantly enhanced** with comprehensive cost tracking and token education features, making it the perfect tool for understanding how LLMs work and what they cost.

**🆕 NEW FEATURES:**
- **🔍 Complete Token Breakdown**: See exactly how your text becomes tokens and costs
- **💰 Real-time Cost Tracking**: Track embedding and query costs with full transparency
- **🎓 Educational Token Visualization**: Understand the relationship between text, tokens, and costs
- **📊 Session Cost Management**: Detailed cost history and breakdowns
- **🔄 Query Expansion Analysis**: See how query expansion affects tokens and costs
- **💡 Live Token Education**: Learn how words become multiple tokens

---

## 🎯 **Hybrid DeepSeek + OpenAI Architecture**

**Why Two API Keys?**
- **DeepSeek**: Primary LLM for chat, coding, and reasoning (much lower cost!)
- **OpenAI**: Embeddings only (DeepSeek doesn't offer embedding APIs yet)
- **Result**: Best of both worlds - advanced reasoning + proven embeddings

**API Keys Required:**
- `DEEPSEEK_API_KEY` - For LLM operations (chat, reasoning, code generation)
- `OPENAI_API_KEY` - For text embeddings (semantic search functionality)

---

## ✨ Complete Feature Set

### 🔤 **Token & Cost Education**
- **Real-time Token Breakdown**: See exactly how "tokenization" becomes ["token", "ization"]
- **Cost Transparency**: Every token tracked with precise pricing ($0.14/$0.28 per 1M tokens)
- **Query vs Context vs Response**: Understand where every token comes from
- **Session Cost Tracking**: Complete history of all embedding and query costs
- **Educational Tooltips**: Learn why token count ≠ word count

### 📄 **Document Processing**
- Upload and chunk text documents with customizable parameters
- **Embedding Cost Tracking**: See exactly what it costs to process your documents
- Real-time embedding with OpenAI's text-embedding-3-small
- Persistent vector storage with ChromaDB

### 🤖 **Multi-Model LLM Testing**
- **DeepSeek Chat**: Optimized for general conversation and analysis
- **DeepSeek Coder**: Specialized for code-related tasks
- **DeepSeek Reasoner**: Advanced reasoning with step-by-step thinking
- Side-by-side comparison of RAG vs non-RAG responses

### 🔍 **Advanced RAG Analysis**
- **Query Expansion**: Automatic query enhancement for better retrieval
- **Retrieval Quality Metrics**: Similarity scores and relevance analysis
- **Source Attribution**: Track which documents contribute to answers
- **Chunk Analysis**: Detailed breakdown of retrieved context

### 📊 **Comprehensive Analytics**
- Real-time parameter adjustment with immediate feedback
- Vector database explorer with similarity testing
- Complete test history and result comparison
- Export capabilities for further analysis

## 🎓 Perfect For Learning

- **Understanding LLMs**: See exactly how AI processes your text
- **Cost Optimization**: Learn what drives LLM costs and how to optimize
- **RAG Experimentation**: Test different retrieval strategies
- **Token Education**: Understand the fundamental unit of LLM processing
- **Parameter Tuning**: See immediate effects of different settings

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- **DeepSeek API key** (primary LLM operations)
- **OpenAI API key** (embeddings for semantic search)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd chatbot-learning-tool
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp env.example .env
# Edit .env and add BOTH API keys:
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Run the application**
```bash
streamlit run app.py
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Set both API keys
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

The application will be available at `http://localhost:8501`

## 📖 How to Use

### 1. **💬 Ask Questions** 
- Ask questions with or without document context
- **See complete token breakdown**: How your question becomes tokens and costs
- **Query expansion analysis**: See how your query gets enhanced for better search
- **Real-time cost tracking**: Watch costs accumulate with full transparency

### 2. **📁 Document Management**
- Upload `.txt` or `.md` files with drag-and-drop
- **Embedding cost tracking**: See exactly what it costs to process documents
- Adjust chunk size and overlap parameters
- **Per-token pricing display**: Understand embedding costs completely

### 3. **⚙️ Parameter Optimization**
- Fine-tune temperature, max tokens, top-p
- **Cost impact preview**: See how parameter changes affect costs
- Test different chunk sizes and retrieval counts
- Enable/disable query expansion with cost comparison

### 4. **🔍 RAG Analysis**
- **Complete retrieval breakdown**: See which documents contribute
- **Query expansion details**: Understand how your query was enhanced
- **Similarity scoring**: Analyze retrieval quality with detailed metrics
- **Source attribution**: Track document contributions to answers

### 5. **📊 Cost & Test History**
- **Session cost tracking**: Complete breakdown of all costs
- **Token education**: Learn exactly where costs come from
- Export test results and cost analysis
- Compare different approaches and their costs

### 6. **🧮 Vector Explorer**
- Browse your vector database interactively
- Test semantic similarity between texts
- Understand how embeddings work

## 💰 Cost Structure & Transparency

### **DeepSeek Pricing** (Primary LLM)
- **Input**: $0.14 per 1M tokens
- **Output**: $0.28 per 1M tokens
- **Usage**: Chat, reasoning, code generation

### **OpenAI Pricing** (Embeddings Only)
- **Embeddings**: $0.02 per 1M tokens
- **Usage**: Document processing and query embedding

### **Cost Tracking Features**
- **Real-time calculation**: See costs as they happen
- **Token transparency**: Understand exactly what you're paying for
- **Session history**: Track all costs across your session
- **Educational breakdown**: Learn how text becomes tokens and costs

## 🔧 Technical Architecture

- **Frontend**: Streamlit with custom educational components
- **Vector Database**: ChromaDB for persistent semantic search
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: DeepSeek V3 (Chat, Coder, Reasoner models)
- **Token Processing**: tiktoken for accurate tokenization
- **Cost Tracking**: Real-time calculation with session persistence

## 📁 Project Structure

```
chatbot-learning-tool/
├── app.py                 # Main Streamlit application with cost tracking
├── config.py             # Configuration + cost calculation logic
├── requirements.txt      # Python dependencies (includes tiktoken)
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-container setup
├── env.example          # Environment template (both API keys)
├── data/               # Local data storage
│   ├── sample_docs/    # Example documents
│   └── chroma_database/ # Vector database files
└── documents/          # Documentation
    ├── user_manual.md  # Comprehensive user guide
    ├── roadmap.md      # Development roadmap
    └── challenges.md   # Testing scenarios
```

## ⚙️ Configuration

### Required Environment Variables

```bash
# BOTH API keys are required
DEEPSEEK_API_KEY=your_deepseek_api_key_here    # Primary LLM operations
OPENAI_API_KEY=your_openai_api_key_here        # Embeddings only

# Optional Model Configuration
EMBEDDING_MODEL=text-embedding-3-small         # OpenAI embedding model
DEFAULT_LLM_MODEL=deepseek-chat               # Primary DeepSeek model

# Optional RAG Parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=100
N_RESULTS=3

# Optional LLM Parameters  
LLM_MAX_TOKENS=8192      # Increased for DeepSeek capabilities
LLM_TEMPERATURE=0.2
LLM_TOP_P=1.0
```

## 🧪 Example Learning Scenarios

### **Understanding Token Costs**
1. Ask: "What is machine learning?"
2. See how your 4-word question becomes tokens
3. Understand why the response costs more than the input
4. Learn how query expansion affects total costs

### **Document Processing Analysis**
1. Upload a research paper
2. See exact embedding costs per document
3. Understand how document size affects processing costs
4. Compare different chunking strategies

### **Query Optimization**
1. Try the same question with/without query expansion
2. See how expansion affects token usage and costs
3. Compare retrieval quality vs. cost trade-offs
4. Optimize for your specific use case

### **Model Comparison**
1. Ask the same question to DeepSeek Chat vs Coder
2. Compare response quality and token usage
3. Understand when to use which model
4. Analyze cost vs. performance trade-offs

## 🎯 Educational Outcomes

After using this tool, you'll understand:
- ✅ **How LLMs process text** (tokenization)
- ✅ **What drives LLM costs** (input/output tokens)
- ✅ **How RAG works** (retrieval + generation)
- ✅ **Query enhancement strategies** (expansion)
- ✅ **Cost optimization techniques** (parameter tuning)
- ✅ **Token vs. word relationships** (why they differ)

## 🤝 Contributing

This tool is designed to be educational and transparent. Contributions welcome for:
- Additional cost tracking features
- New educational visualizations
- Enhanced token analysis
- Performance optimizations
- Documentation improvements

## 📝 License

MIT License - Feel free to use this for learning and research!

---

## 🌟 **Ready to Learn?**

This isn't just a RAG testing tool - it's a **complete education platform** for understanding modern LLMs, their costs, and how to optimize them. Start with a simple question and watch as the app breaks down every token, every cost, and every decision in the process.

**Get started with both API keys and discover how LLMs really work!** 🚀 
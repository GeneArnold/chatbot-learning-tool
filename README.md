# DGT RAG Chatbot Testing Interface

A comprehensive, educational Streamlit application for experimenting with Retrieval-Augmented Generation (RAG) systems. This professional-grade testing interface allows users to compare AI responses with and without document context, providing deep insights into RAG performance and behavior.

## 🚀 Features

### **Core RAG Functionality**
- **Multi-Model Support**: GPT-3.5 Turbo, GPT-4, and GPT-4 Turbo
- **Document Processing**: Upload and process .txt and .md files with configurable chunking
- **Vector Database**: ChromaDB integration with persistent storage
- **Query Expansion**: Advanced query enhancement for better retrieval accuracy
- **Semantic Search**: OpenAI text-embedding-3-small for high-quality embeddings

### **Advanced Testing & Analysis**
- **6-Tab Interface**: Organized workflow for comprehensive testing
- **Parameter Tuning**: Real-time adjustment of chunk size, overlap, retrieval count, and LLM parameters
- **RAG Details**: Deep dive into query expansion, chunk retrieval, and similarity scores
- **Test History**: Complete tracking of all queries and responses with comparison capabilities
- **Vector Explorer**: Interactive exploration of chunk IDs, embeddings, and similarity relationships

### **Document Management**
- **Metadata Tracking**: Complete document provenance with upload times and statistics
- **Individual Document Control**: View, analyze, and delete specific documents
- **Duplicate Detection**: Automatic detection and handling of duplicate uploads
- **Chunk Visualization**: Detailed view of how documents are split and processed

### **Educational Features**
- **Transparent Process**: Full visibility into retrieval and generation steps
- **Performance Metrics**: Similarity scores, chunk analysis, and retrieval statistics
- **Debugging Tools**: Query expansion visualization and document source analysis
- **Comparison Mode**: Side-by-side comparison of RAG vs. non-RAG responses

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatbot_testing
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
chatbot_testing/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables (create this)
├── README.md                  # This file
├── .gitignore                 # Git exclusions
├── images/
│   └── DGT.webp              # DGT logo for branding
├── demo_context_files/        # Sample documents for testing
│   ├── hoover_dam.md
│   └── international_space_station.md
├── documents/                 # Project documentation and tracking
│   ├── challenges.md          # Technical challenges and solutions
│   ├── updates.txt           # Project update log
│   ├── RAG_Testing_Guide.md  # Comprehensive testing guide
│   ├── RAG_Process_Outline.md # RAG process documentation
│   └── main.py               # Legacy implementation (reference)
├── .chroma_database/          # ChromaDB persistent storage (auto-created)
└── uploads/                   # Temporary file storage (auto-created)
```

## 🎯 Usage Guide

### **Getting Started**
1. **Upload Documents**: Use the "Documents" tab to upload .txt or .md files
2. **Configure Parameters**: Adjust chunking and retrieval settings in "Parameters" tab
3. **Ask Questions**: Use the "Ask Questions" tab to test your RAG system
4. **Analyze Results**: Explore detailed analysis in "RAG Details" and "Vector Explorer" tabs

### **Key Tabs Overview**

- **💬 Ask Questions**: Main interface for querying with RAG vs. non-RAG comparison
- **📁 Documents**: Document upload, management, and metadata viewing
- **⚙️ Parameters**: Fine-tune chunk size, overlap, retrieval count, and LLM settings
- **🔍 RAG Details**: Deep analysis of query expansion and retrieval process
- **📊 Test Results**: Complete history and comparison of all test queries
- **🧮 Vector Explorer**: Interactive exploration of embeddings and chunk relationships

### **Advanced Features**

#### **Query Expansion**
The system automatically expands queries for better retrieval:
- Multi-topic detection and handling
- Keyword variations and synonyms
- Context-aware query enhancement

#### **Parameter Optimization**
Experiment with different settings:
- **Chunk Size**: 200-2000 characters
- **Chunk Overlap**: 0-500 characters  
- **Retrieval Count**: 1-20 results
- **Temperature**: 0.0-2.0 for response creativity
- **Max Tokens**: 50-4000 for response length

#### **Document Analysis**
- View individual document chunks and metadata
- Analyze chunk distribution and sizes
- Track document upload history and statistics

## 🎓 Educational Value

This interface is designed as a comprehensive learning tool for understanding:

- **RAG Architecture**: How retrieval and generation work together
- **Vector Databases**: Storage and retrieval of semantic embeddings
- **Parameter Impact**: How different settings affect RAG performance
- **Query Processing**: The journey from question to context to answer
- **Evaluation Methods**: Comparing and analyzing RAG system performance

## 🔧 Technical Details

### **Architecture**
- **Frontend**: Streamlit with custom HTML/CSS for professional UI
- **Vector Database**: ChromaDB with persistent storage
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **LLM**: OpenAI GPT models via API
- **Document Processing**: Custom chunking with configurable overlap

### **Key Technical Features**
- **Unique ID System**: Timestamp-based chunk IDs prevent collisions
- **Metadata Tracking**: Complete document provenance and statistics
- **Session Management**: Persistent state across app interactions
- **Error Handling**: Robust error management and user feedback
- **Performance Optimization**: Efficient vector operations and caching

## 📚 Documentation

The `documents/` folder contains comprehensive project documentation:

- **`challenges.md`**: Technical challenges encountered and solutions implemented
- **`updates.txt`**: Project update log and version history  
- **`RAG_Testing_Guide.md`**: Detailed guide for testing RAG systems
- **`RAG_Process_Outline.md`**: Technical documentation of RAG processes
- **`main.py`**: Legacy implementation for reference

These documents are actively maintained and provide valuable insights into the development process and technical decisions.

## 🤝 Contributing

This project is designed to be educational and extensible. Feel free to:
- Experiment with different embedding models
- Add new document formats
- Implement additional analysis features
- Enhance the UI/UX
- Update documentation as the project evolves

## 📄 License

This project is intended for educational purposes. Please ensure you comply with OpenAI's usage policies when using their API.

## 🆘 Support

For questions or issues:
1. Check the RAG Details tab for debugging information
2. Review parameter settings in the Parameters tab
3. Examine document processing in the Documents tab
4. Use the Vector Explorer for deep technical analysis

---

**Built with ❤️ for RAG education and experimentation** 
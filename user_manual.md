# RAG Chatbot Learning Tool - Comprehensive User Manual

*Master guide to understanding LLM costs, tokens, and RAG optimization*

---

## 📖 **Table of Contents**

1. [🆕 New Features Overview](#new-features-overview)
2. [🎯 API Key Setup](#api-key-setup)
3. [🔤 Understanding Tokens & Costs](#understanding-tokens--costs)
4. [💰 Cost Tracking Features](#cost-tracking-features)
5. [💬 Ask Questions Tab - Complete Guide](#ask-questions-tab---complete-guide)
6. [📁 Document Management](#document-management)
7. [⚙️ Parameter Optimization](#parameter-optimization)
8. [🔍 RAG Analysis](#rag-analysis)
9. [📊 Cost & Test History](#cost--test-history)
10. [🧮 Vector Explorer](#vector-explorer)
11. [🎓 Educational Learning Paths](#educational-learning-paths)
12. [🛠️ Troubleshooting](#troubleshooting)
13. [💡 Best Practices](#best-practices)

---

## 🆕 **New Features Overview**

### **🎓 What Makes This Tool Educational**
This isn't just a RAG testing tool - it's a **complete education platform** for understanding:
- How your text becomes tokens and costs money
- Why "tokenization" becomes ["token", "ization"] 
- What drives LLM costs and how to optimize them
- How RAG retrieval affects token usage
- Real-time cost tracking with full transparency

### **🔍 Major New Features**
- **Complete Token Breakdown**: See exactly how words become tokens
- **Real-time Cost Tracking**: Watch costs accumulate with full transparency
- **Query Expansion Analysis**: Understand how your questions get enhanced
- **Session Cost Management**: Track all costs across your session
- **Educational Visualizations**: Learn the relationship between text, tokens, and costs
- **Three DeepSeek Models**: Chat, Coder, and Reasoner with cost comparisons

---

## 🎯 **API Key Setup**

### **Why Two API Keys?**
This tool uses a **hybrid architecture** for optimal cost and performance:

**🧠 DeepSeek API (Primary LLM)**
- **Purpose**: Chat, reasoning, and code generation
- **Cost**: $0.14/$0.28 per 1M tokens (much cheaper than OpenAI!)
- **Models**: deepseek-chat, deepseek-coder, deepseek-reasoner
- **Required**: `DEEPSEEK_API_KEY=your_deepseek_api_key_here`

**🔍 OpenAI API (Embeddings Only)**
- **Purpose**: Text embeddings for semantic search
- **Cost**: $0.02 per 1M tokens (embeddings only)
- **Why**: DeepSeek doesn't offer embedding APIs yet
- **Required**: `OPENAI_API_KEY=your_openai_api_key_here`

### **Setting Up API Keys**
```bash
# In your .env file:
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🔤 **Understanding Tokens & Costs**

### **What Are Tokens?**
Tokens are the fundamental unit that LLMs use to process text. This tool shows you exactly how this works:

**Example Breakdown:**
- **Your Text**: "What is machine learning?"
- **Tokens**: ["What", " is", " machine", " learning", "?"]
- **Token Count**: 5 tokens
- **Cost**: 5 × $0.14/1M = $0.0000007

### **Why Token Count ≠ Word Count**
- **Complex Words**: "tokenization" → ["token", "ization"] (2 tokens)
- **Punctuation**: "Hello!" → ["Hello", "!"] (2 tokens)
- **Spaces**: Included in tokenization logic
- **Special Characters**: Each may be separate tokens

### **The Complete Cost Equation**
**Total Cost = (Input Tokens × Input Price) + (Output Tokens × Output Price)**

With RAG:
- **Input Tokens**: Your question + retrieved document context
- **Output Tokens**: AI's complete response
- **Query Expansion**: May add extra input tokens for better search

---

## 💰 **Cost Tracking Features**

### **🔍 Real-time Token Breakdown**
Every question shows you:

**Section 1: Complete Input Tokenization**
- **1a. Your Question**: Exact tokens from your input
- **1b. Query Expansion**: Additional tokens from enhanced search
- **1c. Total Input Cost**: Combined cost of question + expansion

**Section 2: Document Context**
- **Tokens Sent to AI**: Retrieved document chunks as tokens
- **Context Cost**: How document retrieval affects input cost

**Section 3: AI Response**
- **Output Tokens**: Tokens in the AI's response
- **Output Cost**: Most expensive part of the process

**Section 4: Complete Cost Calculation**
- **Total Session Cost**: All costs accumulated this session
- **Per-Question Breakdown**: Exact cost attribution

### **💰 Session Cost Tracking**
- **Embedding Costs**: One-time cost when uploading documents
- **Query Costs**: Per-question costs with full breakdown
- **Running Total**: Watch costs accumulate in real-time
- **Cost History**: Complete session financial tracking

### **🎓 Educational Cost Display**
- **Per-Token Pricing**: Understand exact cost per token
- **Input vs Output**: See why responses cost more than questions
- **Expansion Impact**: Learn how query enhancement affects costs
- **Model Comparison**: Compare costs across DeepSeek models

---

## 💬 **Ask Questions Tab - Complete Guide**

### **🎯 Primary Interface for Learning**

#### **Question Input**
- **Text Area**: Enter any question to test
- **Live Token Preview**: See token count as you type
- **Cost Preview**: Estimated cost before asking

#### **🤖 Model Selection**
- **deepseek-chat**: General conversation and analysis
- **deepseek-coder**: Code generation and technical questions
- **deepseek-reasoner**: Complex reasoning with step-by-step thinking
- **Cost Impact**: Each model has same pricing but different capabilities

#### **🔄 Query Expansion**
- **Enable/Disable**: Toggle automatic query enhancement
- **How It Works**: AI generates additional search queries for better retrieval
- **Cost Impact**: Shows exact token cost of expansion
- **Educational Value**: Learn how search queries get optimized

#### **📊 Complete Response Analysis**
When you ask a question, you get:

**🎯 AI Response**
- **Main Answer**: AI's response using retrieved context
- **Source Attribution**: Which documents contributed to the answer

**🔍 Complete Token Breakdown**
- **Section 1**: Input tokenization (question + expansion)
- **Section 2**: Document context tokens
- **Section 3**: AI response tokens
- **Section 4**: Total cost calculation

**📈 Cost Details**
- **Click for Details**: Expandable cost breakdown
- **Token Education**: Learn why costs are what they are
- **Session Tracking**: Running cost total

#### **🆚 RAG vs General Comparison**
- **Side-by-Side**: Compare RAG-enhanced vs general knowledge answers
- **Cost Comparison**: See how document context affects costs
- **Quality Analysis**: Understand the value of RAG retrieval

---

## 📁 **Document Management**

### **📤 Document Upload with Cost Tracking**
- **Drag & Drop**: Easy file upload for .txt and .md files
- **Embedding Cost Preview**: See exact cost before processing
- **Real-time Processing**: Watch documents get chunked and embedded
- **Cost Transparency**: Exact per-document embedding costs

### **📊 Document Status**
- **Total Chunks**: Number of text segments created
- **Embedding Costs**: Exact cost spent on document processing
- **Storage Analysis**: How much content is available for retrieval

### **🗂️ Document Collection Management**
- **Individual Document View**: See chunks from specific documents
- **Cost Per Document**: Understand processing costs per file
- **Delete Options**: Remove specific documents to save on storage

### **⚠️ Database Management**
- **Clear All Data**: Complete reset with cost implications
- **Safety Warnings**: Understand what gets deleted
- **Fresh Start**: Begin new cost tracking session

---

## ⚙️ **Parameter Optimization**

### **🤖 Model Parameters with Cost Impact**

#### **Temperature (0.0 - 1.0)**
- **Low (0.0-0.3)**: Deterministic, shorter responses (lower output costs)
- **High (0.7-1.0)**: Creative, longer responses (higher output costs)
- **Cost Impact**: Higher temperature may increase output token costs

#### **Max Tokens (1-8192)**
- **Purpose**: Limit response length for cost control
- **Cost Impact**: Direct effect on maximum output costs
- **Educational Value**: Learn to balance quality vs. cost

#### **Top-p (0.1-1.0)**
- **Purpose**: Control response diversity
- **Cost Impact**: May affect response length and token usage

### **📄 RAG Parameters with Cost Implications**

#### **Chunk Size (100-2000)**
- **Embedding Cost**: Larger chunks cost more to embed
- **Query Cost**: Affects how much context gets sent to LLM
- **Quality Trade-off**: Larger chunks may provide better context

#### **Chunk Overlap (0-200)**
- **Storage Cost**: More overlap = more chunks = higher embedding costs
- **Retrieval Quality**: Better continuity between chunks

#### **Number of Results (1-10)**
- **Query Cost**: More results = more context tokens = higher costs per question
- **Quality vs Cost**: Find optimal balance for your use case

### **🔄 Query Expansion Settings**
- **Enable/Disable**: Control automatic query enhancement
- **Cost Impact**: Expansion adds input tokens but may improve quality
- **Educational Value**: Learn when expansion helps vs. hurts

---

## 🔍 **RAG Analysis**

### **🎯 Complete Retrieval Breakdown**
After each question, analyze:

#### **📝 Query Processing**
- **Original Question**: Your exact input with token count
- **Expanded Queries**: Additional search terms generated (if enabled)
- **Search Strategy**: How the system found relevant content

#### **📊 Document Retrieval**
- **Retrieved Chunks**: Exact text segments used for context
- **Similarity Scores**: How well each chunk matched your question
- **Source Distribution**: Which documents contributed most

#### **💰 Cost Analysis**
- **Input Tokenization**: Question + expansion + context
- **Processing Costs**: Exact breakdown of where money was spent
- **Efficiency Metrics**: Cost per quality unit

### **🔍 Quality Metrics**
- **Relevance Scoring**: How well retrieved content matched the question
- **Coverage Analysis**: Did retrieval capture all relevant information
- **Source Diversity**: Variety of documents contributing to the answer

---

## 📊 **Cost & Test History**

### **💰 Session Cost Tracking**
- **Total Session Cost**: All costs accumulated this session
- **Embedding vs Query Costs**: Breakdown by operation type
- **Cost per Question**: Individual question cost analysis
- **Running Financial Total**: Real-time cost accumulation

### **📈 Test Results Analysis**
- **Question History**: All questions asked with full context
- **Response Quality**: Compare different approaches
- **Cost Efficiency**: Analyze cost vs. quality trade-offs
- **Parameter Impact**: See how settings affect costs and quality

### **📊 Export and Analysis**
- **CSV Export**: Download complete test history
- **Cost Breakdown**: Detailed financial analysis
- **Performance Metrics**: Quality and efficiency measurements

---

## 🧮 **Vector Explorer**

### **🔍 Database Exploration**
- **Browse Chunks**: Explore all document segments
- **Similarity Testing**: Test how similar different texts are
- **Embedding Analysis**: Understand how semantic search works

### **🧪 Similarity Experiments**
- **Text Comparison**: Enter two texts and see similarity scores
- **Document Analysis**: Find most similar chunks across documents
- **Search Testing**: Test different query approaches

---

## 🎓 **Educational Learning Paths**

### **🌟 Beginner Path: Understanding Tokens**
1. **Start Simple**: Ask "What is AI?" and examine token breakdown
2. **Compare Models**: Try same question with different DeepSeek models
3. **Cost Analysis**: Understand why output costs more than input
4. **Token Education**: Learn why "AI" is one token but "machine learning" is multiple

### **🔬 Intermediate Path: RAG Optimization**
1. **Upload Documents**: Process a document and see embedding costs
2. **Parameter Testing**: Try different chunk sizes and see cost impacts
3. **Query Expansion**: Enable/disable expansion and compare costs
4. **Quality vs Cost**: Find optimal balance for your use case

### **🎯 Advanced Path: Cost Optimization**
1. **Multi-Model Testing**: Compare deepseek-chat vs deepseek-coder costs
2. **Parameter Optimization**: Fine-tune for specific cost targets
3. **Expansion Strategy**: Learn when query expansion helps vs. hurts
4. **Session Management**: Track complete session costs and optimize

### **🏆 Expert Path: Production Readiness**
1. **Benchmark Testing**: Establish baseline costs for your use case
2. **Quality Metrics**: Define success criteria beyond just cost
3. **Scaling Analysis**: Understand costs at different usage levels
4. **Optimization Strategies**: Advanced techniques for cost control

---

## 🛠️ **Troubleshooting**

### **🔑 API Key Issues**
- **DeepSeek Key**: Verify `DEEPSEEK_API_KEY` is set correctly
- **OpenAI Key**: Verify `OPENAI_API_KEY` is set correctly
- **Testing**: Use "Ask Questions" tab to test both APIs

### **💰 Cost Tracking Issues**
- **Missing Costs**: Refresh page to reload session data
- **Incorrect Totals**: Clear session data and restart
- **Token Mismatches**: Check for special characters in input

### **📄 Document Processing Issues**
- **Embedding Errors**: Check OpenAI API key and quota
- **Large Files**: Break into smaller documents for better processing
- **Format Issues**: Ensure files are .txt or .md format

### **🔍 Query Issues**
- **No Results**: Check if documents are properly processed
- **Poor Quality**: Adjust chunk size or number of results
- **High Costs**: Reduce max tokens or disable query expansion

---

## 💡 **Best Practices**

### **💰 Cost Management**
- **Start Small**: Begin with short questions to understand costs
- **Monitor Session**: Watch running totals to avoid surprises
- **Optimize Parameters**: Find balance between quality and cost
- **Use Appropriate Models**: Chat for general, Coder for technical questions

### **📄 Document Strategy**
- **Quality Content**: Better documents lead to better results
- **Optimal Chunking**: Test different chunk sizes for your content
- **Relevant Documents**: Only include documents relevant to expected questions

### **🔍 Query Optimization**
- **Clear Questions**: Specific questions get better results
- **Expansion Testing**: Try with/without expansion for your use case
- **Model Selection**: Use the right model for your question type

### **🎓 Learning Approach**
- **Start with Education**: Use this tool to learn about LLMs first
- **Experiment Safely**: Small costs while learning principles
- **Build Understanding**: Focus on token education before optimization
- **Practice Different Scenarios**: Test various document and question types

---

## 🌟 **Ready to Master LLMs?**

This tool transforms abstract LLM concepts into concrete, visual understanding. Every token is tracked, every cost is explained, and every decision is transparent. Start with a simple question and discover exactly how modern AI works - from text to tokens to costs to results.

**Your LLM education journey begins with understanding tokens. Let's start exploring!** 🚀 
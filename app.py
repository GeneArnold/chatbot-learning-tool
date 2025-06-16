import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import chromadb
import openai
from dotenv import load_dotenv
import json
from datetime import datetime
import numpy as np
import pandas as pd
from config import Config

load_dotenv()

# Initialize configuration and ensure directories exist
Config.ensure_directories()

# Validate configuration
config_errors = Config.validate_config()
if config_errors:
    for error in config_errors:
        st.error(f"⚠️ Configuration Error: {error}")

# Show Docker info if in Docker environment
if Config.IS_DOCKER:
    st.info(f"🐳 Running in Docker mode - Data directory: {Config.DATA_DIR}")

# =====================
# TOOLTIP SYSTEM
# =====================
def tooltip(text, help_text):
    """Create a text element with tooltip using Streamlit's help parameter."""
    return st.write(f"**{text}**", help=help_text)

def tooltip_metric(label, value, help_text):
    """Create a metric with tooltip."""
    return st.metric(label, value, help=help_text)

def tooltip_slider(label, min_val, max_val, default_val, step, help_text, key=None):
    """Create a slider with tooltip."""
    return st.slider(label, min_val, max_val, default_val, step, help=help_text, key=key)

def tooltip_selectbox(label, options, help_text, key=None, format_func=None):
    """Create a selectbox with tooltip."""
    return st.selectbox(label, options, help=help_text, key=key, format_func=format_func)

def tooltip_text_area(label, default_text, help_text, key=None):
    """Text area with tooltip"""
    st.markdown(tooltip(label, help_text), unsafe_allow_html=True)
    return st.text_area("", default_text, key=key, label_visibility="collapsed")

def display_cost_breakdown(cost_data, title="Cost Breakdown"):
    """Display detailed cost breakdown with per-token pricing"""
    if not cost_data:
        return
    
    st.markdown(f"### 💰 {title}")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_cost = cost_data.get('total_cost', 0)
        st.metric("Total Cost", f"${total_cost:.6f}")
    with col2:
        input_tokens = cost_data.get('input_tokens', 0)
        st.metric("Input Tokens", f"{input_tokens:,}")
    with col3:
        output_tokens = cost_data.get('output_tokens', 0)
        st.metric("Output Tokens", f"{output_tokens:,}")
    
    # Detailed breakdown
    with st.expander("💡 Detailed Cost Breakdown"):
        model_name = cost_data.get('model', 'unknown')
        
        # Input costs
        st.write("**📥 Input Costs:**")
        input_cost_per_token = Config.get_cost_per_token(model_name, 'input')
        input_cost = cost_data.get('input_cost', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• **Rate:** ${input_cost_per_token:.8f} per token")
            st.write(f"• **Tokens:** {input_tokens:,}")
        with col2:
            st.write(f"• **Cost:** ${input_cost:.6f}")
        
        # Output costs (if applicable)
        if output_tokens > 0:
            st.write("**📤 Output Costs:**")
            output_cost_per_token = Config.get_cost_per_token(model_name, 'output')
            output_cost = cost_data.get('output_cost', 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"• **Rate:** ${output_cost_per_token:.8f} per token")
                st.write(f"• **Tokens:** {output_tokens:,}")
            with col2:
                st.write(f"• **Cost:** ${output_cost:.6f}")

def display_query_cost_breakdown(original_query, expanded_queries, output_tokens, model_name):
    """Display immediate cost breakdown for the current query"""
    with st.expander("💰 Cost Breakdown for This Query", expanded=False):
        # Calculate costs
        original_tokens = Config.count_tokens(original_query, model_name)
        if len(expanded_queries) > 1:
            expansion_text = " ".join(expanded_queries[1:])
            expansion_tokens = Config.count_tokens(expansion_text, model_name)
        else:
            expansion_tokens = 0
        
        total_input_tokens = original_tokens + expansion_tokens
        cost_data = Config.calculate_cost(model_name, total_input_tokens, output_tokens)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Cost", f"${cost_data['input_cost']:.6f}")
        with col2:
            st.metric("Output Cost", f"${cost_data['output_cost']:.6f}")
        with col3:
            st.metric("Total Cost", f"${cost_data['total_cost']:.6f}")

def display_token_breakdown(original_query, expanded_queries, context_chunks, response_text, model_name):
    """Display comprehensive token breakdown for educational purposes"""
    with st.expander("🔍 Token Breakdown - See How Text Becomes Tokens & Costs", expanded=False):
        st.markdown("### 🎓 Understanding Tokens & Costs")
        st.info("💡 **What are tokens?** Tokens are pieces of text that AI models process. Words can be split into multiple tokens (e.g., 'tokenization' → 'token' + 'ization'). Each token costs money to process.")
        
        # 1. COMPLETE INPUT TOKENIZATION BREAKDOWN
        st.markdown("### 1. 🔤 Input Tokenization Breakdown")
        
        # 1a. Original Query
        st.markdown("#### a) Your Question")
        query_breakdown = Config.tokenize_text(original_query, model_name)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("**Original Text:**")
            st.code(original_query, language=None)
        with col2:
            st.write(f"**Tokenized View ({query_breakdown['count']} tokens):**")
            # Create a visual representation of tokens
            token_display = ""
            for i, token in enumerate(query_breakdown['tokens']):
                # Escape special characters and add color coding
                escaped_token = token.replace('\n', '\\n').replace('\t', '\\t')
                if escaped_token.strip() == '':
                    escaped_token = '[SPACE]'
                token_display += f"`{escaped_token}` "
            st.markdown(token_display)
        
        # Cost for original query
        query_cost = Config.calculate_cost(model_name, query_breakdown['count'], 0)
        st.write(f"**Cost:** {query_breakdown['count']} tokens × ${Config.get_cost_per_token(model_name, 'input'):.8f} = **${query_cost['input_cost']:.6f}**")
        
        # 1b. Query Expansion (always show section for clarity)
        st.markdown("#### b) Query Expansion")
        if len(expanded_queries) > 1:
            expansion_text = " ".join(expanded_queries[1:])
            expansion_breakdown = Config.tokenize_text(expansion_text, model_name)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("**Expanded Queries:**")
                for i, exp_query in enumerate(expanded_queries[1:], 2):
                    st.write(f"{i}. {exp_query}")
            with col2:
                st.write(f"**Additional Tokens ({expansion_breakdown['count']} tokens):**")
                expansion_display = ""
                for token in expansion_breakdown['tokens']:
                    escaped_token = token.replace('\n', '\\n').replace('\t', '\\t')
                    if escaped_token.strip() == '':
                        escaped_token = '[SPACE]'
                    expansion_display += f"`{escaped_token}` "
                st.markdown(expansion_display)
            
            expansion_cost = Config.calculate_cost(model_name, expansion_breakdown['count'], 0)
            st.write(f"**Additional Cost:** {expansion_breakdown['count']} tokens × ${Config.get_cost_per_token(model_name, 'input'):.8f} = **${expansion_cost['input_cost']:.6f}**")
        else:
            st.write("**Status:** No query expansion (expansion disabled or no RAG mode)")
            expansion_breakdown = {'count': 0}
            st.write("**Additional Tokens:** 0")
            st.write("**Additional Cost:** $0.000000")
        
        # 1c. Total Input Summary
        st.markdown("#### c) Total Input Cost")
        total_input_tokens = query_breakdown['count'] + expansion_breakdown['count']
        total_input_cost = Config.calculate_cost(model_name, total_input_tokens, 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Query", f"{query_breakdown['count']} tokens")
        with col2:
            st.metric("Query Expansion", f"+{expansion_breakdown['count']} tokens")
        with col3:
            st.metric("Total Input", f"{total_input_tokens} tokens")
        
        st.write(f"**📊 Input Math:** {query_breakdown['count']} + {expansion_breakdown['count']} = **{total_input_tokens} tokens** = **${total_input_cost['input_cost']:.6f}**")
        
        st.markdown("---")
        
        # 2. DOCUMENT CONTEXT BREAKDOWN
        if context_chunks:
            st.markdown("### 2. 📄 Document Context → Tokens Sent to AI")
            st.write("*These are the relevant chunks from your documents that provide context to the AI:*")
            
            total_context_tokens = 0
            for i, chunk in enumerate(context_chunks, 1):
                chunk_breakdown = Config.tokenize_text(chunk, model_name)
                total_context_tokens += chunk_breakdown['count']
                
                st.markdown(f"**Chunk {i}: {chunk_breakdown['count']} tokens**")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write("**Text:**")
                    st.text_area("", chunk, height=100, disabled=True, key=f"chunk_{i}")
                with col2:
                    st.write("**Token Preview (first 20 tokens):**")
                    preview_tokens = chunk_breakdown['tokens'][:20]
                    preview_display = ""
                    for token in preview_tokens:
                        escaped_token = token.replace('\n', '\\n').replace('\t', '\\t')
                        if escaped_token.strip() == '':
                            escaped_token = '[SPACE]'
                        preview_display += f"`{escaped_token}` "
                    if len(chunk_breakdown['tokens']) > 20:
                        preview_display += f"... +{len(chunk_breakdown['tokens']) - 20} more tokens"
                    st.markdown(preview_display)
                st.markdown("---")
            
            context_cost = Config.calculate_cost(model_name, total_context_tokens, 0)
            st.write(f"**Context Cost:** {total_context_tokens} tokens × ${Config.get_cost_per_token(model_name, 'input'):.8f} = **${context_cost['input_cost']:.6f}**")
            
            st.markdown("---")
        else:
            total_context_tokens = 0
        
        # 3. AI RESPONSE BREAKDOWN
        st.markdown("### 3. 🤖 AI Response → Output Tokens")
        response_breakdown = Config.tokenize_text(response_text, model_name)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("**Response Text:**")
            st.text_area("", response_text, height=150, disabled=True, key="response_display")
        with col2:
            st.write(f"**Token Preview ({response_breakdown['count']} tokens):**")
            # Show first 30 tokens of response
            preview_tokens = response_breakdown['tokens'][:30]
            preview_display = ""
            for token in preview_tokens:
                escaped_token = token.replace('\n', '\\n').replace('\t', '\\t')
                if escaped_token.strip() == '':
                    escaped_token = '[SPACE]'
                preview_display += f"`{escaped_token}` "
            if len(response_breakdown['tokens']) > 30:
                preview_display += f"... +{len(response_breakdown['tokens']) - 30} more tokens"
            st.markdown(preview_display)
        
        response_cost = Config.calculate_cost(model_name, 0, response_breakdown['count'])
        st.write(f"**Response Cost:** {response_breakdown['count']} tokens × ${Config.get_cost_per_token(model_name, 'output'):.8f} = **${response_cost['output_cost']:.6f}**")
        
        st.markdown("---")
        
        # 4. TOTAL BREAKDOWN
        st.markdown("### 💰 Complete Cost Calculation")
        query_tokens = query_breakdown['count']
        expansion_tokens = expansion_breakdown['count']
        context_tokens = total_context_tokens
        output_tokens = response_breakdown['count']
        
        total_input_tokens = query_tokens + expansion_tokens + context_tokens
        total_cost_data = Config.calculate_cost(model_name, total_input_tokens, output_tokens)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Input", f"{total_input_tokens:,} tokens", 
                     help=f"Query: {query_tokens} + Expansion: {expansion_tokens} + Context: {context_tokens}")
        with col2:
            st.metric("Total Output", f"{output_tokens:,} tokens")
        with col3:
            st.metric("Grand Total Cost", f"${total_cost_data['total_cost']:.6f}")
        
        # Crystal clear final breakdown
        st.markdown(f"""
        **🧮 Final Cost Breakdown:**
        - **Input Costs:**
          - Your Question: {query_tokens} tokens = ${Config.calculate_cost(model_name, query_tokens, 0)['input_cost']:.6f}
          - Query Expansion: {expansion_tokens} tokens = ${Config.calculate_cost(model_name, expansion_tokens, 0)['input_cost']:.6f}
          - Document Context: {context_tokens} tokens = ${Config.calculate_cost(model_name, context_tokens, 0)['input_cost']:.6f}
          - **Input Subtotal: {total_input_tokens} tokens = ${total_cost_data['input_cost']:.6f}**
        
        - **Output Costs:**
          - AI Response: {output_tokens} tokens = ${Config.calculate_cost(model_name, 0, output_tokens)['output_cost']:.6f}
        
        **🎯 GRAND TOTAL: {total_input_tokens + output_tokens} tokens = ${total_cost_data['total_cost']:.6f}**
        """)
        
        st.success("🔍 **Now you can see exactly where every token comes from!** Your question gets tokenized, sometimes expanded for better search, combined with relevant document chunks, and sent to the AI. The AI's response also gets tokenized for the output cost.")

# =====================
# CONFIGURATION KNOBS (now using Config class)
# =====================
# These are now loaded from config.py and can be overridden by environment variables

# =============================================================================
# Model options - DEEPSEEK LLM + OPENAI EMBEDDINGS (hybrid approach)
# =============================================================================

# DeepSeek model options (PRIMARY for LLM)
model_options = [
    ("deepseek-chat", "DeepSeek Chat"),
    ("deepseek-coder", "DeepSeek Coder"),
    ("deepseek-reasoner", "DeepSeek Reasoner")
]
model_display_names = {
    "deepseek-chat": "DeepSeek Chat",
    "deepseek-coder": "DeepSeek Coder",
    "deepseek-reasoner": "DeepSeek Reasoner"
}

# OpenAI model options (PRESERVED FOR POTENTIAL RESTORATION)
# model_options = [
#     ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
#     ("gpt-4", "GPT-4"),
#     ("gpt-4-turbo-preview", "GPT-4 Turbo")
# ]
# model_display_names = {
#     "gpt-3.5-turbo": "GPT-3.5 Turbo",
#     "gpt-4": "GPT-4", 
#     "gpt-4-turbo-preview": "GPT-4 Turbo"
# }

# Functions from main.py (adapted for OpenAI embeddings)
def get_files(sample_docs_dir=None):
    """Return list of .txt and .md files in sample_docs directory."""
    if sample_docs_dir is None:
        sample_docs_dir = Config.SAMPLE_DOCS_DIR
    
    if not os.path.exists(sample_docs_dir):
        return []
    return [
        f for f in os.listdir(sample_docs_dir)
        if f.endswith(".txt") or f.endswith(".md")
    ]

def read_text_file(filepath):
    """Read a text file as UTF-8, return content or None on error."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading {filepath}: {e}")
        return None

def chunk_text(text, chunk_size=None, overlap=None):
    """Split text into overlapping chunks."""
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if overlap is None:
        overlap = Config.CHUNK_OVERLAP
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def store_embeddings(collection, chunks, embeddings, metadatas=None):
    """Store embeddings with optional metadata for each chunk."""
    if metadatas is None:
        metadatas = [{"source": "unknown"} for _ in chunks]
    
    # Generate unique IDs using timestamp and index to avoid collisions
    import time
    timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
    unique_ids = [f"chunk_{timestamp}_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=unique_ids
    )

# =============================================================================
# HYBRID FUNCTIONS: OPENAI EMBEDDINGS + DEEPSEEK LLM
# =============================================================================

def embed_chunks_openai(chunks):
    """Given a list of text chunks, return a list of embedding vectors using OpenAI.
    NOTE: Using OpenAI for embeddings since DeepSeek doesn't provide embedding API."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set (required for embeddings).")
    
    client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # OpenAI has a limit on batch size, so we'll process in batches
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [embedding.embedding for embedding in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def embed_query_openai(query):
    """Embed a single query using OpenAI.
    NOTE: Using OpenAI for embeddings since DeepSeek doesn't provide embedding API."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set (required for embeddings).")
    
    client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
    response = client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=[query]
    )
    return response.data[0].embedding

def ask_deepseek(context_chunks, user_query, model=None, max_tokens=None, temperature=None, top_p=None, system_prompt=None, stop=None):
    """Ask DeepSeek model with context chunks and user query. Returns (response, usage_info)"""
    # Use Config defaults if parameters not provided
    if model is None:
        model = Config.DEFAULT_LLM_MODEL
    if max_tokens is None:
        max_tokens = Config.LLM_MAX_TOKENS
    if temperature is None:
        temperature = Config.LLM_TEMPERATURE
    if top_p is None:
        top_p = Config.LLM_TOP_P
    if system_prompt is None:
        system_prompt = Config.SYSTEM_PROMPT
        
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    if not Config.DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")
    
    client = openai.OpenAI(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com/v1"
    )
    
    # Build the request parameters
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    # Only add stop parameter if it's not None
    if stop is not None:
        request_params["stop"] = stop
    
    response = client.chat.completions.create(**request_params)
    
    # Extract usage information
    usage_info = {
        'input_tokens': response.usage.prompt_tokens if response.usage else 0,
        'output_tokens': response.usage.completion_tokens if response.usage else 0,
        'total_tokens': response.usage.total_tokens if response.usage else 0
    }
    
    return response.choices[0].message.content.strip(), usage_info

# =============================================================================
# OPENAI FUNCTIONS (PRESERVED FOR POTENTIAL RESTORATION)
# =============================================================================

# def embed_chunks_openai(chunks):
#     """Given a list of text chunks, return a list of embedding vectors using OpenAI."""
#     if not Config.OPENAI_API_KEY:
#         raise ValueError("OPENAI_API_KEY environment variable not set.")
#     
#     client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
#     
#     # OpenAI has a limit on batch size, so we'll process in batches
#     batch_size = 100
#     all_embeddings = []
#     
#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i + batch_size]
#         response = client.embeddings.create(
#             model=Config.EMBEDDING_MODEL,
#             input=batch
#         )
#         batch_embeddings = [embedding.embedding for embedding in response.data]
#         all_embeddings.extend(batch_embeddings)
#     
#     return all_embeddings

# def embed_query_openai(query):
#     """Embed a single query using OpenAI."""
#     if not Config.OPENAI_API_KEY:
#         raise ValueError("OPENAI_API_KEY environment variable not set.")
#     
#     client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
#     response = client.embeddings.create(
#         model=Config.EMBEDDING_MODEL,
#         input=[query]
#     )
#     return response.data[0].embedding

# def ask_openai(context_chunks, user_query, model=None, max_tokens=None, temperature=None, top_p=None, system_prompt=None, stop=None):
#     # Use Config defaults if parameters not provided
#     if model is None:
#         model = Config.DEFAULT_LLM_MODEL
#     if max_tokens is None:
#         max_tokens = Config.LLM_MAX_TOKENS
#     if temperature is None:
#         temperature = Config.LLM_TEMPERATURE
#     if top_p is None:
#         top_p = Config.LLM_TOP_P
#     if system_prompt is None:
#         system_prompt = Config.SYSTEM_PROMPT
#         
#     context = "\n".join(context_chunks)
#     prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
#     
#     if not Config.OPENAI_API_KEY:
#         raise ValueError("OPENAI_API_KEY environment variable not set.")
#     client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
#     
#     # Build the request parameters
#     request_params = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ],
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#         "top_p": top_p,
#     }
#     
#     # Only add stop parameter if it's not None
#     if stop is not None:
#         request_params["stop"] = stop
#     
#     response = client.chat.completions.create(**request_params)
#     return response.choices[0].message.content.strip()

# =============================================================================
# ACTIVE FUNCTION ALIASES (Hybrid approach: OpenAI embeddings + DeepSeek LLM)
# =============================================================================

# Active function names that the rest of the app uses
embed_chunks = embed_chunks_openai  # Using OpenAI for embeddings (DeepSeek doesn't offer embedding API)
embed_query = embed_query_openai    # Using OpenAI for embeddings (DeepSeek doesn't offer embedding API)
ask_llm = ask_deepseek             # Using DeepSeek for LLM

def expand_query(query):
    """
    Expand a natural language query into multiple keyword variations
    to improve retrieval for complex questions, including multi-topic queries.
    """
    # Convert to lowercase for processing
    query_lower = query.lower()
    
    # Common query expansions
    expansions = [query]  # Always include original
    
    # Multi-topic detection and expansion
    topics = []
    if 'international space station' in query_lower or 'iss' in query_lower:
        topics.append('international space station')
    if 'golden gate bridge' in query_lower:
        topics.append('golden gate bridge')
    if 'hoover dam' in query_lower:
        topics.append('hoover dam')
    if 'empire state building' in query_lower:
        topics.append('empire state building')
    
    # If multiple topics detected, create individual queries for each
    if len(topics) > 1:
        for topic in topics:
            # Create topic-specific versions of the query
            topic_query = query_lower.replace('combine', '').replace('both', '').replace('and', '').strip()
            expansions.append(f"{topic_query} {topic}")
            
            # Add specific expansions for each topic
            if 'date' in query_lower or 'when' in query_lower or 'timeline' in query_lower:
                expansions.append(f"{topic} dates")
                expansions.append(f"{topic} timeline")
                expansions.append(f"{topic} construction")
                expansions.append(f"{topic} history")
    
    # Single topic expansions
    # Height/tall variations
    if any(word in query_lower for word in ['tall', 'height', 'high']):
        if 'golden gate bridge' in query_lower:
            expansions.append("golden gate bridge tower height")
            expansions.append("tower height 746 feet")
            expansions.append("bridge tower dimensions")
    
    # Length/long variations  
    if any(word in query_lower for word in ['long', 'length']):
        if 'golden gate bridge' in query_lower:
            expansions.append("golden gate bridge length")
            expansions.append("bridge span length")
            expansions.append("total length")
    
    # Construction/dates variations
    if any(word in query_lower for word in ['built', 'construction', 'when', 'date', 'timeline']):
        if 'golden gate bridge' in query_lower:
            expansions.append("golden gate bridge construction")
            expansions.append("bridge built 1937")
            expansions.append("construction started 1933")
        if 'international space station' in query_lower or 'iss' in query_lower:
            expansions.append("international space station construction")
            expansions.append("ISS timeline")
            expansions.append("space station modules")
            expansions.append("ISS assembly")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expansions = []
    for exp in expansions:
        if exp not in seen:
            seen.add(exp)
            unique_expansions.append(exp)
    
    return unique_expansions

def retrieve_with_query_expansion(collection, query, n_results=3):
    """
    Retrieve chunks using query expansion for better coverage.
    Returns chunks, scores, metadatas, and the actual expanded queries used.
    """
    expanded_queries = expand_query(query)
    all_chunks = []
    all_scores = []
    all_metadatas = []
    seen_chunks = set()
    queries_used = []  # Track which queries actually returned results
    
    for exp_query in expanded_queries:
        try:
            query_embedding = embed_query(exp_query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "distances", "metadatas"]
            )
            
            chunks = results.get("documents", [[]])[0]
            scores = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            # Only count this query as "used" if it returned results
            if chunks:
                queries_used.append(exp_query)
            
            # Add unique chunks
            for chunk, score, metadata in zip(chunks, scores, metadatas):
                if chunk not in seen_chunks:
                    all_chunks.append(chunk)
                    all_scores.append(score)
                    all_metadatas.append(metadata)
                    seen_chunks.add(chunk)
        except Exception as e:
            continue  # Skip failed expansions
    
    # Sort by score and return top n_results
    if all_chunks:
        combined = list(zip(all_chunks, all_scores, all_metadatas))
        combined.sort(key=lambda x: x[1])  # Sort by score (lower is better)
        
        # Return top n_results
        top_results = combined[:n_results]
        return ([chunk for chunk, _, _ in top_results], 
                [score for _, score, _ in top_results],
                [metadata for _, _, metadata in top_results],
                queries_used)  # Return the queries that were actually used
    
    return [], [], [], queries_used

def delete_document_by_source(collection, source_filename):
    """Delete all chunks from a specific document source."""
    try:
        # Get all data to find chunks from this source (IDs returned by default)
        all_data = collection.get(include=["documents", "metadatas"])
        
        # Find IDs of chunks from this source
        ids_to_delete = []
        if all_data.get("metadatas"):
            for i, metadata in enumerate(all_data["metadatas"]):
                if metadata.get("source") == source_filename:
                    ids_to_delete.append(all_data["ids"][i])
        
        # Delete the chunks
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        return 0
    except Exception as e:
        st.error(f"Error deleting document {source_filename}: {e}")
        return 0

# Initialize session state
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = []
if 'files_processed' not in st.session_state:
    st.session_state.files_processed = False
if 'test_results' not in st.session_state:
    st.session_state.test_results = []
if 'document_metadata' not in st.session_state:
    st.session_state.document_metadata = {}  # Track document info: {filename: {chunks: count, upload_time: datetime, etc.}}

# Cost tracking session state
if 'session_costs' not in st.session_state:
    st.session_state.session_costs = {
        'embedding_costs': [],  # For document processing: [{'tokens': int, 'cost': float, 'model': str, 'timestamp': str}]
        'query_costs': [],      # For individual queries: [{'original_tokens': int, 'expansion_tokens': int, 'input_cost': float, 'output_tokens': int, 'output_cost': float, 'total_cost': float, 'model': str, 'timestamp': str}]
        'total_embedding_cost': 0.0,
        'total_query_cost': 0.0
    }

# Try to recover existing database on startup
if not st.session_state.files_processed:
    try:
        client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        existing_collections = client.list_collections()
        
        # Check if our collection exists
        collection_exists = any(col.name == "simple_chunks" for col in existing_collections)
        
        if collection_exists:
            st.session_state.collection = client.get_collection("simple_chunks")
            
            # Get all documents from the collection to rebuild all_chunks and metadata
            all_data = st.session_state.collection.get(include=["documents", "metadatas"])
            if all_data["documents"]:
                st.session_state.all_chunks = all_data["documents"]
                
                # Rebuild document metadata from stored metadata
                st.session_state.document_metadata = {}
                if all_data.get("metadatas"):
                    for metadata in all_data["metadatas"]:
                        source = metadata.get("source", "unknown")
                        if source not in st.session_state.document_metadata:
                            st.session_state.document_metadata[source] = {
                                "chunks": 0,
                                "upload_time": metadata.get("upload_time", "unknown"),
                                "total_size": 0
                            }
                        st.session_state.document_metadata[source]["chunks"] += 1
                        st.session_state.document_metadata[source]["total_size"] += metadata.get("chunk_size", 0)
                
                st.session_state.files_processed = True
                st.session_state.database_recovered = True
                
    except Exception as e:
        # If there's any error, just continue with empty state
        pass

# =====================
# MAIN UI - TAB STRUCTURE
# =====================

# Display title with DGT logo
try:
    logo_data = __import__('base64').b64encode(open(Config.LOGO_PATH, 'rb').read()).decode()
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <img src="data:image/webp;base64,{}" width="60" style="margin-right: 15px;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 600;">RAG Chatbot Testing Interface</h1>
    </div>
    """.format(logo_data), unsafe_allow_html=True)
except FileNotFoundError:
    # Fallback if logo not found
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 600;">🤖 RAG Chatbot Testing Interface</h1>
    </div>
    """, unsafe_allow_html=True)
st.write("Compare AI responses with and without document context (RAG)")



# Display manual in full-width container when requested
if st.session_state.get('show_manual', False):
    st.markdown("---")
    
    # Manual header with close button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("# 📖 RAG Chatbot Learning Tool - User Manual")
    with col2:
        if st.button("❌ Close"):
            st.session_state.show_manual = False
            st.rerun()
    
    # Display the manual content
    try:
        with open("user_manual.md", "r", encoding="utf-8") as f:
            manual_content = f.read()
        
        # Remove the title from content since we already show it above
        lines = manual_content.split('\n')
        # Skip the first few lines that contain the title
        content_start = 0
        for i, line in enumerate(lines):
            if line.startswith('---') and i > 0:
                content_start = i + 1
                break
        
        if content_start > 0:
            manual_content = '\n'.join(lines[content_start:])
        
        st.markdown(manual_content)
        
        # Close button at bottom
        st.markdown("---")
        if st.button("❌ Close", key="close_manual_bottom"):
            st.session_state.show_manual = False
            st.rerun()
            
    except FileNotFoundError:
        st.error("User manual not found. Please ensure user_manual.md exists in the project directory.")
    except Exception as e:
        st.error(f"Error loading user manual: {e}")
    
    # Stop here - don't show the rest of the interface when manual is open
    st.stop()

# Show database recovery notification
if st.session_state.get('database_recovered', False):
    doc_count = len(st.session_state.document_metadata)
    doc_names = list(st.session_state.document_metadata.keys())
    st.success(f"🔄 **Database recovered!** Found existing vector database with {len(st.session_state.all_chunks)} chunks from {doc_count} documents: {', '.join(doc_names)}")
    if st.button("✅ Dismiss"):
        st.session_state.database_recovered = False
        st.rerun()



# Status indicator
col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1.5])
with col1:
    if st.session_state.files_processed:
        st.success("📄 RAG Enabled")
    else:
        st.info("💭 General Mode")

with col2:
    tooltip_metric("Documents", len(st.session_state.all_chunks) if st.session_state.files_processed else 0, 
                  "Number of text chunks available for RAG retrieval")

with col3:
    tooltip_metric("Tests Run", len(st.session_state.test_results), 
                  "Total number of questions asked and answers generated")

with col4:
    # User Manual button integrated into status row
    st.write("")  # Add some spacing to align with metrics
    if st.button("📖 Manual", help="Complete guide to using every feature of this RAG testing tool", use_container_width=True):
        st.session_state.show_manual = True
        st.rerun()

# Cost tracking display (only if RAG is enabled and there are query costs)
if st.session_state.files_processed and st.session_state.session_costs['total_query_cost'] > 0:
    st.markdown("### 💰 Session Costs")
    
    # Cost overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Query Costs", 
            f"${st.session_state.session_costs['total_query_cost']:.6f}",
            help="Cost for all questions asked this session (DeepSeek LLM)"
        )
    with col2:
        queries_count = len(st.session_state.session_costs['query_costs'])
        st.metric(
            "Queries", 
            f"{queries_count}",
            help="Number of questions asked this session"
        )
    with col3:
        avg_cost = st.session_state.session_costs['total_query_cost'] / max(1, queries_count)
        st.metric(
            "Avg Cost/Query", 
            f"${avg_cost:.6f}",
            help="Average cost per question"
        )
    with col4:
        if st.button("💡 Cost Details", help="View detailed cost breakdown for this session"):
            st.session_state.show_cost_details = not st.session_state.get('show_cost_details', False)
    
    # Detailed cost breakdown (if toggled)
    if st.session_state.get('show_cost_details', False):
        with st.expander("💰 Detailed Session Cost Breakdown", expanded=True):
            # DeepSeek model pricing reference
            st.write("**📊 Current Pricing:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("• **DeepSeek Models:** $0.14 per 1M input tokens")
            with col2:
                st.write("• **DeepSeek Models:** $0.28 per 1M output tokens")
            
            st.divider()
            
            # Query cost history
            if st.session_state.session_costs['query_costs']:
                st.write("**📈 Query Cost History:**")
                for i, cost in enumerate(st.session_state.session_costs['query_costs'], 1):
                    st.markdown(f"**Query {i} - ${cost['total_cost']:.6f}**")
                    
                    # Show original query
                    st.write(f"**Original Query:** \"{cost.get('original_query', 'N/A')}\"")
                    
                    # Show expanded queries if any
                    expanded_queries = cost.get('expanded_queries', [])
                    if len(expanded_queries) > 1:
                        st.write("**Query Expansion:**")
                        for j, exp_query in enumerate(expanded_queries, 1):
                            if j == 1:
                                st.write(f"  {j}. {exp_query} *(original)*")
                            else:
                                st.write(f"  {j}. {exp_query}")
                    else:
                        st.write("**Query Expansion:** None (expansion disabled or no RAG)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Input:** {cost['original_tokens']:,} tokens")
                        if cost.get('expansion_tokens', 0) > 0:
                            st.write(f"**Expansion:** +{cost['expansion_tokens']:,} tokens")
                            st.write(f"**Total Input:** {cost['original_tokens'] + cost['expansion_tokens']:,} tokens")
                        st.write(f"**Input Cost:** ${cost['input_cost']:.6f}")
                    with col2:
                        st.write(f"**Output:** {cost['output_tokens']:,} tokens")
                        st.write(f"**Output Cost:** ${cost['output_cost']:.6f}")
                        st.write(f"**Model:** {cost['model']}")
                        st.write(f"**Time:** {cost['timestamp']}")
                    st.markdown("---")
            
            # Clear costs button
            if st.button("🗑️ Clear Session Costs", help="Reset all cost tracking for this session"):
                st.session_state.session_costs = {
                    'embedding_costs': [],
                    'query_costs': [],
                    'total_embedding_cost': 0.0,
                    'total_query_cost': 0.0
                }
                st.session_state.show_cost_details = False
                st.success("Session costs cleared!")
                st.rerun()

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Ask Questions", "📁 Documents", "⚙️ Parameters", "🔍 RAG Details", "📊 Test Results", "🧮 Vector Explorer"])

# =====================
# TAB 1: ASK QUESTIONS
# =====================
with tab1:
    st.header("Ask Questions")
    
    # Instructions with tooltip
    st.info("💡 **Try this:** Ask a question first to see the baseline AI response, then upload documents and ask the same question to see how RAG improves the answer!")
    
    # Test case input
    col1, col2 = st.columns([3, 1])
    with col1:
        # Initialize query in session state if not exists
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""
            
        user_query = st.text_area(
            "Ask any question:", 
            value=st.session_state.current_query,
            height=100,
            help="Enter your question here. The system will answer using general knowledge if no documents are uploaded, or use RAG if documents are available.",
            key="query_input"
        )
        
        # Update session state
        st.session_state.current_query = user_query
            
    with col2:
        test_name = st.text_input("Test Name (optional)", 
                                help="Give this test a name to easily identify it in the results history.")
        
        # Clear button - matches the width of the text input above
        if st.button("🗑️ Clear Query", help="Clear the question text area", use_container_width=True):
            st.session_state.current_query = ""
            st.rerun()
    
    # Model selection with tooltip
    selected_model_idx = tooltip_selectbox(
        "Select DeepSeek Model",
        range(len(model_options)),
        "Choose which DeepSeek model to use for generating answers. DeepSeek Chat is optimized for general queries, DeepSeek Coder excels at code-related tasks, and DeepSeek Reasoner provides step-by-step reasoning.",
        format_func=lambda x: model_options[x][1]
    )
    selected_model = model_options[selected_model_idx][0]
    
    # Get Answer button
    if st.button("🚀 Get Answer", type="primary"):
        if not user_query:
            st.error("Please enter a question!")
        else:
            # Get current parameters from session state (set in Parameters tab)
            temperature = st.session_state.get('temperature', Config.LLM_TEMPERATURE)
            max_tokens = st.session_state.get('max_tokens', Config.LLM_MAX_TOKENS)
            top_p = st.session_state.get('top_p', Config.LLM_TOP_P)
            system_prompt = st.session_state.get('system_prompt', Config.SYSTEM_PROMPT)
            chunk_size = st.session_state.get('chunk_size', Config.CHUNK_SIZE)
            chunk_overlap = st.session_state.get('chunk_overlap', Config.CHUNK_OVERLAP)
            n_results = st.session_state.get('n_results', Config.N_RESULTS)
            
            # Determine if we have documents for RAG
            has_documents = st.session_state.files_processed and st.session_state.collection is not None
            
            if has_documents:
                st.info("🔍 **Using RAG**: Searching your documents for relevant context...")
                
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        # Check if query expansion is enabled
                        use_expansion = st.session_state.get('use_query_expansion', True)
                        
                        if use_expansion:
                            # Use query expansion for better retrieval
                            relevant_chunks, scores, metadatas, queries_used = retrieve_with_query_expansion(
                                st.session_state.collection, 
                                user_query, 
                                n_results
                            )
                            # Use the actual queries that were used for retrieval
                            expanded_queries = queries_used if queries_used else [user_query]
                        else:
                            relevant_chunks, scores, metadatas = [], [], []
                            expanded_queries = [user_query]  # No expansion when disabled
                        
                        # Fallback to standard retrieval if expansion fails or is disabled  
                        if not relevant_chunks:
                            query_embedding = embed_query(user_query)
                            results = st.session_state.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=n_results,
                                include=["documents", "distances", "metadatas"]
                            )
                            relevant_chunks = results.get("documents", [[]])[0]
                            scores = results.get("distances", [[]])[0]
                            metadatas = results.get("metadatas", [[]])[0]
                            # When falling back, we only used the original query
                            expanded_queries = [user_query]
                        
                        # Store retrieval results for RAG Details tab
                        st.session_state.last_retrieval = {
                            'query': user_query,
                            'expanded_queries': expanded_queries,  # Use the actual queries that were used
                            'chunks': relevant_chunks,
                            'scores': scores,
                            'metadatas': metadatas,
                            'n_results': n_results
                        }
                        
                        # Get RAG answer
                        model_name = model_display_names.get(selected_model, selected_model)
                        
                        try:
                            llm_answer, usage_info = ask_llm(
                                relevant_chunks, 
                                user_query,
                                model=selected_model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                system_prompt=system_prompt,
                                stop=None
                            )
                            
                            # Track costs for this query
                            from datetime import datetime
                            expanded_queries = st.session_state.last_retrieval.get('expanded_queries', [user_query])
                            
                            # Calculate original query vs expansion tokens
                            original_tokens = Config.count_tokens(user_query, selected_model)
                            if len(expanded_queries) > 1:
                                expansion_text = " ".join(expanded_queries[1:])
                                expansion_tokens = Config.count_tokens(expansion_text, selected_model)
                            else:
                                expansion_tokens = 0
                            
                            # Calculate costs
                            total_input_tokens = original_tokens + expansion_tokens
                            output_tokens = usage_info['output_tokens']
                            cost_data = Config.calculate_cost(selected_model, total_input_tokens, output_tokens)
                            
                            # Store cost tracking
                            cost_record = {
                                'original_query': user_query,
                                'expanded_queries': expanded_queries,
                                'original_tokens': original_tokens,
                                'expansion_tokens': expansion_tokens,
                                'input_cost': cost_data['input_cost'],
                                'output_tokens': output_tokens,
                                'output_cost': cost_data['output_cost'],
                                'total_cost': cost_data['total_cost'],
                                'model': selected_model,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            st.session_state.session_costs['query_costs'].append(cost_record)
                            st.session_state.session_costs['total_query_cost'] += cost_data['total_cost']
                            
                            # Display the answer
                            st.subheader("🤖 RAG-Enhanced Answer")
                            st.markdown(f"""
                            <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 0.375rem; padding: 1rem; margin: 1rem 0;">
                                <div style="color: #155724; white-space: pre-wrap;">{llm_answer}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display cost breakdown for this query
                            display_query_cost_breakdown(user_query, expanded_queries, output_tokens, selected_model)
                            
                            # Display comprehensive token breakdown for education
                            display_token_breakdown(user_query, expanded_queries, relevant_chunks, llm_answer, selected_model)
                            
                            # Quick context preview
                            st.write(f"**Context used:** {len(relevant_chunks)} chunks from your documents")
                            with st.expander("📄 View context chunks"):
                                for i, (chunk, score) in enumerate(zip(relevant_chunks, scores), 1):
                                    st.write(f"**Chunk {i}** (Score: {score:.4f})")
                                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                                    st.divider()
                            
                            answer_type = "RAG"
                            context_used = relevant_chunks
                            
                        except Exception as e:
                            st.error(f"DeepSeek error: {e}")
                            llm_answer = "Error: Could not get answer from DeepSeek."
                            answer_type = "Error"
                            context_used = []
                            usage_info = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                            
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        llm_answer = None
                        answer_type = "Error"
                        context_used = []
                        usage_info = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            else:
                st.info("💭 **No documents uploaded**: Answering based on AI's general knowledge only...")
                
                with st.spinner("Generating answer without document context..."):
                    try:
                        # Ask without context (non-RAG)
                        llm_answer, usage_info = ask_llm(
                            [],  # No context chunks
                            user_query,
                            model=selected_model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            system_prompt=system_prompt,
                            stop=None
                        )
                        
                        # Track costs for this query (no expansion in non-RAG mode)
                        from datetime import datetime
                        original_tokens = Config.count_tokens(user_query, selected_model)
                        output_tokens = usage_info['output_tokens']
                        cost_data = Config.calculate_cost(selected_model, original_tokens, output_tokens)
                        
                        # Store cost tracking
                        cost_record = {
                            'original_query': user_query,
                            'expanded_queries': [user_query],  # No expansion in non-RAG mode
                            'original_tokens': original_tokens,
                            'expansion_tokens': 0,
                            'input_cost': cost_data['input_cost'],
                            'output_tokens': output_tokens,
                            'output_cost': cost_data['output_cost'],
                            'total_cost': cost_data['total_cost'],
                            'model': selected_model,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        st.session_state.session_costs['query_costs'].append(cost_record)
                        st.session_state.session_costs['total_query_cost'] += cost_data['total_cost']
                        
                        # Display the answer
                        st.subheader("🤖 General Knowledge Answer")
                        st.markdown(f"""
                        <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 0.375rem; padding: 1rem; margin: 1rem 0;">
                            <div style="color: #0c5460; white-space: pre-wrap;">{llm_answer}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("💡 Upload documents to see how RAG can provide more specific, context-aware answers!")
                        
                        # Display cost breakdown for this query
                        display_query_cost_breakdown(user_query, [user_query], output_tokens, selected_model)
                        
                        # Display comprehensive token breakdown for education
                        display_token_breakdown(user_query, [user_query], [], llm_answer, selected_model)
                        
                        answer_type = "General"
                        context_used = []
                        
                    except Exception as e:
                        st.error(f"DeepSeek error: {e}")
                        llm_answer = "Error: Could not get answer from DeepSeek."
                        answer_type = "Error"
                        context_used = []
                        usage_info = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            
            # Save test result
            if llm_answer and not llm_answer.startswith("Error:"):
                test_result = {
                    'test_name': test_name or f"Test_{datetime.now().strftime('%H%M%S')}",
                    'question': user_query,
                    'model': selected_model,
                    'model_display_name': model_display_names.get(selected_model, selected_model),
                    'answer_type': answer_type,
                    'parameters': {
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'top_p': top_p,
                        'system_prompt': system_prompt,
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap,
                        'n_results': n_results
                    },
                    'context_used': context_used,
                    'answer': llm_answer,
                    'timestamp': datetime.now().isoformat(),
                    'usage_info': usage_info
                }
                st.session_state.test_results.append(test_result)

# =====================
# TAB 2: DOCUMENTS
# =====================
with tab2:
    st.header("📁 Document Management")
    
    # Current status
    if st.session_state.files_processed:
        st.success(f"✅ **Documents loaded:** {len(st.session_state.all_chunks)} chunks ready for RAG")
        
        # Show document statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            tooltip_metric("Total Chunks", len(st.session_state.all_chunks), 
                          "Number of text segments created from your documents")
        with col2:
            avg_chunk_size = sum(len(chunk) for chunk in st.session_state.all_chunks) / len(st.session_state.all_chunks) if st.session_state.all_chunks else 0
            tooltip_metric("Avg Chunk Size", f"{avg_chunk_size:.0f} chars", 
                          "Average number of characters per chunk")
        with col3:
            total_chars = sum(len(chunk) for chunk in st.session_state.all_chunks)
            tooltip_metric("Total Content", f"{total_chars:,} chars", 
                          "Total amount of text content available for retrieval")
        
        # Embedding costs display (one-time document processing costs)
        if st.session_state.session_costs['total_embedding_cost'] > 0:
            st.markdown("### 💰 Embedding Costs (Document Processing)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Cost", 
                    f"${st.session_state.session_costs['total_embedding_cost']:.6f}",
                    help="One-time cost for creating embeddings from your documents (OpenAI)"
                )
            with col2:
                total_embedding_tokens = sum(cost['tokens'] for cost in st.session_state.session_costs['embedding_costs'])
                st.metric(
                    "Tokens Used", 
                    f"{total_embedding_tokens:,}",
                    help="Total tokens processed for embeddings"
                )
            with col3:
                embeddings_count = len(st.session_state.session_costs['embedding_costs'])
                st.metric(
                    "Processing Sessions", 
                    f"{embeddings_count}",
                    help="Number of times documents were processed for embeddings"
                )
            with col4:
                current_rate = Config.get_cost_per_token(Config.EMBEDDING_MODEL, 'input')
                st.metric(
                    "Rate per Token", 
                    f"${current_rate:.8f}",
                    help=f"Current {Config.EMBEDDING_MODEL} rate per token"
                )
            
            # Detailed breakdown
            with st.expander("💡 Detailed Embedding Cost Breakdown"):
                st.write("**📊 Current Pricing:**")
                st.write(f"• **{Config.EMBEDDING_MODEL}:** ${Config.get_cost_per_token(Config.EMBEDDING_MODEL, 'input'):.8f} per token")
                
                st.divider()
                
                if st.session_state.session_costs['embedding_costs']:
                    st.write("**📈 Document Processing History:**")
                    for i, cost in enumerate(st.session_state.session_costs['embedding_costs'], 1):
                        st.markdown(f"**Processing {i} - ${cost['cost']:.6f} ({cost['files_processed']} files)**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Tokens:** {cost['tokens']:,}")
                            st.write(f"**Cost:** ${cost['cost']:.6f}")
                        with col2:
                            st.write(f"**Model:** {cost['model']}")
                            st.write(f"**Time:** {cost['timestamp']}")
                            st.write(f"**Files:** {cost['files_processed']}")
                        st.markdown("---")
        
        # Document Collection Overview
        st.subheader("📚 Document Collection")
        if st.session_state.document_metadata:
            st.write("**Documents in Vector Database:**")
            
            # Create a table showing document metadata
            doc_data = []
            for filename, metadata in st.session_state.document_metadata.items():
                doc_data.append({
                    "Document": filename,
                    "Chunks": metadata["chunks"],
                    "Total Size": f"{metadata.get('total_size', 0):,} chars",
                    "Upload Time": metadata["upload_time"]
                })
            
            # Display as a nice table
            df = pd.DataFrame(doc_data)
            st.dataframe(df, use_container_width=True)
            
            # Individual document management
            st.subheader("🗂️ Individual Document Management")
            selected_doc = st.selectbox(
                "Select document to manage:",
                options=list(st.session_state.document_metadata.keys()),
                help="Choose a document to view details or delete"
            )
            
            if selected_doc:
                doc_info = st.session_state.document_metadata[selected_doc]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Chunks", doc_info["chunks"])
                with col2:
                    st.metric("Size", f"{doc_info.get('total_size', 0):,} chars")
                with col3:
                    st.metric("Uploaded", doc_info["upload_time"])
                
                # Document actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"🗑️ Delete {selected_doc}", type="secondary", 
                                help=f"Remove all chunks from {selected_doc}"):
                        if st.session_state.collection:
                            deleted_count = delete_document_by_source(st.session_state.collection, selected_doc)
                            if deleted_count > 0:
                                # Update session state
                                del st.session_state.document_metadata[selected_doc]
                                
                                # Refresh all_chunks by getting from database
                                all_data = st.session_state.collection.get(include=["documents"])
                                st.session_state.all_chunks = all_data.get("documents", [])
                                
                                # Check if any documents remain
                                if not st.session_state.document_metadata:
                                    st.session_state.files_processed = False
                                
                                st.success(f"✅ Deleted {deleted_count} chunks from {selected_doc}")
                                st.rerun()
                            else:
                                st.error(f"Failed to delete {selected_doc}")
                
                with col2:
                    if st.button(f"👁️ View Chunks from {selected_doc}", 
                                help=f"Show all text chunks from {selected_doc}"):
                        st.session_state.show_doc_chunks = selected_doc
                
                with col3:
                    if st.button("📊 View All Chunks", help="See all document chunks in the database"):
                        st.session_state.show_all_chunks = True
        
        else:
            st.info("📄 No document metadata available. This may be from an older database version.")
        
        # Clear All Data functionality
        st.divider()
        st.subheader("🗑️ Database Management")
        
        # Enhanced warning section
        st.warning("""
        ⚠️ **DESTRUCTIVE ACTION WARNING**
        
        The "Clear All Data" button permanently deletes:
        • All vector embeddings and similarity search data
        • All uploaded document files (.txt, .md)
        • All test results and question/answer history
        • All document metadata and session data
        
        **This action CANNOT be undone!** 
        
        📖 See the User Manual (button above) for complete details on what gets deleted.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Data", type="secondary", 
                        help="⚠️ PERMANENT DELETION: Remove all documents, chunks, embeddings, and test results. This cannot be undone!"):
                st.session_state.show_clear_confirmation = True
        
        with col2:
            if st.session_state.get('show_clear_confirmation', False):
                st.error("⚠️ **FINAL WARNING: This will permanently delete ALL data!**")
                st.write("📖 Check the User Manual above for complete details on what gets deleted.")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Yes, Clear Everything", type="primary"):
                        try:
                            # Clear ChromaDB collection
                            if st.session_state.collection:
                                client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
                                try:
                                    client.delete_collection("simple_chunks")
                                except:
                                    pass  # Collection might not exist
                            
                            # Clear sample_docs directory
                            sample_docs_dir = str(Config.SAMPLE_DOCS_DIR)
                            if os.path.exists(sample_docs_dir):
                                for f in os.listdir(sample_docs_dir):
                                    if f.endswith(('.txt', '.md')):
                                        os.remove(os.path.join(sample_docs_dir, f))
                            
                            # Reset all session state
                            st.session_state.collection = None
                            st.session_state.all_chunks = []
                            st.session_state.files_processed = False
                            st.session_state.document_metadata = {}
                            st.session_state.test_results = []
                            st.session_state.last_retrieval = None
                            st.session_state.show_clear_confirmation = False
                            st.session_state.file_uploader_key += 1
                            
                            # Clear cost tracking
                            st.session_state.session_costs = {
                                'embedding_costs': [],
                                'query_costs': [],
                                'total_embedding_cost': 0.0,
                                'total_query_cost': 0.0
                            }
                            
                            # Clear any display states
                            if 'show_doc_chunks' in st.session_state:
                                del st.session_state.show_doc_chunks
                            if 'show_all_chunks' in st.session_state:
                                del st.session_state.show_all_chunks
                            if 'show_cost_details' in st.session_state:
                                del st.session_state.show_cost_details
                            
                            st.success("✅ All data cleared successfully! The system has been reset.")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error clearing data: {str(e)}")
                
                with col_b:
                    if st.button("❌ Cancel"):
                        st.session_state.show_clear_confirmation = False
                        st.rerun()

        
        # Show document-specific chunks if requested
        if st.session_state.get('show_doc_chunks'):
            selected_doc = st.session_state.show_doc_chunks
            st.subheader(f"📄 Chunks from {selected_doc}")
            
            # Get chunks from this document
            if st.session_state.collection:
                try:
                    all_data = st.session_state.collection.get(include=["documents", "metadatas"])
                    doc_chunks = []
                    
                    if all_data.get("metadatas"):
                        for i, metadata in enumerate(all_data["metadatas"]):
                            if metadata.get("source") == selected_doc:
                                chunk_text = all_data["documents"][i]
                                chunk_idx = metadata.get("chunk_index", i)
                                doc_chunks.append((chunk_idx, chunk_text))
                    
                    # Sort by chunk index
                    doc_chunks.sort(key=lambda x: x[0])
                    
                    for chunk_idx, chunk_text in doc_chunks:
                        with st.expander(f"Chunk {chunk_idx + 1} ({len(chunk_text)} characters)"):
                            st.text(chunk_text)
                    
                    if st.button("Hide Document Chunks"):
                        del st.session_state.show_doc_chunks
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error retrieving chunks: {e}")
        
        # Show all chunks if requested
        elif st.session_state.get('show_all_chunks', False):
            st.subheader("📋 All Document Chunks")
            for i, chunk in enumerate(st.session_state.all_chunks):
                with st.expander(f"Chunk {i+1} ({len(chunk)} characters)"):
                    st.text(chunk)
            
            if st.button("Hide All Chunks"):
                st.session_state.show_all_chunks = False
                st.rerun()
    
    else:
        st.info("💭 **No documents loaded** - Upload documents below to enable RAG")
    
    # Document upload section
    st.subheader("Upload New Documents")
    st.write("Upload documents to enable RAG and see how context improves answers")
    
    # Initialize file uploader key in session state if not exists
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=['txt', 'md'], 
        accept_multiple_files=True,
        help="Upload .txt or .md files. These will be processed into chunks for RAG retrieval.",
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )
    
    if uploaded_files:
        # Check for duplicate files already in database
        duplicate_files = []
        new_files = []
        
        for file in uploaded_files:
            if file.name in st.session_state.get('document_metadata', {}):
                duplicate_files.append(file.name)
            else:
                new_files.append(file)
        
        # Show warnings for duplicates
        if duplicate_files:
            st.warning(f"⚠️ **Duplicate files detected:** {', '.join(duplicate_files)}")
            st.write("These files are already in your database. Processing will **replace** the existing versions.")
        
        # Show upload preview
        st.write("**Files ready to process:**")
        for file in uploaded_files:
            status = "🔄 (will replace existing)" if file.name in duplicate_files else "🆕 (new)"
            st.write(f"- {file.name} ({file.size:,} bytes) {status}")
        
        # Get current chunking parameters
        chunk_size = st.session_state.get('chunk_size', Config.CHUNK_SIZE)
        chunk_overlap = st.session_state.get('chunk_overlap', Config.CHUNK_OVERLAP)
        
        st.info(f"Will process with current settings: {chunk_size} chars per chunk, {chunk_overlap} chars overlap")
        
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files to sample_docs directory
                    sample_docs_dir = str(Config.SAMPLE_DOCS_DIR)
                    os.makedirs(sample_docs_dir, exist_ok=True)
                    
                    # Clear existing files
                    for f in os.listdir(sample_docs_dir):
                        if f.endswith(('.txt', '.md')):
                            os.remove(os.path.join(sample_docs_dir, f))
                    
                    # Save new files
                    for uploaded_file in uploaded_files:
                        with open(os.path.join(sample_docs_dir, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Process files
                    files = get_files(sample_docs_dir)
                    st.write(f"**Files found:** {', '.join(files)}")
                    
                    # Read and chunk each file
                    all_chunks = []
                    all_metadatas = []
                    chunk_summary = []
                    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    for fname in files:
                        path = os.path.join(sample_docs_dir, fname)
                        content = read_text_file(path)
                        if content:
                            chunks = chunk_text(content, chunk_size, chunk_overlap)
                            chunk_summary.append(f"{fname}: {len(chunks)} chunks")
                            all_chunks.extend(chunks)
                            
                            # Create metadata for each chunk
                            for i, chunk in enumerate(chunks):
                                metadata = {
                                    "source": fname,
                                    "upload_time": upload_time,
                                    "chunk_index": i,
                                    "chunk_size": len(chunk),
                                    "total_chunks_in_doc": len(chunks)
                                }
                                all_metadatas.append(metadata)
                        else:
                            st.error(f"Skipping {fname} due to read error.")
                    
                    if chunk_summary:
                        st.write("**Chunking Summary:**")
                        for summary in chunk_summary:
                            st.write(f"- {summary}")
                        st.write(f"**Total chunks:** {len(all_chunks)}")
                    
                    if all_chunks:
                        # Create embeddings
                        with st.spinner("Creating embeddings with OpenAI..."):
                            embeddings = embed_chunks(all_chunks)
                            st.write(f"**Embeddings created:** {len(embeddings)} vectors of length {len(embeddings[0])}")
                            
                            # Track embedding costs
                            from datetime import datetime
                            total_text = " ".join(all_chunks)
                            total_tokens = Config.count_tokens(total_text, Config.EMBEDDING_MODEL)
                            embedding_cost_data = Config.calculate_cost(Config.EMBEDDING_MODEL, input_tokens=total_tokens, output_tokens=0)
                            
                            # Store embedding cost
                            embedding_record = {
                                'tokens': total_tokens,
                                'cost': embedding_cost_data['total_cost'],
                                'model': Config.EMBEDDING_MODEL,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'files_processed': len(files)
                            }
                            st.session_state.session_costs['embedding_costs'].append(embedding_record)
                            st.session_state.session_costs['total_embedding_cost'] += embedding_cost_data['total_cost']
                            
                            # Display embedding cost information
                            st.success(f"💰 **Embedding Cost:** ${embedding_cost_data['total_cost']:.6f} ({total_tokens:,} tokens @ ${Config.get_cost_per_token(Config.EMBEDDING_MODEL, 'input'):.8f} per token)")
                        
                        # Store in vector database
                        client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
                        
                        # Get or create collection
                        if st.session_state.collection is None:
                            try:
                                client.delete_collection("simple_chunks")
                            except:
                                pass
                            st.session_state.collection = client.get_or_create_collection("simple_chunks")
                        
                        # Delete existing documents if they're being replaced
                        for fname in files:
                            if fname in st.session_state.get('document_metadata', {}):
                                # Delete existing chunks from this document
                                deleted_count = delete_document_by_source(st.session_state.collection, fname)
                                # Remove from metadata tracking
                                if fname in st.session_state.document_metadata:
                                    del st.session_state.document_metadata[fname]
                                st.write(f"🔄 Replaced existing {fname} ({deleted_count} chunks removed)")
                        
                        # Add new documents to the collection
                        store_embeddings(st.session_state.collection, all_chunks, embeddings, all_metadatas)
                        
                        # Update document metadata tracking for new/updated documents
                        for metadata in all_metadatas:
                            source = metadata["source"]
                            if source not in st.session_state.document_metadata:
                                st.session_state.document_metadata[source] = {
                                    "chunks": 0,
                                    "upload_time": metadata["upload_time"],
                                    "total_size": 0
                                }
                            st.session_state.document_metadata[source]["chunks"] += 1
                            st.session_state.document_metadata[source]["total_size"] += metadata["chunk_size"]
                        
                        # Refresh all_chunks from the database to include all documents
                        all_data = st.session_state.collection.get(include=["documents"])
                        st.session_state.all_chunks = all_data.get("documents", [])
                        st.session_state.files_processed = True
                        
                        # Clear the file uploader by incrementing its key
                        st.session_state.file_uploader_key += 1
                        
                        st.success(f"✅ Successfully processed {len(files)} files into {len(all_chunks)} chunks! Now ask questions to see RAG in action.")
                        st.rerun()
                    else:
                        st.error("No chunks to embed.")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

# =====================
# TAB 3: PARAMETERS
# =====================
with tab3:
    st.header("⚙️ Parameter Configuration")
    
    # Model Parameters
    st.subheader("🤖 Model Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = tooltip_slider(
            "Temperature", 0.0, 1.0, 
            st.session_state.get('temperature', Config.LLM_TEMPERATURE), 0.1,
            "Controls randomness in responses. Lower = more deterministic, Higher = more creative",
            key="temp_slider"
        )
        st.session_state.temperature = temperature
        
        # Model-aware max tokens limit (DeepSeek supports up to 8192, OpenAI varies)
        max_tokens_limit = 8192  # DeepSeek V3 limit
        # If we were using OpenAI models, we'd need model-specific limits:
        # max_tokens_limit = Config.MODEL_CONFIGS.get(selected_model, {}).get('max_tokens', 4096)
        
        max_tokens = tooltip_slider(
            "Max Tokens", 64, max_tokens_limit, 
            st.session_state.get('max_tokens', Config.LLM_MAX_TOKENS), 64,
            f"Maximum length of the generated response. DeepSeek V3 supports up to {max_tokens_limit} tokens",
            key="tokens_slider"
        )
        st.session_state.max_tokens = max_tokens
    
    with col2:
        top_p = tooltip_slider(
            "Top P", 0.0, 1.0, 
            st.session_state.get('top_p', Config.LLM_TOP_P), 0.1,
            "Controls diversity via nucleus sampling. Lower = more focused, Higher = more diverse",
            key="top_p_slider"
        )
        st.session_state.top_p = top_p
    
    # System Prompt
    st.markdown("**System Prompt** ℹ️")
    st.caption("Instructions that control how the AI behaves. This prompt determines whether the AI uses only provided context or can use general knowledge.")
    
    system_prompt = st.text_area(
        "System Prompt", 
        value=st.session_state.get('system_prompt', Config.SYSTEM_PROMPT),
        height=150,
        help="Instructions that control how the AI behaves. This prompt determines whether the AI uses only provided context or can use general knowledge.",
        key="system_prompt_area",
        label_visibility="collapsed"
    )
    st.session_state.system_prompt = system_prompt
    
    st.divider()
    
    # RAG Parameters
    st.subheader("🔍 RAG Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = tooltip_slider(
            "Chunk Size", 100, 1000, 
            st.session_state.get('chunk_size', Config.CHUNK_SIZE), 50,
            "Size of text chunks in characters. Smaller = more specific, Larger = more context",
            key="chunk_size_slider"
        )
        st.session_state.chunk_size = chunk_size
        
        chunk_overlap = tooltip_slider(
            "Chunk Overlap", 0, 200, 
            st.session_state.get('chunk_overlap', Config.CHUNK_OVERLAP), 25,
            "Characters shared between adjacent chunks. Prevents information loss at boundaries",
            key="chunk_overlap_slider"
        )
        st.session_state.chunk_overlap = chunk_overlap
    
    with col2:
        n_results = tooltip_slider(
            "Number of Results", 1, 10, 
            st.session_state.get('n_results', Config.N_RESULTS), 1,
            "How many relevant chunks to retrieve for each query. More = richer context but potential noise",
            key="n_results_slider"
        )
        st.session_state.n_results = n_results
    
    # Query expansion toggle
    st.session_state.use_query_expansion = st.checkbox(
        "🔍 Enable Query Expansion", 
        value=st.session_state.get('use_query_expansion', True),
        help="Automatically expand natural language questions into multiple keyword variations for better retrieval. Helps with questions like 'how tall is...' vs 'tower height'"
    )
    
    # Parameter presets
    st.subheader("🎯 Parameter Presets")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Precise Mode", help="Optimized for factual, deterministic answers with DeepSeek"):
            st.session_state.temperature = 0.1
            st.session_state.max_tokens = 512  # Increased from 256 for DeepSeek's capabilities
            st.session_state.top_p = 0.9
            st.session_state.n_results = 3
            st.rerun()
    
    with col2:
        if st.button("⚖️ Balanced Mode", help="Good balance of accuracy and creativity with DeepSeek"):
            st.session_state.temperature = 0.4
            st.session_state.max_tokens = 1024  # Increased from 512 for DeepSeek's capabilities
            st.session_state.top_p = 1.0
            st.session_state.n_results = 5
            st.rerun()
    
    with col3:
        if st.button("🎨 Creative Mode", help="More creative and varied responses with DeepSeek"):
            st.session_state.temperature = 0.8
            st.session_state.max_tokens = 2048  # Increased from 1024 for DeepSeek's capabilities
            st.session_state.top_p = 1.0
            st.session_state.n_results = 7
            st.rerun()
    
    # Show current parameter summary
    st.subheader("📋 Current Configuration")
    config_data = {
        "Model Parameters": {
            "Temperature": st.session_state.get('temperature', Config.LLM_TEMPERATURE),
            "Max Tokens": st.session_state.get('max_tokens', Config.LLM_MAX_TOKENS),
            "Top P": st.session_state.get('top_p', Config.LLM_TOP_P)
        },
        "RAG Parameters": {
            "Chunk Size": st.session_state.get('chunk_size', Config.CHUNK_SIZE),
            "Chunk Overlap": st.session_state.get('chunk_overlap', Config.CHUNK_OVERLAP),
            "Results Retrieved": st.session_state.get('n_results', Config.N_RESULTS)
        }
    }
    st.json(config_data)

# =====================
# TAB 4: RAG DETAILS
# =====================
with tab4:
    st.header("🔍 RAG Retrieval Details")
    
    if 'last_retrieval' in st.session_state and st.session_state.last_retrieval is not None:
        retrieval = st.session_state.last_retrieval
        
        st.subheader(f"Last Query: \"{retrieval['query']}\"")
        
        # Show query expansion details - always show this section for clarity
        st.subheader("🔍 Query Expansion Analysis")
        if 'expanded_queries' in retrieval:
            expanded_queries = retrieval['expanded_queries']
            if len(expanded_queries) > 1:
                st.success("✅ **Query expansion used** - Your query was expanded for better search coverage")
                st.write("**Original query was expanded into these variations:**")
                for i, exp_query in enumerate(expanded_queries, 1):
                    if i == 1:
                        st.write(f"{i}. {exp_query} *(original)*")
                    else:
                        st.write(f"{i}. {exp_query} *(expansion)*")
                
                # Show if expansion helped
                unique_queries = len(set(expanded_queries))
                if unique_queries < len(expanded_queries):
                    st.info(f"💡 **Note:** Some expansions were duplicates. {unique_queries} unique queries were actually used.")
            else:
                st.warning("⚠️ **Query expansion attempted but no additional queries generated** - Only the original query was used")
                st.write(f"**Query used:** {expanded_queries[0] if expanded_queries else 'Unknown'}")
        else:
            st.info("📝 **Query expansion disabled** - Only the original query was used for retrieval")
        st.divider()
        
        # Document source analysis
        if retrieval.get('metadatas'):
            sources = {}
            for metadata in retrieval['metadatas']:
                source = metadata.get('source', 'Unknown')
                if source not in sources:
                    sources[source] = 0
                sources[source] += 1
            
            st.subheader("📚 Document Sources in Results")
            for source, count in sources.items():
                st.write(f"• **{source}**: {count} chunk(s)")
            
            if len(sources) > 1:
                st.success("✅ **Multi-document retrieval successful** - Found relevant chunks from multiple sources")
            elif len(sources) == 1:
                st.warning("⚠️ **Single-document retrieval** - All chunks came from one document. For multi-topic queries, try increasing 'Number of Results' in Parameters.")
            st.divider()
        
        # Retrieval statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            tooltip_metric("Chunks Retrieved", len(retrieval['chunks']), 
                          "Number of document chunks found relevant to the query")
        with col2:
            avg_score = sum(retrieval['scores']) / len(retrieval['scores']) if retrieval['scores'] else 0
            tooltip_metric("Avg Similarity", f"{avg_score:.4f}", 
                          "Average similarity score (lower = better match)")
        with col3:
            best_score = min(retrieval['scores']) if retrieval['scores'] else 0
            tooltip_metric("Best Match", f"{best_score:.4f}", 
                          "Similarity score of the most relevant chunk")
        
        # Detailed chunk analysis
        st.subheader("📄 Retrieved Chunks Analysis")
        
        # Scoring reference table
        st.write("**Quality Score Reference:**")
        
        # Create a simple table showing the scoring system
        col1, col2, col3 = st.columns([1, 2, 4])
        with col1:
            st.write("**Rating**")
        with col2:
            st.write("**Score Range**")
        with col3:
            st.write("**Description**")
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 4])
        with col1:
            st.write("🟢 Excellent")
        with col2:
            st.write("< 0.4")
        with col3:
            st.write("Highly relevant to your query")
        
        col1, col2, col3 = st.columns([1, 2, 4])
        with col1:
            st.write("🟡 Good")
        with col2:
            st.write("0.4 - 0.7")
        with col3:
            st.write("Contains relevant information")
        
        col1, col2, col3 = st.columns([1, 2, 4])
        with col1:
            st.write("🔴 Fair")
        with col2:
            st.write("> 0.7")
        with col3:
            st.write("Some relevance but may be noisy")
        
        st.write("*Note: Lower scores indicate better matches (measuring distance in vector space)*")
        st.divider()
        
        # Get metadata for retrieved chunks if available
        chunk_metadatas = retrieval.get('metadatas', [])
        
        for i, (chunk, score) in enumerate(zip(retrieval['chunks'], retrieval['scores']), 1):
            # Determine match quality - Fixed thresholds to be more accurate
            if score < 0.4:
                quality = "🟢 Excellent"
                quality_color = "success"
            elif score < 0.7:
                quality = "🟡 Good"
                quality_color = "warning"
            else:
                quality = "🔴 Fair"
                quality_color = "error"
            
            # Get metadata for this chunk if available
            metadata = chunk_metadatas[i-1] if i-1 < len(chunk_metadatas) else {}
            source_doc = metadata.get('source', 'Unknown')
            chunk_index = metadata.get('chunk_index', 'Unknown')
            
            with st.expander(f"Chunk {i} - {quality} (Score: {score:.4f}) - From: {source_doc}"):
                # Document source information
                if metadata:
                    st.write("**📄 Source Information:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Document:** {source_doc}")
                    with col2:
                        st.write(f"**Chunk Index:** {chunk_index + 1 if isinstance(chunk_index, int) else chunk_index}")
                    with col3:
                        upload_time = metadata.get('upload_time', 'Unknown')
                        st.write(f"**Uploaded:** {upload_time}")
                    st.divider()
                
                # Chunk content
                st.write("**📝 Content:**")
                st.text_area("Chunk Content", chunk, height=150, key=f"chunk_content_{i}", disabled=True, label_visibility="collapsed")
                
                # Technical details
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Length", f"{len(chunk)} chars")
                with col2:
                    st.metric("Words", f"{len(chunk.split())} words")
                with col3:
                    st.metric("Similarity", f"{score:.4f}")
                with col4:
                    total_chunks = metadata.get('total_chunks_in_doc', 'Unknown')
                    st.metric("Doc Chunks", total_chunks)
                
                # Similarity explanation
                if score < 0.4:
                    st.success("🎯 **Excellent match** - This chunk is highly relevant to your query")
                elif score < 0.7:
                    st.warning("👍 **Good match** - This chunk contains relevant information")
                else:
                    st.error("⚠️ **Fair match** - This chunk may contain some relevant information but could be noisy")
    
    else:
        st.info("🔍 **No retrieval data available**\n\nAsk a question in the 'Ask Questions' tab with documents uploaded to see detailed RAG retrieval information here.")
        
        # Show general RAG process info
        st.subheader("How RAG Retrieval Works")
        st.write("""
        1. **Query Embedding**: Your question is converted to a vector representation
        2. **Similarity Search**: The system finds document chunks with similar vectors
        3. **Ranking**: Chunks are ranked by similarity score (lower = better match)
        4. **Context Assembly**: Top chunks are combined as context for the AI
        5. **Answer Generation**: The AI uses only the retrieved context to answer
        """)

# =====================
# TAB 5: TEST RESULTS
# =====================
with tab5:
    st.header("📊 Test Results & History")
    
    if st.session_state.test_results:
        # Results summary
        total_tests = len(st.session_state.test_results)
        rag_tests = sum(1 for r in st.session_state.test_results if r['answer_type'] == 'RAG')
        general_tests = total_tests - rag_tests
        
        col1, col2, col3 = st.columns(3)
        with col1:
            tooltip_metric("Total Tests", total_tests, "Total number of questions asked")
        with col2:
            tooltip_metric("RAG Tests", rag_tests, "Tests that used document context")
        with col3:
            tooltip_metric("General Tests", general_tests, "Tests using only general knowledge")
        
        # Export all results
        if st.button("📥 Export All Results"):
            all_results_json = json.dumps(st.session_state.test_results, indent=2)
            st.download_button(
                label="Download All Results (JSON)",
                data=all_results_json,
                file_name=f"rag_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Clear results
        if st.button("🗑️ Clear All Results", help="Remove all test results from history"):
            st.session_state.test_results = []
            st.rerun()
        
        st.divider()
        
        # Individual results
        st.subheader("Individual Test Results")
        
        for i, result in enumerate(reversed(st.session_state.test_results)):
            answer_type_emoji = "🔍" if result['answer_type'] == "RAG" else "💭"
            timestamp = result['timestamp'][:19].replace('T', ' ')
            
            with st.expander(f"{answer_type_emoji} {result['test_name']} - {result['answer_type']} ({timestamp})"):
                # Question and Answer
                st.write(f"**Question:** {result['question']}")
                st.write(f"**Answer:** {result['answer']}")
                
                # Test details
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Mode:** {result['answer_type']} ({'With documents' if result['answer_type'] == 'RAG' else 'General knowledge'})")
                    st.write(f"**Model:** {result['model_display_name']}")
                with col2:
                    st.write(f"**Temperature:** {result['parameters']['temperature']}")
                    st.write(f"**Max Tokens:** {result['parameters']['max_tokens']}")
                
                # Context information for RAG tests
                if result['answer_type'] == 'RAG' and result['context_used']:
                    st.write(f"**Context Chunks Used:** {len(result['context_used'])}")
                    
                    # Use a simple checkbox instead of toggle buttons to avoid session state conflicts
                    show_context = st.checkbox(f"👁️ View Context", key=f"show_context_{i}")
                    
                    if show_context:
                        st.write("**Context chunks used for this answer:**")
                        for j, chunk in enumerate(result['context_used'], 1):
                            st.write(f"**Chunk {j}:**")
                            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            if j < len(result['context_used']):
                                st.divider()
                
                # Export individual result
                if st.button(f"Export Test {total_tests - i}", key=f"export_{i}"):
                    result_json = json.dumps(result, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=result_json,
                        file_name=f"test_result_{result['test_name']}.json",
                        mime="application/json",
                        key=f"download_{i}"
                    )
    
    else:
        st.info("📊 **No test results yet**\n\nAsk questions in the 'Ask Questions' tab to see results here.")
        
        # Show testing suggestions
        st.subheader("Testing Ideas")
        st.write("""
        **Try these testing scenarios:**
        
        1. **Baseline Comparison**: Ask a question without documents, then upload relevant documents and ask the same question
        2. **Parameter Testing**: Try the same question with different temperature settings
        3. **Chunk Size Impact**: Upload documents, change chunk size in Parameters, reprocess, and compare answers
        4. **Model Comparison**: Ask the same question using different DeepSeek models
        5. **Complex Queries**: Test with factual questions, analytical questions, and synthesis questions
        """)

# =====================
# TAB 6: VECTOR EXPLORER
# =====================
with tab6:
    st.header("🧮 Vector Database Explorer")
    
    if st.session_state.files_processed and st.session_state.collection is not None:
        st.success("✅ **Vector database is active** - Explore the relationship between chunks, vectors, and IDs")
        
        # Database overview
        st.subheader("📊 Database Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tooltip_metric("Total Vectors", len(st.session_state.all_chunks), 
                          "Each chunk has a corresponding vector representation")
        with col2:
            # Get vector dimension from the collection
            try:
                # Try to get a sample embedding to determine dimensions
                sample_result = st.session_state.collection.get(
                    limit=1,
                    include=["embeddings"]
                )
                if sample_result.get("embeddings") and len(sample_result["embeddings"]) > 0:
                    vector_dim = len(sample_result["embeddings"][0])
                else:
                    vector_dim = 1536  # DeepSeek embedding model default (updated from OpenAI)
            except:
                vector_dim = 1536
            
            tooltip_metric("Vector Dimensions", vector_dim, 
                          "Each vector has this many numerical dimensions")
        with col3:
            total_numbers = len(st.session_state.all_chunks) * vector_dim
            tooltip_metric("Total Numbers", f"{total_numbers:,}", 
                          "Total numerical values stored in the vector database")
        
        st.divider()
        
        # Chunk ID Explorer
        st.subheader("🔍 Chunk ID Explorer")
        st.write("Select a chunk ID to see its content and vector information")
        
        # Get actual chunk IDs from the database
        try:
            all_data = st.session_state.collection.get()  # IDs are returned by default
            actual_chunk_ids = all_data.get("ids", [])
        except Exception as e:
            st.error(f"Error getting chunk IDs: {e}")
            actual_chunk_ids = []
        
        if actual_chunk_ids:
            col1, col2 = st.columns([2, 1])
            with col1:
                chunk_id = st.selectbox(
                    "Select Chunk ID", 
                    options=[""] + actual_chunk_ids,
                    help="Choose a chunk ID to explore its content and vector data"
                )
            with col2:
                st.info(f"Total chunks: {len(actual_chunk_ids)}")
        else:
            st.error("No chunk IDs found in database")
            chunk_id = None
        
        if chunk_id:
            try:
                # Query the specific chunk by ID - explicitly request embeddings
                result = st.session_state.collection.get(
                    ids=[chunk_id],
                    include=["documents", "embeddings"]  # Explicitly request embeddings
                )
                
                if result["documents"] and len(result["documents"]) > 0:
                    chunk_content = result["documents"][0]
                    # Fix: Handle NumPy array boolean evaluation issue
                    embeddings_list = result.get("embeddings")
                    chunk_embedding = embeddings_list[0] if embeddings_list is not None and len(embeddings_list) > 0 else None
                    
                    st.success(f"✅ Found chunk: **{chunk_id}**")
                    
                    # Chunk details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📄 Chunk Content")
                        st.text_area("Content", chunk_content, height=200, disabled=True, label_visibility="collapsed")
                        
                        # Content statistics
                        st.write("**Content Statistics:**")
                        st.write(f"- Characters: {len(chunk_content):,}")
                        st.write(f"- Words: {len(chunk_content.split()):,}")
                        st.write(f"- Lines: {len(chunk_content.splitlines()):,}")
                    
                    with col2:
                        st.subheader("🧮 Vector Information")
                        if chunk_embedding is not None:
                            try:
                                # Convert to list if it's a NumPy array to avoid boolean ambiguity
                                if hasattr(chunk_embedding, 'tolist'):
                                    vector_list = chunk_embedding.tolist()
                                else:
                                    vector_list = list(chunk_embedding)
                                
                                st.write(f"**Vector Dimensions:** {len(vector_list)}")
                                st.write(f"**Vector Range:** {min(vector_list):.6f} to {max(vector_list):.6f}")
                                st.write(f"**Vector Magnitude:** {sum(x*x for x in vector_list)**0.5:.6f}")
                                
                                # Show first few vector values
                                st.write("**First 10 Vector Values:**")
                                vector_preview = ", ".join([f"{x:.6f}" for x in vector_list[:10]])
                                st.code(f"[{vector_preview}, ...]")
                                
                                # Option to show full vector
                                if st.checkbox("Show Full Vector", help="Display all vector dimensions (warning: very long!)"):
                                    st.write("**Complete Vector:**")
                                    st.json(vector_list)
                            except Exception as e:
                                st.error(f"Error processing vector: {e}")
                                st.write(f"Vector type: {type(chunk_embedding)}")
                                st.write(f"Vector shape: {getattr(chunk_embedding, 'shape', 'No shape attribute')}")
                        else:
                            st.warning("Vector data not available for this chunk")
                    
                    st.divider()
                    
                    # Find similar chunks
                    st.subheader("🔗 Similar Chunks")
                    st.write("Find other chunks similar to this one:")
                    
                    similarity_count = st.slider("Number of similar chunks to find", 1, 10, 5, 
                                               help="How many similar chunks to retrieve")
                    
                    if st.button("🔍 Find Similar Chunks"):
                        try:
                            # Use our DeepSeek embedding function instead of query_texts
                            query_embedding = embed_query(chunk_content)
                            similar_results = st.session_state.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=similarity_count + 1  # +1 because it will include itself
                            )
                            
                            similar_chunks = similar_results.get("documents", [[]])[0]
                            similar_ids = similar_results.get("ids", [[]])[0]
                            similar_scores = similar_results.get("distances", [[]])[0]
                            
                            st.write(f"**Found {len(similar_chunks)} similar chunks:**")
                            
                            for i, (sim_id, sim_chunk, sim_score) in enumerate(zip(similar_ids, similar_chunks, similar_scores)):
                                if sim_id == chunk_id:
                                    continue  # Skip the original chunk
                                
                                # Determine similarity quality
                                if sim_score < 0.3:
                                    quality = "🟢 Very Similar"
                                elif sim_score < 0.6:
                                    quality = "🟡 Somewhat Similar"
                                else:
                                    quality = "🔴 Distantly Similar"
                                
                                with st.expander(f"{sim_id} - {quality} (Score: {sim_score:.4f})"):
                                    st.text(sim_chunk[:300] + "..." if len(sim_chunk) > 300 else sim_chunk)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Similarity Score", f"{sim_score:.4f}")
                                    with col2:
                                        st.metric("Length", f"{len(sim_chunk)} chars")
                        
                        except Exception as e:
                            st.error(f"Error finding similar chunks: {e}")
                
                else:
                    st.error(f"❌ Chunk ID '{chunk_id}' not found")
                    st.write("**Available chunk IDs:**")
                    if actual_chunk_ids:
                        st.write(", ".join(actual_chunk_ids[:10]) + ("..." if len(actual_chunk_ids) > 10 else ""))
                    else:
                        st.write("No chunk IDs available")
            
            except Exception as e:
                st.error(f"Error retrieving chunk: {e}")
        
        st.divider()
        
        # Bulk chunk browser
        st.subheader("📚 Bulk Chunk Browser")
        st.write("Browse all chunks with their IDs and basic info")
        
        # Initialize session state for chunk browser
        if 'show_all_chunks' not in st.session_state:
            st.session_state.show_all_chunks = False
        
        # Toggle buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Show All Chunk IDs and Previews"):
                st.session_state.show_all_chunks = True
        with col2:
            if st.button("❌ Hide Chunk Browser") and st.session_state.show_all_chunks:
                st.session_state.show_all_chunks = False
        
        if st.session_state.show_all_chunks:
            try:
                # Get all data including documents and metadata (IDs returned by default)
                all_data = st.session_state.collection.get(include=["documents", "metadatas"])
                chunk_ids = all_data.get("ids", [])
                chunk_docs = all_data.get("documents", [])
                chunk_metas = all_data.get("metadatas", [])
                
                st.write(f"**All {len(chunk_ids)} chunks in the database:**")
                
                for i, (chunk_id, chunk_content) in enumerate(zip(chunk_ids, chunk_docs)):
                    preview = chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content
                    
                    # Get metadata if available
                    metadata = chunk_metas[i] if i < len(chunk_metas) else {}
                    source_doc = metadata.get('source', 'Unknown')
                    
                    col1, col2, col3, col4 = st.columns([1.5, 2, 1, 1])
                    with col1:
                        st.code(chunk_id, language=None)
                    with col2:
                        st.text(preview)
                    with col3:
                        st.caption(f"{len(chunk_content)} chars")
                    with col4:
                        st.caption(f"From: {source_doc}")
                    
                    if i < len(chunk_ids) - 1:
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error retrieving chunk data: {e}")
                # Fallback to session state data
                st.write(f"**Fallback: {len(st.session_state.all_chunks)} chunks from session state:**")
                for i, chunk in enumerate(st.session_state.all_chunks):
                    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.code(f"chunk_{i}")
                    with col2:
                        st.text(preview)
                    with col3:
                        st.caption(f"{len(chunk)} chars")
                    
                    if i < len(st.session_state.all_chunks) - 1:
                        st.divider()
        
        # Educational section
        st.divider()
        st.subheader("🎓 How Vector Storage Works")
        st.write("""
        **The Vector Database Process:**
        
        1. **Text → Vector**: Each chunk is converted to a {}-dimensional vector using OpenAI's embedding model (text-embedding-3-small)
        2. **Storage**: Vectors are stored with unique IDs (chunk_0, chunk_1, etc.) alongside the original text
        3. **Retrieval**: When you ask a question, your query becomes a vector and finds similar vectors
        4. **Similarity**: Lower distance scores mean higher similarity (0.0 = identical, 1.0+ = very different)
        5. **Context**: The most similar chunks are retrieved and sent to the AI for answering
        
        **Key Insights:**
        - Each chunk exists as both **text** (human readable) and **vector** (mathematical representation)
        - Vectors capture semantic meaning, not just word matching
        - Similar concepts have similar vectors, even with different words
        - The vector space has {} dimensions - each dimension captures some aspect of meaning
        """.format(vector_dim, vector_dim))
    
    else:
        st.info("🔍 **No vector database available**\n\nUpload documents in the 'Documents' tab to create a vector database, then return here to explore how chunks and vectors are stored.")
        
        # Show educational content about vectors
        st.subheader("🎓 What You'll See Here")
        st.write("""
        Once you upload documents, this tab will let you:
        
        **🔍 Explore Individual Chunks:**
        - Enter any chunk ID (like "chunk_0") to see its content
        - View the actual vector numbers that represent that chunk
        - See statistics about the chunk and its vector
        
        **🔗 Find Relationships:**
        - Discover which chunks are similar to each other
        - See similarity scores between different pieces of content
        - Understand how the AI finds relevant information
        
        **📊 Database Insights:**
        - Browse all chunk IDs and their content previews
        - Understand the scale of your vector database
        - Learn how text becomes mathematical representations
        
        This helps you understand the "magic" behind RAG - how text becomes numbers that can be mathematically compared for similarity!
        """) 
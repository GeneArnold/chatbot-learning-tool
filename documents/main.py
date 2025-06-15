# Step1: Return list of files in sample_docs directory
# Step2: Chunk these files
# Step3: Embed these chunks
# Step4: Store these chunks in a vector database
# Step5: Query the vector database for relevant chunks
# Step6: Use the LLM to answer the question
# Step7: Return the answer

import os
os.environ["LLAMA_LOG_LEVEL"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
import chromadb
import openai
from rich.console import Console
from rich.table import Table
from llama_cpp import Llama
from rich.panel import Panel
from rich.spinner import Spinner
from dotenv import load_dotenv
load_dotenv()

sample_docs_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
console = Console()

# =====================
# CONFIGURATION KNOBS
# =====================
# --- ChromaDB (Retriever) ---
CHROMA_N_RESULTS = 3  # Number of chunks to retrieve per query
CHUNK_SIZE = 500      # Number of characters per chunk
CHUNK_OVERLAP = 100   # Overlap between chunks
# (distance_metric and min_score are not directly exposed in ChromaDB's Python API, but can be added if needed)

# --- LLM (Generator) ---
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 1.0  # Only used by OpenAI, not llama.cpp
LLM_STOP = ["\n"]  # Stop sequence for Llama and OpenAI
LLM_SYSTEM_PROMPT = "You are a helpful assistant. Answer only using the provided context. If the answer is not present, say you don't know."
LLM_MODEL = "gpt-3.5-turbo"  # OpenAI model name

def get_files():
    """Return list of .txt and .md files in sample_docs directory."""
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
        print(f"Error reading {filepath}: {e}")
        return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def store_embeddings(chunks, embeddings):
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

def embed_chunks(chunks):
    """
    Given a list of text chunks, return a list of embedding vectors.
    """
    return embedder.encode(chunks, show_progress_bar=True)

def ask_openai(context_chunks, user_query, model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P, system_prompt=LLM_SYSTEM_PROMPT, stop=LLM_STOP):
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )
    return response.choices[0].message.content.strip()

def ask_llama(context_chunks, user_query, llm, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE, stop=LLM_STOP):
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    output = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
    return output["choices"][0]["text"].strip()

# Hardcoded model options for easier Streamlit UI integration in the future
model_options = [
    ("llama-2-7b.Q4_0.gguf", "Llama 2 7B"),
    ("openhermes-2.5-mistral-7b.Q4_K_M.gguf", "OpenHermes 2.5 Mistral 7B"),
    ("Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf", "Nous Hermes 2 Mistral 7B DPO"),
    ("openai", "OpenAI (gpt-3.5-turbo)")
]
model_display_names = {
    "llama-2-7b.Q4_0.gguf": "Llama 2 7B",
    "openhermes-2.5-mistral-7b.Q4_K_M.gguf": "OpenHermes 2.5 Mistral 7B",
    "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf": "Nous Hermes 2 Mistral 7B DPO",
    "openai": "OpenAI (gpt-3.5-turbo)"
}

# Step 1: Get files
files = get_files()
file_table = Table(title="Files Found in sample_docs")
file_table.add_column("File Name", style="green")
for fname in files:
    file_table.add_row(fname)
console.print(file_table)

# Step 2: Read and chunk each file
all_chunks = []
chunk_table = Table(title="Chunking Summary")
chunk_table.add_column("File", style="cyan")
chunk_table.add_column("Chunks", style="magenta")
for fname in files:
    path = os.path.join(sample_docs_dir, fname)
    content = read_text_file(path)
    if content:
        chunks = chunk_text(content)
        chunk_table.add_row(fname, str(len(chunks)))
        all_chunks.extend(chunks)
    else:
        console.print(f"[red]Skipping {fname} due to read error.[/red]")
console.print(chunk_table)

if all_chunks:
    console.print(f"[bold green]Total chunks from all files:[/bold green] {len(all_chunks)}")
    console.print(f"[bold]First chunk preview:[/bold] {all_chunks[0]}")
else:
    console.print("[red]No chunks to embed.[/red]")

# Step3: Embed these chunks
embedder = SentenceTransformer("all-MiniLM-L6-v2")
if all_chunks:
    embeddings = embed_chunks(all_chunks)
    console.print(f"[bold green]Number of embeddings:[/bold green] {len(embeddings)}")
    console.print(f"[bold green]Embedding vector length:[/bold green] {len(embeddings[0])}")
else:
    console.print("[red]No chunks to embed.[/red]")

# Step4: Store these chunks in a vector database
client = chromadb.PersistentClient(path=".chroma_database")
collection = client.get_or_create_collection("simple_chunks")

# ---
# For now, clear the collection at the start of each run to avoid duplicate chunks.
# This is a simple way to ensure we don't accumulate duplicate data while developing.
# In a production system, a better solution is to use unique, deterministic IDs for each chunk
# and only add new/changed data. We'll add that later for incremental/efficient updates.
client.delete_collection("simple_chunks")  # Delete the entire collection to remove all documents
collection = client.get_or_create_collection("simple_chunks")  # Recreate the collection
# ---

if all_chunks:
    store_embeddings(all_chunks, embeddings)
    console.print(f"[bold green]Stored {len(all_chunks)} chunks in ChromaDB.[/bold green]")

# Step 5: Query the vector database for relevant chunks
user_query = console.input("[bold blue]Ask a question:[/bold blue] ")
query_embedding = embedder.encode([user_query])
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=CHROMA_N_RESULTS
)
relevant_chunks = results.get("documents", [[]])[0]
scores = results.get("distances", [[]])[0]

result_table = Table(title="Relevant Chunks from ChromaDB")
result_table.add_column("Rank", style="cyan", justify="right")
result_table.add_column("Score (lower=better)", style="magenta")
result_table.add_column("Chunk Preview", style="white")
for i, (chunk, score) in enumerate(zip(relevant_chunks, scores), 1):
    preview = chunk[:100].replace('\n', ' ') + ("..." if len(chunk) > 100 else "")
    result_table.add_row(str(i), f"{score:.4f}", preview)
console.print(result_table)


print("\nRelevant chunks retrieved from ChromaDB:")
for i, chunk in enumerate(relevant_chunks, 1):
    print(f"{i}. {chunk[:200]}")  # Print a preview of each chunk


# Step6: Use the LLM to answer the question
console.print("\n[bold blue]Select which LLM to use to answer your question:[/bold blue]")
for idx, (_, desc) in enumerate(model_options, 1):
    default_str = " [default]" if idx == 1 else ""
    console.print(f"  {idx}. {desc}{default_str}")

while True:
    llm_choice_input = console.input("[bold blue]Enter the number of the model to use [1]: [/bold blue]").strip()
    if llm_choice_input == "":
        llm_choice = model_options[0][0]  # Default to first model
        break
    if llm_choice_input.isdigit():
        idx = int(llm_choice_input)
        if 1 <= idx <= len(model_options):
            llm_choice = model_options[idx-1][0]
            break
    console.print(f"[red]Invalid input. Please enter a number between 1 and {len(model_options)}.[/red]")

model_name = model_display_names.get(llm_choice, llm_choice)
if llm_choice == "openai":
    with console.status(f"[bold green]Thinking... (running {model_name} model)[/bold green]", spinner="dots"):
        try:
            llm_answer = ask_openai(relevant_chunks, user_query)
        except Exception as e:
            console.print(f"[red]OpenAI error: {e}[/red]")
            llm_answer = "Error: Could not get answer from OpenAI."
else:
    llama_model_path = os.path.join(os.path.dirname(__file__), "models", llm_choice)
    llm = Llama(model_path=llama_model_path, n_ctx=2048)
    with console.status(f"[bold green]Thinking... (running {model_name} model)[/bold green]", spinner="dots"):
        llm_answer = ask_llama(relevant_chunks, user_query, llm)


# Step7: Return the answer
# console.print(f"[bold yellow]LLM Answer:[/bold yellow] {llm_answer}")
# After getting llm_answer
console.print(Panel(llm_answer, title="LLM Answer", style="bold yellow"))
#console.print(f"[bold green]Final answer:[/bold green] {llm_answer}")
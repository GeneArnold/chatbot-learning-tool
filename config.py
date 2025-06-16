import os
from pathlib import Path

class Config:
    """Centralized configuration for the RAG Chatbot application"""
    
    # Base directories - can be overridden by environment variables
    BASE_DIR = Path(os.getenv('APP_BASE_DIR', '.'))
    DATA_DIR = Path(os.getenv('APP_DATA_DIR', BASE_DIR / 'data'))
    
    # Database configuration
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', str(DATA_DIR / 'chroma_database'))
    
    # File storage
    UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', DATA_DIR / 'uploads'))
    SAMPLE_DOCS_DIR = Path(os.getenv('SAMPLE_DOCS_DIR', DATA_DIR / 'sample_docs'))
    
    # Static assets
    IMAGES_DIR = Path(os.getenv('IMAGES_DIR', BASE_DIR / 'images'))
    LOGO_PATH = IMAGES_DIR / 'DGT.webp'
    
    # API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Embedding and model defaults
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    DEFAULT_LLM_MODEL = os.getenv('DEFAULT_LLM_MODEL', 'gpt-3.5-turbo')
    
    # RAG Parameters (with environment variable overrides)
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
    N_RESULTS = int(os.getenv('N_RESULTS', '3'))
    
    # LLM Parameters
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '256'))
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.2'))
    LLM_TOP_P = float(os.getenv('LLM_TOP_P', '1.0'))
    
    # Streamlit configuration
    STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', '8501'))
    STREAMLIT_HOST = os.getenv('STREAMLIT_HOST', '0.0.0.0')
    
    # Environment detection
    IS_DOCKER = os.getenv('RUNNING_IN_DOCKER', 'false').lower() == 'true'
    IS_PRODUCTION = os.getenv('ENVIRONMENT', 'development').lower() == 'production'
    
    # System prompt
    SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 
        "You are a helpful assistant that can ONLY answer questions using the provided context. "
        "You must NOT use any external knowledge or training data. If no context is provided or "
        "the answer is not in the provided context, you MUST respond with exactly: "
        "'I don't have enough information to answer this question based on the provided context.'"
    )
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.UPLOAD_DIR,
            cls.SAMPLE_DOCS_DIR,
            cls.IMAGES_DIR,
            Path(cls.CHROMA_DB_PATH).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def validate_config(cls):
        """Validate critical configuration settings"""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY environment variable not set")
            
        if not cls.LOGO_PATH.exists() and not cls.IS_DOCKER:
            errors.append(f"Logo file not found: {cls.LOGO_PATH}")
            
        return errors
    
    @classmethod
    def get_docker_info(cls):
        """Return Docker-specific configuration info"""
        return {
            "is_docker": cls.IS_DOCKER,
            "is_production": cls.IS_PRODUCTION,
            "data_dir": str(cls.DATA_DIR),
            "chroma_path": cls.CHROMA_DB_PATH,
            "upload_dir": str(cls.UPLOAD_DIR)
        } 
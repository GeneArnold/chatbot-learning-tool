import os
from pathlib import Path
import tiktoken

class Config:
    """Centralized configuration for the RAG Chatbot application"""
    
    # Base directories - can be overridden by environment variables
    BASE_DIR = Path(os.getenv('APP_BASE_DIR', '.'))
    DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
    
    # Database configuration
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', str(DATA_DIR / 'chroma_database'))
    
    # File storage
    SAMPLE_DOCS_DIR = Path(os.getenv('SAMPLE_DOCS_DIR', DATA_DIR / 'sample_docs'))
    
    # Static assets
    IMAGES_DIR = Path(os.getenv('IMAGES_DIR', BASE_DIR / 'images'))
    LOGO_PATH = IMAGES_DIR / 'DGT.webp'
    
    # =============================================================================
    # API Configuration - HYBRID APPROACH: DEEPSEEK CHAT + OPENAI EMBEDDINGS
    # =============================================================================
    
    # DeepSeek API Configuration (PRIMARY for LLM)
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    
    # OpenAI API Configuration (FOR EMBEDDINGS - DeepSeek doesn't offer embedding API)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Still needed for embeddings
    
    # =============================================================================
    # COST CONFIGURATION - Per-token pricing for transparency
    # =============================================================================
    
    # DeepSeek pricing (per 1M tokens) - Updated rates as of 2024
    DEEPSEEK_PRICING = {
        'deepseek-chat': {
            'input_per_1m': 0.14,   # $0.14 per 1M input tokens
            'output_per_1m': 0.28   # $0.28 per 1M output tokens
        },
        'deepseek-coder': {
            'input_per_1m': 0.14,   # $0.14 per 1M input tokens
            'output_per_1m': 0.28   # $0.28 per 1M output tokens
        },
        'deepseek-reasoner': {
            'input_per_1m': 0.14,   # $0.14 per 1M input tokens
            'output_per_1m': 0.28   # $0.28 per 1M output tokens
        }
    }
    
    # OpenAI pricing (per 1M tokens) - For embeddings only
    OPENAI_PRICING = {
        'text-embedding-3-small': {
            'input_per_1m': 0.02,   # $0.02 per 1M tokens
            'output_per_1m': 0.0    # Embeddings have no output cost
        },
        'text-embedding-3-large': {
            'input_per_1m': 0.13,   # $0.13 per 1M tokens
            'output_per_1m': 0.0    # Embeddings have no output cost
        },
        'text-embedding-ada-002': {
            'input_per_1m': 0.10,   # $0.10 per 1M tokens
            'output_per_1m': 0.0    # Embeddings have no output cost
        }
    }
    
    @classmethod
    def get_cost_per_token(cls, model_name, token_type='input'):
        """
        Get cost per individual token for a specific model
        
        Args:
            model_name: Name of the model
            token_type: 'input' or 'output'
        
        Returns:
            float: Cost per single token
        """
        # Check DeepSeek models first
        if model_name in cls.DEEPSEEK_PRICING:
            cost_per_1m = cls.DEEPSEEK_PRICING[model_name][f'{token_type}_per_1m']
            return cost_per_1m / 1_000_000
        
        # Check OpenAI models
        elif model_name in cls.OPENAI_PRICING:
            cost_per_1m = cls.OPENAI_PRICING[model_name][f'{token_type}_per_1m']
            return cost_per_1m / 1_000_000
        
        # Default fallback
        else:
            return 0.0
    
    @classmethod
    def calculate_cost(cls, model_name, input_tokens=0, output_tokens=0):
        """
        Calculate total cost for a model usage
        
        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            dict: {
                'input_cost': float,
                'output_cost': float, 
                'total_cost': float,
                'input_cost_per_token': float,
                'output_cost_per_token': float
            }
        """
        input_cost_per_token = cls.get_cost_per_token(model_name, 'input')
        output_cost_per_token = cls.get_cost_per_token(model_name, 'output')
        
        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
            'input_cost_per_token': input_cost_per_token,
            'output_cost_per_token': output_cost_per_token
        }
    
    @classmethod
    def count_tokens(cls, text, model_name=None):
        """Count tokens in text for a given model"""
        if model_name is None:
            model_name = cls.DEFAULT_LLM_MODEL
            
        # Use appropriate encoding based on model
        if 'deepseek' in model_name.lower():
            # DeepSeek uses similar tokenization to GPT models
            encoding = tiktoken.get_encoding("cl100k_base")
        elif 'gpt' in model_name.lower() or 'embedding' in model_name.lower():
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Fallback to GPT-4 encoding
            encoding = tiktoken.get_encoding("cl100k_base")
            
        return len(encoding.encode(text))
    
    @classmethod
    def tokenize_text(cls, text, model_name=None):
        """Get actual token breakdown for educational display
        
        Returns:
            dict: {
                'tokens': list of token strings,
                'token_ids': list of token IDs,
                'count': int
            }
        """
        if model_name is None:
            model_name = cls.DEFAULT_LLM_MODEL
            
        # Use appropriate encoding based on model
        if 'deepseek' in model_name.lower():
            encoding = tiktoken.get_encoding("cl100k_base")
        elif 'gpt' in model_name.lower() or 'embedding' in model_name.lower():
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
            
        # Get token IDs
        token_ids = encoding.encode(text)
        
        # Decode each token individually to show breakdown
        tokens = []
        for token_id in token_ids:
            token_str = encoding.decode([token_id])
            tokens.append(token_str)
            
        return {
            'tokens': tokens,
            'token_ids': token_ids,
            'count': len(token_ids)
        }
    
    @classmethod
    def count_tokens_list(cls, text_list, model_name=None):
        """
        Count tokens for a list of texts
        
        Args:
            text_list: List of texts to count tokens for
            model_name: Model name for encoding
        
        Returns:
            int: Total number of tokens across all texts
        """
        total_tokens = 0
        for text in text_list:
            total_tokens += cls.count_tokens(text, model_name)
        return total_tokens
    
    # =============================================================================
    # Model Configuration - HYBRID: DEEPSEEK LLM + OPENAI EMBEDDINGS
    # =============================================================================
    
    # Embedding model (OpenAI - DeepSeek doesn't provide embedding API)
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')  # Must use OpenAI
    
    # LLM model (DeepSeek primary)
    DEFAULT_LLM_MODEL = os.getenv('DEFAULT_LLM_MODEL', 'deepseek-chat')  # DeepSeek chat model
    
    # Model-specific configuration for dynamic parameter adjustment
    MODEL_CONFIGS = {
        # DeepSeek models (PRIMARY for LLM)
        'deepseek-chat': {
            'max_tokens': 8192,
            'default_tokens': 512,  # Higher default due to better capability
            'provider': 'deepseek'
        },
        'deepseek-coder': {
            'max_tokens': 8192,
            'default_tokens': 512,
            'provider': 'deepseek'
        },
        'deepseek-reasoner': {
            'max_tokens': 8192,
            'default_tokens': 512,
            'provider': 'deepseek'
        },
        # OpenAI models (PRESERVED FOR FALLBACK)
        # 'gpt-3.5-turbo': {
        #     'max_tokens': 4096,
        #     'default_tokens': 256,
        #     'provider': 'openai'
        # },
        # 'gpt-4': {
        #     'max_tokens': 8192,
        #     'default_tokens': 256,
        #     'provider': 'openai'
        # },
        # 'gpt-4-turbo-preview': {
        #     'max_tokens': 4096,
        #     'default_tokens': 256,
        #     'provider': 'openai'
        # }
    }
    
    # RAG Parameters (with environment variable overrides)
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
    N_RESULTS = int(os.getenv('N_RESULTS', '3'))
    
    # LLM Parameters (adjusted for DeepSeek capabilities)
    LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '512'))  # Increased from 256 for DeepSeek
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
        
        # Check for both API keys since we use hybrid approach
        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY environment variable not set (required for LLM)")
            
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY environment variable not set (required for embeddings)")
            
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
            "chroma_path": cls.CHROMA_DB_PATH
        } 
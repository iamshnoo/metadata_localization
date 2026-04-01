"""
Configuration file for the MetaCulture Pipeline
"""

import os

class Config:
    # Base paths
    BASE = "/scratch/amukher6/"
    
    # Data paths
    DATA_ROOT = BASE + "metacul/data/now"
    MALLET_OUTPUT_BASE = BASE + "metacul/data/lmw_output/now/"
    TRAINING_DATA_BASE = BASE + "metacul/training_data"
    MODEL_CACHE = BASE + "models/transformers/"
    
    # Llama Model Configuration
    LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    TORCH_DTYPE = "bfloat16"
    DEVICE_MAP = "auto"
    LOW_CPU_MEM_USAGE = True
    
    # Partition configuration
    PIVOT_YEARS = [2012, 2015, 2018, 2021]
    
    # Country to continent mapping
    COUNTRY_TO_CONTINENT = {
        "bd": "Asia", "hk": "Asia", "in": "Asia", "lk": "Asia", 
        "my": "Asia", "ph": "Asia", "pk": "Asia", "sg": "Asia",
        "ca": "America", "jm": "America", "us": "America",
        "gb": "Europe", "ie": "Europe",
        "gh": "Africa", "ke": "Africa", "ng": "Africa", "za": "Africa", "tz": "Africa",
    }
    
    # Country code mapping
    COUNTRY_CODE_MAP = {
        "bd": "Bangladesh", "ca": "Canada", "gb": "United Kingdom",
        "gh": "Ghana", "hk": "Hong Kong", "ie": "Ireland",
        "in": "India", "jm": "Jamaica", "ke": "Kenya",
        "lk": "Sri Lanka", "my": "Malaysia", "ng": "Nigeria",
        "ph": "Philippines", "pk": "Pakistan", "tz": "Tanzania",
        "sg": "Singapore", "us": "United States", "za": "South Africa", 
    }
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

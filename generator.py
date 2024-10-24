import requests
import json
import os
import hashlib
import time
from collections import OrderedDict
from tqdm import tqdm
from datasets import Dataset
import logging
from dotenv import load_dotenv
import sys
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

class SFTDatasetGenerator:
    def __init__(self):
        self._load_environment()
        self.url = "https://integrate.api.nvidia.com/v1/text/completions"
        self.headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json"
        }

    def _load_environment(self):
        """Load and validate environment variables."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Get required environment variables
        self.nvidia_api_key = os.getenv('NVIDIA_API_KEY')
        self.hf_token = os.getenv('HF_TOKEN')
        
        # Validate environment variables
        if not self.nvidia_api_key:
            raise ConfigurationError("NVIDIA_API_KEY not found in environment variables")
        if not self.hf_token:
            raise ConfigurationError("HF_TOKEN not found in environment variables")

    def generate_completion(self, prompt: str, max_tokens: int = 50, 
                          temperature: float = 0.7, max_retries: int = 3, 
                          retry_delay: int = 5) -> Optional[str]:
        """Generate completion using NVIDIA API with retry mechanism."""
        data = {
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "n": 1,
            "stop": None
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, headers=self.headers, 
                                      json=data, timeout=10)
                response.raise_for_status()
                return response.json()['choices'][0]['text'].strip()
            except requests.exceptions.RequestException as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.warning("Max retries reached. Skipping this prompt.")
                    return None

    @staticmethod
    def hash_string(input_string: str, algorithm: str = 'sha256') -> str:
        """Generate hash for input string."""
        hash_algorithms = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha256': hashlib.sha256
        }
        
        if algorithm not in hash_algorithms:
            raise ValueError("Unsupported hashing algorithm")
            
        hash_object = hash_algorithms[algorithm]()
        hash_object.update(input_string.encode('utf-8'))
        return hash_object.hexdigest()

    def process_prompts(self, prompts: List[str], samples_per_prompt: int, 
                       output_dir: str) -> Dict[str, Dataset]:
        """Process multiple prompts and generate datasets."""
        datasets = {}
        os.makedirs(output_dir, exist_ok=True)

        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Processing prompt {i}/{len(prompts)}: {prompt[:50]}...")
            
            # Generate completions
            completions = []
            for _ in tqdm(range(samples_per_prompt), 
                         desc=f"Generating completions for prompt {i}"):
                completion = self.generate_completion(prompt)
                if completion:
                    completions.append(completion)

            # Save raw completions
            output_file = os.path.join(output_dir, f"completions_prompt_{i}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                for completion in completions:
                    f.write(f"{completion}\n")

            # Create and deduplicate dataset
            empty_strings = [''] * len(completions)
            index = [self.hash_string(c) for c in completions]

            unique_data = OrderedDict()
            for idx, completion, empty in zip(index, completions, empty_strings):
                if idx not in unique_data:
                    unique_data[idx] = (completion, empty)

            deduplicated_dataset = Dataset.from_dict({
                "index": list(unique_data.keys()),
                "instruction": [item[0] for item in unique_data.values()],
                "output": [item[1] for item in unique_data.values()]
            })

            datasets[f"prompt_{i}"] = deduplicated_dataset
            
            logger.info(f"Prompt {i} stats:")
            logger.info(f"Original size: {len(completions)}")
            logger.info(f"Deduplicated size: {len(deduplicated_dataset)}")
            logger.info(f"Duplicates removed: {len(completions) - len(deduplicated_dataset)}")

        return datasets

    def upload_datasets(self, datasets: Dict[str, Dataset], 
                       repo_id: str, split: str = "train"):
        """Upload datasets to Hugging Face Hub."""
        for name, dataset in datasets.items():
            repo_name = f"{repo_id}_{name}"
            logger.info(f"Uploading dataset {name} to {repo_name}")
            dataset.push_to_hub(repo_name, split=split, token=self.hf_token)

def main():
    try:
        # Import configuration
        try:
            import config
            logger.info("Successfully loaded configuration")
        except ImportError as e:
            logger.error(f"Failed to import configuration: {e}")
            logger.error("Make sure config.py exists in the current directory")
            sys.exit(1)

        # Initialize generator
        generator = SFTDatasetGenerator()
        
        # Create output directory
        os.makedirs(config.OUTPUT_DIRECTORY, exist_ok=True)
        
        # Process prompts and generate datasets
        datasets = generator.process_prompts(
            prompts=config.PROMPTS,
            samples_per_prompt=config.SAMPLES_PER_PROMPT,
            output_dir=config.OUTPUT_DIRECTORY
        )
        
        # Upload datasets
        generator.upload_datasets(datasets, config.REPOSITORY)
        logger.info("All datasets processed and uploaded successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
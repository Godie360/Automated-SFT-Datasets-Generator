# Configuration settings for SFT Dataset Generation

# List of prompts to generate datasets from
PROMPTS = [
    "Tafadhali tengeneza swali moja kuhusu afya ya uzazi kwa lugha ya Kiswahili.",
    "Tafadhali tengeneza swali moja kuhusu lishe bora kwa lugha ya Kiswahili.",
    "Tafadhali tengeneza swali moja kuhusu mazoezi ya viungo kwa lugha ya Kiswahili."
]

# Number of samples to generate per prompt
SAMPLES_PER_PROMPT = 5000

# Directory where raw completions will be saved
OUTPUT_DIRECTORY = "completions"

# Base repository name for Hugging Face datasets
REPOSITORY = "sartifyllc/sft_question_Datasets_Nvidia"

# Model parameters (optional)
MODEL_PARAMS = {
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.95
}
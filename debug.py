import os
import logging
from dotenv import load_dotenv
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables and packages are present."""
    logger.info("Checking environment setup...")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check if .env file exists
    env_file = ".env"
    if not os.path.exists(env_file):
        logger.error(f"Missing {env_file} file in current directory")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Check required environment variables
    required_vars = ['NVIDIA_API_KEY', 'HF_TOKEN']
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            logger.info(f"Found {var}: {value[:8]}...")
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("Environment variables loaded successfully")
    return True

def check_config():
    """Check if configuration is valid."""
    logger.info("Checking configuration...")
    
    try:
        import config
        
        # Check required attributes
        required_attrs = ['PROMPTS', 'SAMPLES_PER_PROMPT', 'OUTPUT_DIRECTORY', 'REPOSITORY']
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(config, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            logger.error(f"Missing required attributes in config: {', '.join(missing_attrs)}")
            return False
        
        # Validate prompts
        if not isinstance(config.PROMPTS, list) or not config.PROMPTS:
            logger.error("PROMPTS must be a non-empty list")
            return False
        
        # Validate samples per prompt
        if not isinstance(config.SAMPLES_PER_PROMPT, int) or config.SAMPLES_PER_PROMPT <= 0:
            logger.error("SAMPLES_PER_PROMPT must be a positive integer")
            return False
        
        logger.info("Configuration is valid")
        logger.debug(f"PROMPTS: {config.PROMPTS}")
        logger.debug(f"SAMPLES_PER_PROMPT: {config.SAMPLES_PER_PROMPT}")
        logger.debug(f"OUTPUT_DIRECTORY: {config.OUTPUT_DIRECTORY}")
        logger.debug(f"REPOSITORY: {config.REPOSITORY}")
        return True
        
    except ImportError:
        logger.error("Could not import config.py")
        return False
    except Exception as e:
        logger.error(f"Error checking configuration: {str(e)}")
        return False

def main():
    # Run checks
    checks = [
        ("Environment", check_environment()),
        ("Configuration", check_config())
    ]
    
    # Print results
    logger.info("\nDebug Results:")
    all_passed = True
    for check_name, result in checks:
        status = "✅ Passed" if result else "❌ Failed"
        logger.info(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll checks passed! The script should be ready to run.")
    else:
        logger.error("\nSome checks failed. Please fix the issues above before running the main script.")

if __name__ == "__main__":
    main()
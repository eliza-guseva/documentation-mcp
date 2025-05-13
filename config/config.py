import json
import os
from config.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = 'config.json'

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    ...

class ConfigManager:
    """Manages loading and accessing configuration."""
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = None
        self.load_config()

    def load_config(self):
        logger.info(f"Attempting to load configuration from {self.config_path}")
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("Configuration loaded successfully.")
            self._validate_config()
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {self.config_path}.")
            logger.error("Please create it based on config.json.template and ensure the GitHub token is set.")
            raise ConfigError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {self.config_path}.")
            raise ConfigError(f"Invalid JSON in config file: {self.config_path}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while loading config: {e}")
            raise ConfigError(f"Failed to load config: {e}")

    def _validate_config(self):
        required_keys = [
            "repositories",
            "documentation_urls",
            "log_file"
        ]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            logger.error(f"Missing required keys in config file: {missing_keys}")
            raise ConfigError(f"Missing keys in config: {missing_keys}")

        if not self.config.get("github_token") or self.config["github_token"] == "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN":
            logger.warning("GitHub token is missing or still set to the placeholder value.")
            # right now we are ignoring repositories
            ...

        logger.info("Configuration validated.")

    def get(self, key, default=None):
        return self.config.get(key, default)

    @property
    def github_token(self):
        return self.get("github_token")

    @property
    def local_storage_path(self):
        # Ensure the storage path exists
        path = self.get("local_storage_path")
        os.makedirs(path, exist_ok=True)
        return path
    
    @property
    def documentation_urls(self):
        # Retrieve the value, defaulting to an empty dict if not found or wrong type
        config = self.get("documentation_urls", {})
        all_urls = [url for url_list in config.values() for url in url_list]
        return list(set(all_urls)) # Return unique URLs

    @property
    def update_interval_days(self):
        return self.get("update_interval_days", 3)

    @property
    def log_file(self):
        return self.get("log_file", "documentation.log")


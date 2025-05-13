import time
from config.config import ConfigManager, ConfigError
from data_collection.fetcher import DocumentationFetcher
from config.utils import get_logger


logger = get_logger(__name__) 

def run_initial_fetch(config_path: str = 'config.json', max_depth: int = 5) -> None:
    """Orchestrates the initial data collection process.
    Args:
        config_path (str): Path to the configuration file.
        max_depth (int): Maximum depth for repository fetching.
    """
    try:
        # 1. Load Configuration
        logger.info("Loading configuration...") # Log immediately
        config_manager = ConfigManager(config_path)
        logger.info(f"Configuration loaded. Log file set to: {config_manager.log_file}") 

    except ConfigError as e:
        logger.critical(f"Configuration Error: {e}. Halting operation.")
        print(f"CRITICAL: Configuration Error - {e}. Cannot proceed.") # Also print for visibility before logging might be fully set up
        return # Halt operation
    except Exception as e:
        logger.critical(f"Unexpected error during initialization: {e}", exc_info=True)
        print(f"CRITICAL: Unexpected error during initialization - {e}. Cannot proceed.")
        return # Halt operation

    logger.info("Initialization complete. Starting data fetch.")
    start_time = time.time()

    # 2. Initialize Fetchers (DocumentationFetcher for now and RepositoryFetcher in the future)
    doc_fetcher = DocumentationFetcher(
        storage_path=config_manager.local_storage_path,
        max_depth=max_depth
    )
    # 3. Fetch Documentation 
    doc_urls = config_manager.documentation_urls
    logger.info(f"Fetching {len(doc_urls)} documentation sources")
    doc_fetcher.fetch_all_documentation(doc_urls)
    logger.info("Documentation fetch process finished (placeholder).")

    end_time = time.time()
    logger.info(f"Initial data collection finished in {end_time - start_time:.2f} seconds.")
    
    # 4. Fetch Repositories # TODO


import json
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Optional, Tuple, Dict, Set
from config.utils import get_logger
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import bs4


logger = get_logger(__name__)


# --- Helper Functions ---

def url_to_source_identifier(url: str) -> str:
    """
    Converts a URL into a filesystem-friendly source identifier string.

    Example: "https://python.langchain.com/api_reference/" -> "python.langchain.com_api_reference"
             "https://ai.pydantic.dev/" -> "ai.pydantic.dev_index"
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or "unknown_host"
    # Strip leading/trailing slashes, replace internal slashes, handle root path
    path_slug = parsed_url.path.strip('/').replace('/', '_') or "index"
    return f"{hostname}_{path_slug}"

def source_identifier_to_url_info(identifier: str) -> Tuple[str, str]:
    """
    Converts a source identifier string back into hostname and path components.

    Example: "python.langchain.com_api_reference" -> ("python.langchain.com", "/api/reference")
             "ai.pydantic.dev_index" -> ("ai.pydantic.dev", "/")
    """
    parts = identifier.split('_', 1) 
    hostname = parts[0]
    if len(parts) > 1:
        path_slug = parts[1]
        # Reverse the slug transformation
        reconstructed_path = '/' + path_slug.replace('_', '/') if path_slug != "index" else "/"
    else:
        # Handle cases where there might not be a path part (e.g., just hostname?)
        reconstructed_path = "/"

    return hostname, reconstructed_path

# --- End Helper Functions ---

class DocumentationFetcher:
    """Handles fetching documentation from web URLs using RecursiveUrlLoader."""
    def __init__(self, storage_path: str, max_depth: int = 5):
        self.storage_path = Path(storage_path).resolve()
        self.docs_raw_path = self.storage_path / "documentation_raw"
        self.max_depth = max_depth
        self.docs_raw_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Documentation storage path: {self.docs_raw_path}")

    def _default_extractor(self, html: str) -> str:
        """Basic extractor using BeautifulSoup to get text content."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            # Find the main content area if possible (this might need customization per site)
            # Example heuristic: look for <main>, <article>, or role="main"
            main_content = soup.find("main") or soup.find("article") or soup.find(role="main")
            if main_content:
                return main_content.get_text(separator=" ", strip=True)
            else:
                # Fallback to body if no main content area found
                body = soup.find("body")
                if body:
                    return body.get_text(separator=" ", strip=True)
                else:
                    # If no body, try returning everything (might include scripts/styles)
                    return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.error(f"Error parsing HTML with BeautifulSoup: {e}")
            return "" # Return empty string on parsing error
    
    @staticmethod
    def _process_potential_code_blocks(
        potential_containers: List[bs4.element.Tag], 
        processed_elements: Set[bs4.element.Tag], 
        code_blocks: Dict[str, str], 
        soup: bs4.BeautifulSoup
    ) -> None:
        """
        Identifies and extracts code blocks from a list of potential HTML container elements.

        This method iterates through `potential_containers` (e.g., `<div>` or `<pre>`),
        attempts to find code within them, and extracts the programming language
        and an optional filename. Each found code block is replaced in the `soup`
        with a unique placeholder string. The mapping between these placeholders
        and their corresponding Markdown-formatted code block is stored in the
        `code_blocks` dictionary. The `processed_elements` set is used to prevent
        re-processing of already handled HTML elements, particularly in cases of
        nested tags.

        Args:
            potential_containers: A list of BeautifulSoup Tag objects that are
                                  candidates for containing code blocks.
            processed_elements: A set of BeautifulSoup Tag objects that have already
                                been processed. This is used to avoid duplicate
                                processing of elements, especially in nested structures.
            code_blocks: A dictionary that will be populated with placeholders as keys
                         and Markdown-formatted code strings as values.
                         This argument is modified in place.
            soup: The BeautifulSoup object representing the parsed HTML document.
                  This object is modified in place; identified code block containers
                  are replaced with placeholder tags.
        """
        for i, container in enumerate(potential_containers):
            # Skip if this element or its relevant child was already processed (e.g., pre inside div)
            if container in processed_elements or container.find('code') in processed_elements:
                continue

            code_element = None
            element_to_replace = container
            filename_span = None
            language = "plaintext" # Default

            if container.name == 'div':
                pre_tag = container.find("pre")
                code_element = pre_tag.find("code") if pre_tag else container.find("code")
                if not code_element and pre_tag: # Code directly in pre? 
                        code_element = pre_tag
                
                if not code_element: # If still no code element found in div, skip
                    continue
                    
                # Try finding language from div classes
                wrapper_classes = container.get("class", [])
                for cls in wrapper_classes:
                    if cls.startswith("language-"):
                        language = cls.replace("language-", "")
                        break
                # Try finding filename (more common with div wrappers)
                filename_span = container.find_previous_sibling("span", class_="filename") or container.select_one(".filename")

            elif container.name == 'pre':
                code_element = container.find("code")
                if not code_element: # Code directly in pre?
                    code_element = container
                # Language might be in a class on the <code> tag inside <pre>
                if code_element and code_element.name == 'code':
                        code_classes = code_element.get("class", [])
                        for cls in code_classes:
                            if cls.startswith("language-"):
                                language = cls.replace("language-", "")
                                break
                # Filenames less common with plain <pre>

            if not code_element: # Skip if no code element identified
                    continue

            # Fallback language detection if not found on div/code element classes
            if language == "plaintext" and code_element and code_element.name == 'code':
                code_classes = code_element.get("class", [])
                for cls in code_classes:
                    if cls.startswith("language-"):
                        language = cls.replace("language-", "")
                        break

            filename = filename_span.text if filename_span else ""
            if filename_span:
                processed_elements.add(filename_span)
                filename_span.extract() # Remove it so it's not in the code text
                
            # Get the actual code text
            code_text = code_element.get_text()
                
            placeholder = f"CODE_BLOCK_{i}"
            # Include filename in the markdown if present
            filename_marker = f":{filename}" if filename else ""
            code_blocks[placeholder] = f"```{language}{filename_marker}\\n{code_text.strip()}\\n```"
            
            # Replace the identified container (div or pre) with the placeholder
            placeholder_tag = soup.new_tag("p") # Using <p> tag for replacement
            placeholder_tag.string = placeholder
            processed_elements.add(element_to_replace)
            if code_element != element_to_replace: # Track the inner code element too if different
                    processed_elements.add(code_element)
            element_to_replace.replace_with(placeholder_tag)
        
    
    
    @staticmethod
    def _process_lines(
        lines: List[str], 
        heading_placeholders: Dict[str, str], 
        code_blocks: Dict[str, str],
        markdown_lines: List[str]
    ) -> None:
        """
        Processes individual lines of text, replacing placeholders and handling formatting.

        This method iterates through a list of raw text lines. For each line, it:
        1. Strips leading/trailing whitespace.
        2. Skips empty lines or lines containing common navigation phrases.
        3. Replaces heading placeholders (from `heading_placeholders`) with
           their Markdown equivalents.
        4. Replaces code block placeholders (from `code_blocks`) with their
           Markdown formatted code.
        5. Handles a legacy format for code blocks denoted by leading/trailing
           vertical bars (`|`), converting them to standard Markdown code blocks.
        6. Formats lines containing "Parameters:" or "Returns:" with Markdown bolding.
        7. Appends the processed, non-empty lines to the `markdown_lines` list.

        Args:
            lines: A list of raw string lines extracted from the HTML.
            heading_placeholders: A dictionary mapping placeholder strings to
                                  Markdown-formatted heading strings.
            code_blocks: A dictionary mapping placeholder strings to
                         Markdown-formatted code block strings.
            markdown_lines: A list to which processed Markdown lines will be
                            appended. This argument is modified in place.
        """
        in_code_block = False # State for legacy vertical bar code blocks
        code_block_lines: List[str] = [] # Buffer for legacy code block lines

        for raw_line in lines:
                line = raw_line.strip() # Strip individual lines HERE
                
                # Skip navigation lines and empty lines (after stripping)
                if not line or "Table of contents" in line or "Skip to content" in line:
                    continue

                for placeholder, markdown_heading in heading_placeholders.items():
                     if placeholder in line:
                         line = line.replace(placeholder, markdown_heading)
                
                # Replace ALL occurrences on the line
                for placeholder, markdown_code in code_blocks.items():
                    if placeholder in line:
                        line = line.replace(placeholder, markdown_code)

                # Handle code blocks with vertical bar markers (keep for legacy/other formats)
                if line.startswith("|") and line.endswith("|") and len(line) > 2:
                    if not in_code_block:
                        in_code_block = True
                        code_block_lines = ["```python"] # Assuming python if using bars
                    # Clean the line (remove vertical bars)
                    code_line = line[1:-1].strip()
                    code_block_lines.append(code_line)
                elif in_code_block:
                    # End of code block
                    code_block_lines.append("```")
                    markdown_lines.extend(code_block_lines)
                    in_code_block = False
                    code_block_lines = []
                 # If it was a vertical bar line (start or middle), don't add the original line   
                    continue 
                
                # Handle parameter tables (keep basic formatting for these)
                elif "Parameters:" in line:
                    markdown_lines.append("**Parameters:**")
                elif "Returns:" in line:
                    markdown_lines.append("**Returns:**")
                # Handle everything else as regular text (this now includes lines that had placeholders replaced)
                else:
                    # Avoid adding empty lines resulting from stripped content
                    if line: 
                        markdown_lines.append(line)
                        
    def _markdown_extractor(self, html: str) -> str:
        """Extract content from PydanticAI HTML documentation into Markdown format.
        
        This extractor is specialized for PydanticAI's documentation structure,
        and focuses on preserving code examples and explicit HTML headings.
        It was configured by trial and error. It works semi-accurately for LangChain docs.
        Maybe in the future we'll have a separate extractor for each documentation source.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove navigation elements and other non-content areas
            for element in soup.select("script, style, nav, header, footer"):
                element.extract()

            # --- Process and Replace HTML Headings (h1-h6) ---
            heading_placeholders = {}
            heading_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for i, heading in enumerate(heading_tags):
                level = int(heading.name[1])  # Get heading level from tag name (e.g., 'h2' -> 2)
                placeholder = f"__{heading.name.upper()}_PLACEHOLDER_{i}__"
                markdown_heading = f"{'#' * level} {heading.get_text(strip=True)}"
                heading_placeholders[placeholder] = markdown_heading

                # Replace heading tag with a simple placeholder text node
                placeholder_node = soup.new_string(placeholder)
                heading.replace_with(placeholder_node)

            # Extract HTML code blocks first
            code_blocks = {}

            
            # --- Clean up whitespace-only text nodes ---
            for text_node in soup.find_all(string=True):
                if text_node.strip() == '':
                    text_node.extract()
            
            # Get the text content, preserving some structure
            # Using separator='\n' and NO STRIPPING initially
            text = soup.get_text("\\n", strip=False) 
            
            # Process the text to create markdown
            lines = text.splitlines()
            markdown_lines = []
            in_code_block = False # Keep track of manually defined code blocks (using '|')
            code_block_lines = []
            
            self._process_lines(lines, heading_placeholders, code_blocks, markdown_lines)
            
            # Add any remaining vertical bar code block
            if in_code_block and code_block_lines:
                code_block_lines.append("```")
                markdown_lines.extend(code_block_lines)
            
            # Join and clean up the markdown - filter empty strings before joining
            markdown = "\\n\\n".join(filter(None, markdown_lines))
            return markdown
            
        except Exception as e:
            print(f"Error converting HTML to Markdown: {e}")
            # Fallback still returns text, but maybe log the error more visibly
            logger.error(f"Error in _markdown_extractor: {e}", exc_info=True)
            return soup.get_text(separator="\\n\\n", strip=True)

    def fetch_documentation(self, url: str):
        """
        Fetches documentation from a given URL recursively.

        Args:
            url: The starting URL to fetch documentation from.
            max_depth: The maximum depth for recursive fetching.

        Returns:
            A list of LangChain Document objects, or None if fetching fails.
        """
        logger.info(f"Attempting to fetch documentation from: {url} (max_depth={self.max_depth})")
        try:
            loader = RecursiveUrlLoader(
                url=url,
                max_depth=self.max_depth,
                extractor=self._markdown_extractor,
                prevent_outside=True, # Stay within the initial domain
                use_async=False, # Use synchronous fetching to avoid event loop issues
                timeout=30, # Timeout for each request
                check_response_status=True, # Raise errors for non-200 responses
            )
            # Configure retries (optional, requires 'tenacity' library)
            # loader.session_kwargs = {"retries": 3}

            docs = loader.load()
            logger.info(f"Successfully fetched {len(docs)} documents from {url}")

            # Use the helper function to generate the directory name
            base_dirname = url_to_source_identifier(url)

            # Use Path for directory operations
            output_dir = self.docs_raw_path / base_dirname
            output_dir.mkdir(parents=True, exist_ok=True)
            saved_count = 0

            for i, doc in enumerate(docs):
                try:
                    # Create a filename based on the document source URL if available
                    doc_source = doc.metadata.get('source', f'doc_{i}')
                    # Use Path(urlparse().path).name for basename equivalent
                    doc_filename_base = Path(urlparse(doc_source).path).name or f"page_{i}"
                    # Sanitize filename (Path handles this implicitly better, but manual still ok)
                    safe_filename_stem = "".join(c for c in doc_filename_base if c.isalnum() or c in ('_', '-')).rstrip()
                    if not safe_filename_stem:
                        safe_filename_stem = f"page_{i}"
                    safe_filename = safe_filename_stem + ".json"

                    # Use Path for output path
                    output_path = output_dir / safe_filename

                    # Serialize the Document object's relevant fields to JSON
                    doc_data = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    try:
                        # Use Path.write_text for simple text/json writing
                        output_path.write_text(json.dumps(doc_data, ensure_ascii=False, indent=2), encoding='utf-8')
                        saved_count += 1
                    except TypeError as te:
                        logger.error(f"Serialization error for {output_path}: {te}. Metadata might contain non-serializable types.", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error writing JSON to {output_path}: {e}", exc_info=True)

                except Exception as e:
                    logger.error(f"Error preparing to save document {i} from {url} (source: {doc.metadata.get('source', 'N/A')}): {e}", exc_info=True)

            logger.info(f"Attempted to save content of {len(docs)} documents to {output_dir}, successfully saved: {saved_count}")
            # --- End of basic storage ---

            return docs

        except Exception as e:
            logger.error(f"Failed to fetch documentation from {url}: {e}", exc_info=True)
            return None

    def fetch_all_documentation(self, urls: list[str]):
        """Fetches documentation from a list of URLs."""
        logger.info(f"Starting documentation fetch process for {len(urls)} URLs...")
        all_docs = []
        fetch_success_count = 0
        fetch_fail_count = 0
        for url in urls:
            fetched_docs = self.fetch_documentation(url)
            if fetched_docs is not None:
                all_docs.extend(fetched_docs)
                fetch_success_count += 1
            else:
                fetch_fail_count += 1
        logger.info(f"Documentation fetch process finished. Fetched from {fetch_success_count} URLs, failed for {fetch_fail_count}.")
        return all_docs

    def load_single_document(self, file_path: Path) -> Optional[Document]:
        """
        Loads a single Document object from a JSON file.

        Args:
            file_path: The Path object pointing to the .json file.

        Returns:
            A Document object or None if loading fails.
        """
        if not file_path.is_file():
            logger.error(f"File not found or not a file: {file_path}")
            return None
        try:
            data = json.loads(file_path.read_text(encoding='utf-8'))
            if 'page_content' in data and 'metadata' in data:
                return Document(page_content=data['page_content'], metadata=data['metadata'])
            else:
                logger.warning(f"Skipping file {file_path.name}: Missing 'page_content' or 'metadata' key.")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading document from {file_path.name}: {e}", exc_info=True)
            return None

    def load_documents(self, source_identifier: str) -> List[Document]:
        """
        Loads documents from JSON files stored in a subdirectory identified by source_identifier.
        Uses load_single_document for loading individual files.

        Args:
            source_identifier: The unique identifier for the documentation source,
                               derived from the URL during fetching. Expected values include:
                               - 'langchain-ai.github.io_langgraph_reference'
                               - 'python.langchain.com_api_reference'
                               - 'docs.smith.langchain.com_reference_python_reference'
                               - 'ai.pydantic.dev_index'

        Returns:
            A list of LangChain Document objects loaded from the files.
        """
        # Use Path for directory operations
        load_dir = self.docs_raw_path / source_identifier
        loaded_docs: List[Document] = []

        # Use Path.is_dir()
        if not load_dir.is_dir():
            logger.warning(f"Directory not found for identifier '{source_identifier}': {load_dir}")
            return loaded_docs

        logger.info(f"Loading documents from: {load_dir}")
        file_count = 0
        success_count = 0
        load_errors = 0

        # Use Path.iterdir() and check suffix
        for file_path in load_dir.iterdir():
            if file_path.is_file() and file_path.suffix == ".json":
                file_count += 1
                doc = self.load_single_document(file_path)
                if doc:
                    loaded_docs.append(doc)
                    success_count += 1
                else:
                    load_errors += 1 # Error logged within load_single_document

        logger.info(f"Finished loading from {load_dir}. Loaded {success_count} documents successfully ({load_errors} errors out of {file_count} JSON files).")
        return loaded_docs 
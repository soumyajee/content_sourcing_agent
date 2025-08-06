import os
import requests
import PyPDF2
import uuid
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import warnings
from urllib.parse import urlparse, urljoin

# Suppress DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JSON file for audit logs
AUDIT_LOG_FILE = "audit_logs.json"

def init_audit_log_file():
    """Initialize the audit log JSON file if it doesn't exist."""
    if not os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "w") as f:
            json.dump([], f)

def append_audit_log(log_entry: dict):
    """Append an audit log entry to the JSON file."""
    try:
        with open(AUDIT_LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []
    
    logs.append(log_entry)
    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)
    logger.info(f"Logged audit entry to {AUDIT_LOG_FILE}")

def is_valid_pdf(url: str) -> bool:
    """Validate if a URL points to a valid PDF."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept": "application/pdf"
        }
        response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
        if response.status_code != 200:
            return False
        content_type = response.headers.get("Content-Type", "").lower()
        return "application/pdf" in content_type
    except Exception as e:
        logger.error(f"Failed to validate PDF URL {url}: {str(e)}")
        return False

def validate_url(url: str) -> str:
    """Validate and clean a URL."""
    url = url.strip()
    if url.endswith(")"):
        url = url[:-1]  # Remove trailing parenthesis
    try:
        result = urlparse(url)
        if not result.scheme:
            url = "https://" + url
        return url
    except Exception as e:
        logger.error(f"Invalid URL {url}: {str(e)}")
        return ""

def scrape_wikipedia_text(url: str) -> str:
    """Scrape text content from a Wikipedia page using the API."""
    try:
        title = url.split("/wiki/")[-1]
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True
        }
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        return page.get("extract", "")
    except Exception as e:
        logger.error(f"Error scraping Wikipedia text {url}: {str(e)}")
        return ""

def scrape_pdf_links(url: str, subject: str = "algebra") -> list:
    """Scrape PDF links from a URL using requests and BeautifulSoup, filtering by subject."""
    try:
        url = validate_url(url)
        if not url:
            return []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".pdf")]
        
        # Log snippet of HTML for debugging
        logger.debug(f"HTML snippet from {url}: {str(soup)[:500]}")
        
        # Filter for algebra-related PDFs
        filtered_links = [link for link in pdf_links if subject.lower() in link.lower() or "math" in link.lower()]
        if not filtered_links:
            filtered_links = pdf_links  # Fallback to all PDFs if no subject-specific ones
        
        # Ensure absolute URLs and validate
        valid_links = []
        for link in filtered_links:
            absolute_url = urljoin(url, link)
            if is_valid_pdf(absolute_url) and "web.archive.org" not in absolute_url:
                valid_links.append(absolute_url)
        
        if not valid_links:
            logger.warning(f"No valid PDF links found on {url}")
            return []
        
        logger.info(f"Found {len(valid_links)} valid PDF links on {url}: {valid_links}")
        return list(set(valid_links))[:10]  # Limit to 10 URLs
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return []

def format_pdf_links(pdf_links: list) -> str:
    """Format PDF links as JSON using Groq LLM with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            llm = ChatGroq(
                model="llama3-8b-8192",
                api_key=GROQ_API_KEY,
                temperature=0.5,
                max_tokens=4000
            )
            prompt = ChatPromptTemplate.from_template(
                "Convert the following list of PDF URLs into a JSON list of objects with 'url' keys: {urls}. "
                "Return only valid JSON, e.g., [{\"url\": \"https://example.com/sample.pdf\"}]."
            )
            response = llm.invoke(prompt.format(urls=json.dumps(pdf_links)))
            json_output = response.content.strip()
            
            # Validate JSON
            json.loads(json_output)
            logger.info("Successfully formatted PDF links as JSON")
            return json_output
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from LLM (attempt {attempt + 1}/{max_retries}): {json_output}, error: {str(e)}")
        except Exception as e:
            logger.error(f"LLM formatting error (attempt {attempt + 1}/{max_retries}): {str(e)}")
    logger.warning("Falling back to basic JSON formatting after LLM retries")
    return json.dumps([{"url": url} for url in pdf_links])

def download_pdf(url: str, local_dir: str = "./pdfs") -> tuple[str, str]:
    """Download a PDF from a URL and save it locally."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            "Accept": "application/pdf"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()
        filename = f"{uuid.uuid4()}.pdf"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        with open(local_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Downloaded PDF to {local_path}")
        return local_path, filename
    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {str(e)}")
        raise

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        logger.info(f"Extracted text from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        raise

def source_pdf_content(curriculum_id: int, subject: str, source_urls: list) -> list[dict]:
    """Source and scrape PDF or text content from approved URLs."""
    try:
        init_audit_log_file()
        scraped_data = []

        for url in source_urls:
            url = validate_url(url)
            if not url:
                logger.warning(f"Skipping invalid URL: {url}")
                continue

            if "wikipedia.org/wiki" in url:
                # Scrape Wikipedia text
                fragment_id = str(uuid.uuid4())
                text = scrape_wikipedia_text(url)
                if text:
                    audit_log = {
                        "id": len(scraped_data) + 1,
                        "user_id": "system",
                        "action": "wiki_sourced",
                        "timestamp": datetime.utcnow().isoformat(),
                        "details": {
                            "curriculum_id": curriculum_id,
                            "subject": subject,
                            "source_url": url,
                            "fragment_id": fragment_id
                        }
                    }
                    append_audit_log(audit_log)
                    scraped_data.append({
                        "fragment_id": fragment_id,
                        "wiki_url": url,
                        "wiki_text": text,
                        "local_path": None
                    })
                # Also try PDF scraping
                pdf_links = scrape_pdf_links(url, subject)
            elif url.endswith(".pdf"):
                if is_valid_pdf(url):
                    pdf_links = [{"url": url}]
                else:
                    logger.warning(f"Invalid or inaccessible PDF URL: {url}")
                    pdf_links = []
            else:
                pdf_links = scrape_pdf_links(url, subject)
                if not pdf_links and "ck12.org" in url:
                    logger.info(f"CK-12 page may require JavaScript. Please provide a direct PDF URL.")
                    print("Suggested URL: https://assets.openstax.org/oscms-prodcms/media/documents/AlgebraandTrigonometry-OP.pdf")
                    manual_url = input("Enter a direct PDF URL (or press Enter to skip): ").strip()
                    if manual_url and manual_url.endswith(".pdf"):
                        manual_url = validate_url(manual_url)
                        if is_valid_pdf(manual_url):
                            pdf_links = [{"url": manual_url}]
                        else:
                            logger.warning(f"Manual URL {manual_url} is inaccessible")
                            continue
                    else:
                        logger.info("No manual URL provided, skipping CK-12 source")
                        continue

            if not pdf_links:
                logger.warning(f"No PDF links found for {url}")
                continue

            pdf_links_json = format_pdf_links([link["url"] if isinstance(link, dict) else link for link in pdf_links])
            try:
                pdf_links = json.loads(pdf_links_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse pdf_links as JSON for {url}: {str(e)}")
                continue

            if not isinstance(pdf_links, list):
                pdf_links = [pdf_links]

            for item in pdf_links:
                pdf_url = item.get("url") if isinstance(item, dict) else item
                if not pdf_url or not isinstance(pdf_url, str) or not pdf_url.endswith(".pdf"):
                    logger.warning(f"Skipping invalid or non-PDF URL: {pdf_url}")
                    continue

                fragment_id = str(uuid.uuid4())
                try:
                    local_path, filename = download_pdf(pdf_url)
                    pdf_text = extract_pdf_text(local_path)
                except Exception as e:
                    logger.error(f"Failed to process PDF {pdf_url}: {str(e)}")
                    continue

                audit_log = {
                    "id": len(scraped_data) + 1,
                    "user_id": "system",
                    "action": "pdf_sourced",
                    "timestamp": datetime.utcnow().isoformat(),
                    "details": {
                        "curriculum_id": curriculum_id,
                        "subject": subject,
                        "source_url": pdf_url,
                        "fragment_id": fragment_id,
                        "local_filename": filename
                    }
                }
                append_audit_log(audit_log)

                scraped_data.append({
                    "fragment_id": fragment_id,
                    "pdf_url": pdf_url,
                    "pdf_text": pdf_text,
                    "local_path": local_path
                })

        logger.info(f"Sourced {len(scraped_data)} fragments (PDFs or text)")
        return scraped_data

    except Exception as e:
        logger.error(f"Error sourcing content: {str(e)}")
        raise

def main():
    print("PDF Content Sourcing Agent")
    try:
        curriculum_id = int(input("Enter Curriculum ID (e.g., 1): "))
        subject = input("Enter Subject (e.g., Math): ")
        source_urls = input("Enter Source URLs (comma-separated, e.g.,https://en.wikipedia.org/wiki/Algebra): ").split(",")
        source_urls = [url.strip() for url in source_urls]

        fragments = source_pdf_content(curriculum_id, subject, source_urls)
        with open("pdf_data.json", "w") as f:
            json.dump(fragments, f, indent=2)
        for f in fragments:
            if "pdf_url" in f:
                print(f"Fragment ID: {f['fragment_id']}, PDF URL: {f['pdf_url']}, Local Path: {f['local_path']}")
            else:
                print(f"Fragment ID: {f['fragment_id']}, Wiki URL: {f['wiki_url']}, Text Length: {len(f['wiki_text'])}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
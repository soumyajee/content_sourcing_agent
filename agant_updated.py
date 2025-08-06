#!/usr/bin/env python3
"""
Content Sourcing Agent using Groq API with Bloom's Taxonomy assessment
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import logging
import validators
import requests
from bs4 import BeautifulSoup
from groq import Groq, NotFoundError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from config import get_config, AgentConfig
# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class AgentConfig:
    """Configuration for the agent"""
    LLM_API_KEY: str = os.getenv('GROQ_API_KEY', '')
    LLM_MODEL: str = os.getenv('GROQ_MODEL', 'gemma2-9b-it')
    LLM_BASE_URL: str = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/')
    QUALITY_THRESHOLD: float = float(os.getenv('QUALITY_THRESHOLD', 0.5))
    MAX_TOKENS: int = int(os.getenv('MAX_TOKENS', 500))
    DEFAULT_QUALITY_SCORE: float = float(os.getenv('DEFAULT_QUALITY_SCORE', 0.5))

def get_config() -> AgentConfig:
    """Get agent configuration from environment variables"""
    return AgentConfig()

@dataclass
class ContentItem:
    """Data structure for content items"""
    id: str
    title: str
    content: str
    source_url: str
    category: str
    tags: List[str]
    timestamp: str
    quality_score: float
    metadata: Dict[str, Any]
    bloom_level: str = Field(default="Unknown")

class AgentState(BaseModel):
    """State management for the LangGraph agent"""
    query: str = ""
    sources: List[str] = Field(default_factory=list)
    raw_content: List[Dict] = Field(default_factory=list)
    processed_content: List[ContentItem] = Field(default_factory=list)
    stored_content: List[str] = Field(default_factory=list)
    current_step: str = "start"
    errors: List[str] = Field(default_factory=list)

class ContentAPI:
    """API for storing and retrieving content"""
    def __init__(self):
        self.storage = {}
        self.counter = 0

    def store_content(self, content_item: ContentItem) -> str:
        """Store content item and return ID"""
        self.counter += 1
        item_id = f"content_{self.counter}"
        self.storage[item_id] = asdict(content_item)
        return item_id

    def get_content(self, content_id: str) -> Optional[Dict]:
        """Retrieve content by ID"""
        return self.storage.get(content_id)

    def list_all_content(self) -> List[Dict]:
        """List all stored content"""
        return list(self.storage.values())

    def search_content(self, query: str) -> List[Dict]:
        """Search content by query"""
        results = []
        query_lower = query.lower()
        for content in self.storage.values():
            if (query_lower in content['title'].lower() or
                query_lower in content['content'].lower() or
                any(query_lower in tag.lower() for tag in content['tags'])):
                results.append(content)
        return results

class ConfigurableLLM:
    """Configurable LLM class using Groq API"""
    def __init__(self, api_key: str = "", model: str = "", base_url: str = "", max_tokens: int = 500):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        if not model:
            raise ValueError("GROQ_MODEL is required")
        if not base_url:
            raise ValueError("GROQ_BASE_URL is required")
        
        self.client = Groq(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        logger.info(f"Initialized Groq LLM: {base_url} with model: {model}")

    def invoke(self, prompt: str, max_tokens: int = None, temperature: float = 0.7) -> str:
        """Invoke the Groq LLM with a prompt"""
        if not prompt:
            raise ValueError("Prompt cannot be empty or None")
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except NotFoundError as e:
                logger.error(f"Failed to invoke Groq API: {e}")
                raise Exception(f"Failed to invoke Groq API: {e}. Ensure the endpoint is {self.client.base_url}/chat/completions and the model is valid.")
            except Exception as e:
                if '429' in str(e) and attempt < 2:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"Error calling Groq LLM at {self.client.base_url}: {e}")
                return ""
        return ""

    def assess_bloom_taxonomy(self, content: str) -> str:
        """Use Groq LLM to assess content based on Bloom's Taxonomy"""
        if not content:
            return "Unknown"
        prompt = f"""
        Analyze the following educational content and determine the highest applicable level according to Bloom's Taxonomy:
        - Remembering
        - Understanding
        - Applying
        - Analyzing
        - Evaluating
        - Creating
        
        Assign the content to the highest level it demonstrates. Return only the level name (e.g., 'remembering', 'understanding', etc.).
        Example: If the content involves basic recall, return 'remembering'. If it includes analysis, return 'analyzing'.
        
        Content: {content[:500]}...
        
        Level:
        """
        try:
            response = self.invoke(prompt, max_tokens=50, temperature=0.3)
            response = response.strip().lower()
            logger.debug(f"Raw Bloom's Taxonomy response: {response}")
            # Valid Bloom levels
            bloom_levels = ["remembering", "understanding", "applying", "analyzing", "evaluating", "creating"]
            # Check if response matches any Bloom level
            for level in bloom_levels:
                if level in response:
                    return level
            logger.warning(f"Invalid Bloom's Taxonomy response: {response}, defaulting to 'Unknown'")
            return "Unknown"
        except Exception as e:
            logger.error(f"Error assessing Bloom's Taxonomy: {e}")
            return "Unknown"

    def categorize_content(self, content: str) -> str:
        """Use Groq LLM to categorize content"""
        prompt = f"""
        Analyze the following educational content and categorize it into one of these categories:
        - artificial_intelligence
        - computer_science  
        - mathematics
        - science
        - technology
        - education
        - general
        
        Content: {content[:500]}...
        
        Return only the category name:
        """
        response = self.invoke(prompt, max_tokens=50, temperature=0.3)
        category = response.strip().lower()
        valid_categories = [
            'artificial_intelligence', 'computer_science', 'mathematics',
            'science', 'technology', 'education', 'general'
        ]
        return category if category in valid_categories else 'general'

    def extract_tags(self, content: str) -> List[str]:
        """Use Groq LLM to extract relevant tags"""
        prompt = f"""
        Extract 3-5 relevant educational tags from the following content.
        Return only the tags, separated by commas.
        
        Content: {content[:500]}...
        
        Tags:
        """
        response = self.invoke(prompt, max_tokens=100, temperature=0.3)
        if response:
            tags = [tag.strip().lower() for tag in response.split(',')]
            return tags[:5]
        return []

    def assess_quality(self, title: str, content: str) -> float:
        """Use Groq LLM to assess content quality"""
        prompt = f"""
        Assess the educational quality of this content on a scale of 0.0 to 1.0.
        Consider factors like:
        - Educational value
        - Clarity and structure
        - Completeness
        - Accuracy
        
        Title: {title}
        Content: {content[:800]}...
        
        Return only a decimal number between 0.0 and 1.0:
        """
        response = self.invoke(prompt, max_tokens=50, temperature=0.3)
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            logger.warning("Failed to parse quality score, using default")
            return float(os.getenv('DEFAULT_QUALITY_SCORE', 0.5))

class ContentSourcingAgent:
    """Main Content Sourcing Agent using LangGraph"""
    def __init__(self, api_key: Optional[str] = None, model: str = "", base_url: str = "", max_tokens: int = None):
        self.content_api = ContentAPI()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        try:
            self.llm = ConfigurableLLM(
                api_key or os.getenv('GROQ_API_KEY', ''),
                model or os.getenv('GROQ_MODEL', 'gemma2-9b-it'),
                base_url or os.getenv('GROQ_BASE_URL', 'https://api.groq.com/'),
                max_tokens or int(os.getenv('MAX_TOKENS', 500))
            )
        except ValueError as e:
            logger.error(f"LLM initialization failed: {e}")
            self.llm = None
            logger.warning("Using rule-based processing instead")
        
        logger.info("Real web content fetching enabled - no fake content will be used")
        logger.info("Agent ready to process URLs")
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        workflow.add_node("initialize", self._initialize_sources)
        workflow.add_node("fetch_content", self._fetch_content)
        workflow.add_node("process_content", self._process_content)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("store_content", self._store_content)
        workflow.add_node("finalize", self._finalize)
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "fetch_content")
        workflow.add_edge("fetch_content", "process_content")
        workflow.add_edge("process_content", "quality_check")
        workflow.add_edge("quality_check", "store_content")
        workflow.add_edge("store_content", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    def _initialize_sources(self, state: AgentState) -> AgentState:
        """Initialize sources from the provided sources list"""
        logger.info(f"Initializing sources for query: {state.query}")
        if not state.sources:
            raise ValueError("No sources provided. Please provide a list of URLs to process.")
        state.current_step = "initialize"
        valid_sources = []
        for url in state.sources:
            if validators.url(url):
                valid_sources.append(url)
            else:
                error_msg = f"Invalid URL skipped: {url}"
                state.errors.append(error_msg)
                logger.warning(error_msg)
        state.sources = valid_sources
        if not valid_sources:
            raise ValueError("No valid URLs provided after validation.")
        logger.info(f"Selected {len(state.sources)} valid sources")
        return state

    def _fetch_content(self, state: AgentState) -> AgentState:
        """Fetch content from sources"""
        logger.info("Fetching real content from sources...")
        state.current_step = "fetch_content"
        for source_url in state.sources:
            try:
                real_content = self._fetch_content_from_url(source_url)
                state.raw_content.append({
                    'url': source_url,
                    'content': real_content['content'],
                    'title': real_content['title']
                })
                logger.info(f"Fetched real content from: {source_url}")
            except Exception as e:
                error_msg = f"Failed to fetch from {source_url}: {str(e)}"
                state.errors.append(error_msg)
                logger.error(error_msg)
        logger.info(f"Successfully fetched content from {len(state.raw_content)} sources")
        if len(state.raw_content) == 0:
            logger.warning("No content was successfully fetched from any source")
        return state

    def _fetch_content_from_url(self, url: str) -> Dict[str, str]:
        """Fetch actual content from any URL"""
        try:
            logger.info(f"Fetching content from: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.text.strip() if title_tag else "Untitled"
            for script in soup(["script", "style"]):
                script.decompose()
            content = ""
            content_strategies = [
                self._extract_site_specific_content(soup, url),
                self._extract_article_content(soup),
                self._extract_generic_content(soup)
            ]
            for strategy_content in content_strategies:
                if strategy_content and len(strategy_content.strip()) > 100:
                    content = strategy_content
                    break
            if not content or len(content.strip()) < 50:
                body = soup.find('body')
                if body:
                    content = body.get_text()
            content = ' '.join(content.split())
            if len(content) > 8000:
                content = content[:8000] + "..."
            if not content or len(content.strip()) < 50:
                raise Exception("Insufficient content extracted")
            return {'title': title, 'content': content.strip()}
        except requests.exceptions.RequestException as e:
            raise Exception(f"HTTP request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Content extraction failed: {str(e)}")

    def _extract_site_specific_content(self, soup: BeautifulSoup, url: str) -> str:
        """Extract content using site-specific optimizations"""
        content = ""
        if 'wikipedia.org' in url:
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if content_div:
                paragraphs = content_div.find_all('p')
                content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        elif 'medium.com' in url:
            selectors_to_try = [
                'article', '[data-testid="storyContent"]', '.postArticle-content',
                '.section-content', '.story-content', 'main'
            ]
            for selector in selectors_to_try:
                content_elem = soup.select_one(selector)
                if content_elem:
                    paragraphs = content_elem.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'blockquote'])
                    content = '\n\n'.join([elem.get_text().strip() for elem in paragraphs if elem.get_text().strip()])
                    break
        elif 'arxiv.org' in url:
            abstract_elem = soup.find('blockquote', class_='abstract')
            if abstract_elem:
                content = abstract_elem.get_text().strip()
        elif 'nature.com' in url or 'science.org' in url or 'ieee.org' in url:
            selectors_to_try = [
                'article', '.article-content', '.content', '#content',
                '.main-content', '[role="main"]'
            ]
            for selector in selectors_to_try:
                content_elem = soup.select_one(selector)
                if content_elem:
                    paragraphs = content_elem.find_all(['p', 'div', 'section'])
                    content = '\n\n'.join([elem.get_text().strip() for elem in paragraphs if elem.get_text().strip() and len(elem.get_text().strip()) > 30])
                    break
        elif 'khanacademy.org' in url:
            selectors_to_try = [
                'main', '.main-content', '[data-test-id="exercise-content"]',
                '.article-content', '.perseus-renderer', '.content-wrap', '#content'
            ]
            main_content = None
            for selector in selectors_to_try:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            if main_content:
                content_elements = main_content.find_all(['p', 'div', 'section', 'article', 'h1', 'h2', 'h3', 'h4', 'li'])
                content_parts = []
                for elem in content_elements:
                    text = elem.get_text().strip()
                    if (len(text) > 20 and
                        not any(skip in text.lower() for skip in ['cookie', 'login', 'sign up', 'navigation', 'menu'])):
                        content_parts.append(text)
                content = '\n\n'.join(content_parts)
        elif 'towardsdatascience.com' in url or 'blog.' in url or 'substack.com' in url:
            selectors_to_try = [
                'article', '.post-content', '.entry-content', '.content',
                'main', '[role="main"]'
            ]
            for selector in selectors_to_try:
                content_elem = soup.select_one(selector)
                if content_elem:
                    paragraphs = content_elem.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'blockquote'])
                    content = '\n\n'.join([elem.get_text().strip() for elem in paragraphs if elem.get_text().strip()])
                    break
        return content

    def _extract_article_content(self, soup: BeautifulSoup) -> str:
        """Extract content using common article patterns"""
        article_selectors = [
            'article', '[role="main"]', 'main', '.article-content',
            '.post-content', '.entry-content', '.content', '.main-content',
            '#content', '#main', '.story-content', '.article-body', '.post-body'
        ]
        for selector in article_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content_elements = content_elem.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'li'])
                content_parts = []
                for elem in content_elements:
                    text = elem.get_text().strip()
                    if (len(text) > 15 and
                        not any(skip in text.lower() for skip in [
                            'cookie', 'subscribe', 'newsletter', 'advertisement',
                            'login', 'sign up', 'navigation', 'menu', 'footer',
                            'share this', 'follow us', 'read more'
                        ])):
                        content_parts.append(text)
                if content_parts:
                    return '\n\n'.join(content_parts)
        return ""

    def _extract_generic_content(self, soup: BeautifulSoup) -> str:
        """Generic content extraction as final fallback"""
        paragraphs = soup.find_all(['p', 'div', 'section'])
        content_parts = []
        for elem in paragraphs:
            text = elem.get_text().strip()
            if (len(text) > 50 and
                not any(skip in text.lower() for skip in [
                    'javascript', 'cookie policy', 'privacy policy',
                    'terms of service', '© copyright', 'all rights reserved'
                ])):
                content_parts.append(text)
        return '\n\n'.join(content_parts[:20]) if content_parts else ""

    def _process_content(self, state: AgentState) -> AgentState:
        """Process and structure the raw content"""
        logger.info("Processing content...")
        state.current_step = "process_content"
        for item in state.raw_content:
            try:
                processed_item = ContentItem(
                    id=f"item_{len(state.processed_content) + 1}",
                    title=item['title'],
                    content=item['content'].strip(),
                    source_url=item['url'],
                    category=self._determine_category(item['content']),
                    tags=self._extract_tags(item['content']),
                    timestamp=datetime.now().isoformat(),
                    bloom_level=self._determine_bloom_level(item['content']),
                    quality_score=0.0,
                    metadata={
                        'word_count': len(item['content'].split()),
                        'processed_at': datetime.now().isoformat()
                    }
                )
                state.processed_content.append(processed_item)
                logger.info(f"Processed: {processed_item.title}")
            except Exception as e:
                error_msg = f"Failed to process content: {str(e)}"
                state.errors.append(error_msg)
                logger.error(error_msg)
        logger.info(f"Processed {len(state.processed_content)} content items")
        return state

    def _determine_bloom_level(self, content: str) -> str:
        """Determine Bloom's Taxonomy level based on content analysis"""
        if self.llm:
            try:
                return self.llm.assess_bloom_taxonomy(content)
            except Exception as e:
                logger.warning(f"LLM Bloom's Taxonomy assessment failed, using default: {e}")
        return "Unknown"

    def _determine_category(self, content: str) -> str:
        """Determine content category based on content analysis"""
        if self.llm:
            try:
                category = self.llm.categorize_content(content)
                if category != 'general':
                    return category
            except Exception as e:
                logger.warning(f"LLM categorization failed, using rule-based: {e}")
        content_lower = content.lower()
        categories = {
            'artificial_intelligence': ['artificial intelligence', 'ai', 'machine learning', 'neural network'],
            'computer_science': ['algorithm', 'programming', 'software', 'computer'],
            'mathematics': ['mathematics', 'math', 'statistics', 'calculus', 'algebra'],
            'technology': ['technology', 'digital', 'innovation', 'tech'],
            'science': ['research', 'study', 'scientific', 'analysis'],
            'education': ['learning', 'educational', 'teaching', 'curriculum']
        }
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        return 'general'

    def _extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content"""
        if self.llm:
            try:
                tags = self.llm.extract_tags(content)
                if tags:
                    return tags
            except Exception as e:
                logger.warning(f"LLM tag extraction failed, using rule-based: {e}")
        content_lower = content.lower()
        potential_tags = [
            'ai', 'machine learning', 'computer science', 'technology',
            'education', 'research', 'programming', 'algorithms',
            'data science', 'software', 'innovation', 'digital',
            'mathematics', 'statistics', 'calculus', 'algebra'
        ]
        tags = [tag for tag in potential_tags if tag in content_lower]
        return tags[:5]

    def _quality_check(self, state: AgentState) -> AgentState:
        """Perform quality check on processed content"""
        logger.info("Performing quality check...")
        state.current_step = "quality_check"
        for item in state.processed_content:
            try:
                item.quality_score = self._calculate_quality_score(item)
                logger.info(f"Quality score for '{item.title}': {item.quality_score:.2f}")
            except Exception as e:
                error_msg = f"Quality check failed for {item.title}: {str(e)}"
                state.errors.append(error_msg)
                logger.error(error_msg)
        quality_threshold = get_config().QUALITY_THRESHOLD
        state.processed_content = [
            item for item in state.processed_content
            if item.quality_score >= quality_threshold
        ]
        logger.info(f"{len(state.processed_content)} items passed quality check")
        return state

    def _calculate_quality_score(self, item: ContentItem) -> float:
        """Calculate quality score for content item"""
        if self.llm:
            try:
                llm_score = self.llm.assess_quality(item.title, item.content)
                rule_based_score = self._rule_based_quality_score(item)
                return (llm_score * 0.7) + (rule_based_score * 0.3)
            except Exception as e:
                logger.warning(f"LLM quality assessment failed, using rule-based: {e}")
        return self._rule_based_quality_score(item)

    def _rule_based_quality_score(self, item: ContentItem) -> float:
        """Rule-based quality scoring"""
        score = 0.0
        word_count = len(item.content.split())
        if 50 <= word_count <= 1000:
            score += 0.3
        elif word_count > 1000:
            score += 0.2
        if len(item.title) > 10 and len(item.title) < 100:
            score += 0.2
        if len(item.tags) >= 2:
            score += 0.2
        if item.category != 'general':
            score += 0.15
        if any(domain in item.source_url for domain in ['wikipedia.org', 'edu', 'gov', 'khanacademy.org']):
            score += 0.15
        return min(score, 1.0)

    def _store_content(self, state: AgentState) -> AgentState:
        """Store processed content using content API"""
        logger.info("Storing content...")
        state.current_step = "store_content"
        for item in state.processed_content:
            try:
                stored_id = self.content_api.store_content(item)
                state.stored_content.append(stored_id)
                logger.info(f"Stored content: {item.title} (ID: {stored_id})")
            except Exception as e:
                error_msg = f"Failed to store {item.title}: {str(e)}"
                state.errors.append(error_msg)
                logger.error(error_msg)
        logger.info(f"Successfully stored {len(state.stored_content)} items")
        return state

    def _finalize(self, state: AgentState) -> AgentState:
        """Finalize the agent execution"""
        logger.info("Finalizing agent execution...")
        state.current_step = "finalize"
        print("\n" + "="*50)
        print("EXECUTION SUMMARY")
        print("="*50)
        print(f"Query: {state.query}")
        print(f"Sources processed: {len(state.sources)}")
        print(f"Content items fetched: {len(state.raw_content)}")
        print(f"Content items processed: {len(state.processed_content)}")
        print(f"Content items stored: {len(state.stored_content)}")
        print(f"Errors encountered: {len(state.errors)}")
        if state.errors:
            print("\nErrors:")
            for error in state.errors:
                print(f"  - {error}")
        print("\nAgent execution completed successfully!")
        return state

    def run(self, query: str, sources: List[str]) -> AgentState:
        """Run the content sourcing agent with provided sources"""
        logger.info(f"Starting Content Sourcing Agent with query: {query}")
        logger.info(f"Sources to process: {len(sources)}")
        if not sources:
            raise ValueError("No sources provided. Please provide a list of URLs to process.")
        initial_state = AgentState(query=query, sources=sources)
        final_state = self.workflow.invoke(initial_state)
        return final_state

    def search_stored_content(self, query: str) -> List[Dict]:
        """Search stored content"""
        return self.content_api.search_content(query)

    def get_all_stored_content(self) -> List[Dict]:
        """Get all stored content"""
        return self.content_api.list_all_content()

def main():
    """Main function to demonstrate the agent with custom URL sources"""
    print("Universal Content Sourcing Agent Demo")
    print("="*45)
    print("Using Groq API for LLM processing")
    print("Provide URLs via TEST_SOURCES environment variable")
    logger.debug(f"TEST_SOURCES: {os.getenv('TEST_SOURCES')}")
    
    config = get_config()
    try:
        agent = ContentSourcingAgent(
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            max_tokens=config.MAX_TOKENS
        )
        query = os.getenv('TEST_QUERY', 'artificial intelligence and machine learning educational content')
        sources = os.getenv('TEST_SOURCES', '').split(',') if os.getenv('TEST_SOURCES') else []
        
        if not sources or all(not s.strip() for s in sources):
            logger.warning("No valid sources provided in TEST_SOURCES. Using default sources.")
            sources = [
                "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "https://en.wikipedia.org/wiki/Machine_learning",
                "https://www.nature.com/articles/s41586-023-06647-8",
                "https://arxiv.org/abs/2303.08774",
                "https://medium.com/towards-data-science/understanding-machine-learning-in-5-minutes-1b6b4b7a8d0a"
            ]
        
        print("\nSources to process:")
        for i, url in enumerate(sources, 1):
            print(f"   {i}. {url}")
        print(f"\nQuery: {query}")
        
        result = agent.run(query, sources)
        print("\n" + "="*60)
        print("STORED CONTENT")
        print("="*60)
        stored_content = agent.get_all_stored_content()
        if stored_content:
            for i, content in enumerate(stored_content, 1):
                print(f"\n{i}. {content['title']}")
                print(f"   Category: {content['category']}")
                print(f"   Tags: {', '.join(content['tags'])}")
                print(f"   Quality Score: {content['quality_score']:.2f}")
                print(f"   Source: {content['source_url']}")
                print(f"   Word Count: {content['metadata']['word_count']}")
                print(f"   Bloom Level: {content.get('bloom_level', 'Unknown')}")
                content_preview = content['content'][:200] + "..." if len(content['content']) > 200 else content['content']
                print(f"   Preview: {content_preview}")
        else:
            print("No content was stored (may have been filtered by quality threshold)")
        print("\n" + "="*60)
        print("SEARCH DEMO - Finding AI Related Content")
        print("="*60)
        search_results = agent.search_stored_content("artificial intelligence")
        print(f"Found {len(search_results)} results for 'artificial intelligence':")
        for result in search_results:
            print(f"- {result['title']} (Source: {result['source_url']})")
        print(f"\nSuccessfully processed {len(stored_content)} sources!")
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        print(f"Error running agent: {e}")

def demo_with_different_sources():
    """Demonstrate agent with different types of sources"""
    print("\n" + "="*60)
    print("DEMO: Different Source Types")
    print("="*60)
    config = get_config()
    try:
        agent = ContentSourcingAgent(
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            max_tokens=config.MAX_TOKENS
        )
        tech_sources = os.getenv('TECH_SOURCES', '').split(',') if os.getenv('TECH_SOURCES') else [
            "https://www.technologyreview.com/2023/04/27/1072102/the-future-of-generative-ai-is-nearing-a-turning-point/",
            "https://spectrum.ieee.org/ai-chatbot-large-language-model",
            "https://www.scientificamerican.com/article/how-ai-will-change-science/"
        ]
        query = os.getenv('TECH_QUERY', 'latest AI developments and future predictions')
        print("Tech news sources:")
        for i, url in enumerate(tech_sources, 1):
            print(f"   {i}. {url}")
        print(f"\nQuery: {query}")
        result = agent.run(query, tech_sources)
        print(f"\nResults: {len(agent.get_all_stored_content())} articles processed")
        print("\nTO USE WITH YOUR OWN URLs:")
        print("1. Set TEST_SOURCES or TECH_SOURCES in .env with comma-separated URLs")
        print("2. Supports: journals, Medium, blogs, news sites, ArXiv, etc.")
        print("3. Set TEST_QUERY or TECH_QUERY for custom queries")
        print("4. Call agent.run(query, your_sources)")
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"Error running demo: {e}")

if __name__ == "__main__":
    main()
    # demo_with_different_sources()
"""
Blog reader module - Fetch and clean blog content for TTS
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
from rich.console import Console
from urllib.parse import urlparse
import re

console = Console()


def fetch_blog_content(url: str) -> dict:
    """
    Fetch blog post content from a URL.
    
    Args:
        url: The blog post URL
    
    Returns:
        Dict with title, content, and source
    """
    console.print(f"\n[cyan]Fetching:[/cyan] {url}")
    
    # Full browser-like headers to avoid 403 blocks (e.g., Medium)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'lxml')
    
    # Extract title
    title = extract_title(soup)
    
    # Extract main content
    content = extract_content(soup)
    
    # Clean the text
    cleaned_content = clean_text(content)
    
    console.print(f"[green]✓[/green] Title: {title}")
    console.print(f"[green]✓[/green] Content: {len(cleaned_content)} characters")
    
    return {
        'title': title,
        'content': cleaned_content,
        'source': urlparse(url).netloc,
        'url': url
    }


def extract_title(soup: BeautifulSoup) -> str:
    """Extract the article title."""
    # Try common title patterns
    selectors = [
        'h1.entry-title',
        'h1.post-title',
        'h1.article-title',
        'article h1',
        '.post h1',
        'h1',
        'title'
    ]
    
    for selector in selectors:
        element = soup.select_one(selector)
        if element and element.get_text(strip=True):
            return element.get_text(strip=True)
    
    return "Untitled"


def extract_content(soup: BeautifulSoup) -> str:
    """Extract the main article content."""
    # Remove unwanted elements
    for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 
                              'aside', 'form', 'iframe', 'noscript']):
        tag.decompose()
    
    # Remove common ad/navigation classes
    unwanted_classes = [
        'sidebar', 'navigation', 'menu', 'footer', 'header',
        'advertisement', 'social', 'share', 'related', 'comments',
        'newsletter', 'popup', 'modal', 'cookie'
    ]
    
    for class_name in unwanted_classes:
        for element in soup.find_all(class_=re.compile(class_name, re.I)):
            element.decompose()
    
    # Try common content selectors
    content_selectors = [
        'article .entry-content',
        'article .post-content',
        '.article-content',
        '.post-body',
        'article',
        '.content',
        'main',
        '#content'
    ]
    
    for selector in content_selectors:
        content = soup.select_one(selector)
        if content:
            return content.get_text(separator='\n', strip=True)
    
    # Fallback to body
    body = soup.find('body')
    if body:
        return body.get_text(separator='\n', strip=True)
    
    return soup.get_text(separator='\n', strip=True)


def clean_text(text: str) -> str:
    """Clean and normalize text for TTS."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove common web artifacts
    patterns_to_remove = [
        r'Share this:.*?(?=\n|$)',
        r'Click to share.*?(?=\n|$)',
        r'Subscribe to.*?(?=\n|$)',
        r'Follow us on.*?(?=\n|$)',
        r'Advertisement',
        r'Sponsored',
        r'Read more:.*?(?=\n|$)',
        r'Related posts.*',
        r'Like this:.*?(?=\n|$)',
        r'Loading\.\.\..*?(?=\n|$)',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Expand common abbreviations for better TTS
    abbreviations = {
        r'\bDr\.': 'Doctor',
        r'\bMr\.': 'Mister',
        r'\bMrs\.': 'Missus',
        r'\bMs\.': 'Miss',
        r'\betc\.': 'etcetera',
        r'\be\.g\.': 'for example',
        r'\bi\.e\.': 'that is',
        r'\bvs\.': 'versus',
    }
    
    for abbr, expansion in abbreviations.items():
        text = re.sub(abbr, expansion, text)
    
    # Clean up final result
    text = text.strip()
    
    return text


def save_content_to_file(content: dict, output_path: Path) -> Path:
    """Save fetched content to a text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# {content['title']}\n\n")
        f.write(f"Source: {content['source']}\n")
        f.write(f"URL: {content['url']}\n\n")
        f.write("---\n\n")
        f.write(content['content'])
    
    console.print(f"[green]✓[/green] Saved to: {output_path}")
    
    return output_path

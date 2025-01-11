import argparse
from pathlib import Path
import os
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import anthropic
from exa_py import Exa

@dataclass
class Term:
    """A term identified for linking with its explanation"""
    term: str
    reason: str

@dataclass
class Link:
    """A link found for a term"""
    term: str
    url: str
    title: str

def chunk_text(text: str, max_chunk_size: int = 2000) -> List[str]:
    """Split text into chunks of roughly equal size at paragraph boundaries"""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        if current_size + para_size > max_chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks

def parse_claude_response(response: str) -> List[Term]:
    """Parse Claude's response to extract terms and reasons"""
    terms = []
    current_term = None
    current_reason = None
    
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("TERM: "):
            # Save previous term if exists
            if current_term and current_reason:
                terms.append(Term(current_term, current_reason))
            current_term = line[6:].strip()
            current_reason = None
        elif line.startswith("REASON: "):
            current_reason = line[8:].strip()
            
    # Add final term
    if current_term and current_reason:
        terms.append(Term(current_term, current_reason))
        
    return terms

def find_links_for_terms(exa: Exa, terms: List[Term]) -> Dict[str, Link]:
    """Find best link for each term using Exa search"""
    links = {}
    
    for term in terms:
        # Construct a search query that looks for authoritative sources
        # query = f"The best explanation or overview of {term.term} is (site: wikipedia.org OR site: .edu OR site: .gov):"
        
        try:
            # Search with Exa
            results = exa.search(term.term, num_results=1, type="auto")
            if results.results:
                result = results.results[0]
                links[term.term] = Link(
                    term=term.term,
                    url=result.url,
                    title=result.title
                )
        except Exception as e:
            print(f"Error finding link for {term.term}: {e}")
            continue
            
    return links

def add_links_to_text(text: str, links: Dict[str, Link]) -> str:
    """Add markdown links to text for all terms we have links for"""
    # Sort terms by length (descending) to handle overlapping terms correctly
    terms = sorted(links.keys(), key=len, reverse=True)
    
    # Create regex pattern that matches whole words only
    patterns = [re.compile(fr'\b{re.escape(term)}\b') for term in terms]
    
    # Track which terms we've linked to avoid duplicate links
    linked_terms = set()
    
    # Process each term
    result = text
    for term, pattern in zip(terms, patterns):
        if term in linked_terms:
            continue
            
        # Only replace first occurrence
        link = links[term]
        replacement = f"[{term}]({link.url})"
        result = pattern.sub(replacement, result, count=1)
        linked_terms.add(term)
        
    return result

def process_transcript(
    transcript_path: Path,
    claude_client: anthropic.Client,
    exa_client: Exa,
    prompt_template: str
) -> str:
    """Process a transcript file to add reference links"""
    # Read transcript
    text = transcript_path.read_text()
    
    # Split into chunks
    chunks = chunk_text(text)
    
    # Process each chunk
    all_terms = []
    for chunk in chunks:
        # Get Claude's suggestions
        prompt = prompt_template + "\n\n" + chunk
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are a helpful AI assistant.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response
        terms = parse_claude_response(response.content[0].text)
        all_terms.extend(terms)
        
    # Find links for all terms
    links = find_links_for_terms(exa_client, all_terms)
    
    # Add links to text
    linked_text = add_links_to_text(text, links)
    
    return linked_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "transcript", 
        nargs="?",  # Make the argument optional
        default="output/transcripts/transcript.md",
        help="Path to transcript file (default: output/transcripts/transcript.md)"
    )
    parser.add_argument("--output", help="Output file path (default: input path with -linked suffix)")
    args = parser.parse_args()
    
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        
    # Set up output path
    if args.output:
        output_path = Path(args.output)
    else:
        stem = transcript_path.stem
        output_path = transcript_path.parent / f"{stem}-linked{transcript_path.suffix}"
        
    # Read prompt template
    prompt_path = Path("prompts/find_links.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt_template = prompt_path.read_text()
    
    # Initialize clients
    claude_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))
    
    try:
        # Process transcript
        linked_text = process_transcript(
            transcript_path,
            claude_client,
            exa_client,
            prompt_template
        )
        
        # Save output
        output_path.write_text(linked_text)
        print(f"Processed transcript saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing transcript: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    main() 
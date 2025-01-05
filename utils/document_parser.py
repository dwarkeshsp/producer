from pathlib import Path
import tempfile
import os
from ebooklib import epub
from bs4 import BeautifulSoup

class DocumentParser:
    """Simple EPUB document parser that extracts chapters and their content."""
    
    def __init__(self):
        self._temp_file = None
        self._book = None
        self._chapters = []
    
    def load_document(self, file_data, filename=None) -> list[str]:
        """Load an EPUB document and extract chapter titles.
        
        Args:
            file_data: File data from Gradio (FileData object with read() method)
            filename: Optional filename (not used)
        """
        # Clean up any previous temp file
        self.cleanup()
        
        # Get the raw bytes from the Gradio file data
        content = file_data.read() if hasattr(file_data, 'read') else file_data
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.epub') as temp:
            temp.write(content)
            self._temp_file = temp.name
        
        # Read the EPUB
        try:
            self._book = epub.read_epub(self._temp_file)
            print("DEBUG: Successfully read EPUB file")
        except Exception as e:
            print(f"DEBUG: Error reading EPUB: {str(e)}")
            raise ValueError(f"Failed to read EPUB: {str(e)}")
        
        # Extract chapters
        self._chapters = self._extract_chapters()
        print(f"DEBUG: Extracted {len(self._chapters)} chapters")
        
        # Return chapter titles
        return [chapter['title'] for chapter in self._chapters]
    
    def get_chapter_content(self, chapter_idx: int) -> str:
        """Get the content of a specific chapter."""
        if not self._book or not self._chapters:
            raise ValueError("No document loaded")
        
        if not 0 <= chapter_idx < len(self._chapters):
            raise ValueError(f"Invalid chapter index: {chapter_idx}")
        
        chapter = self._chapters[chapter_idx]
        self._current_chapter_title = chapter['title'].strip()  # Store for _get_chapter_text
        
        print(f"DEBUG: Getting content for chapter: {self._current_chapter_title}")
        content = self._get_chapter_text(chapter['item'])
        print(f"DEBUG: Extracted {len(content)} characters of content")
        
        return content
    
    def _extract_chapters(self) -> list[dict]:
        """Extract chapters from the EPUB file."""
        chapters = []
        
        # First try to get chapters from the table of contents
        print("DEBUG: Checking table of contents...")
        if hasattr(self._book, 'toc'):
            # Debug the TOC structure
            print("DEBUG: TOC structure:")
            for item in self._book.toc:
                print(f"DEBUG: TOC item type: {type(item)}")
                if isinstance(item, tuple):
                    print(f"DEBUG: Tuple length: {len(item)}")
                    if len(item) > 1:
                        print(f"DEBUG: Second item type: {type(item[1])}")
                        if isinstance(item[1], (list, tuple)):
                            print(f"DEBUG: Sub-items count: {len(item[1])}")
            
            def process_toc_entries(entries, level=0):
                for item in entries:
                    # Handle both Link objects and tuples
                    if hasattr(item, 'title') and hasattr(item, 'href'):
                        # Direct Link object
                        doc = self._book.get_item_with_href(item.href)
                        if doc:
                            prefix = "  " * level if level > 0 else ""
                            chapters.append({
                                'title': prefix + item.title,
                                'item': doc
                            })
                    elif isinstance(item, tuple):
                        section = item[0]
                        # Process the section
                        if hasattr(section, 'title') and hasattr(section, 'href'):
                            doc = self._book.get_item_with_href(section.href)
                            if doc:
                                prefix = "  " * level if level > 0 else ""
                                chapters.append({
                                    'title': prefix + section.title,
                                    'item': doc
                                })
                        
                        # Process sub-items if they exist
                        if len(item) > 1:
                            if isinstance(item[1], (list, tuple)):
                                process_toc_entries(item[1], level + 1)
                            elif hasattr(item[1], 'title'):  # Single sub-item
                                process_toc_entries([item[1]], level + 1)
            
            process_toc_entries(self._book.toc)
            print(f"DEBUG: Found {len(chapters)} chapters in TOC")
            print("DEBUG: Chapter titles found:")
            for ch in chapters:
                print(f"  - {ch['title']}")
        
        # If no chapters found in TOC, scan the documents
        if not chapters:
            print("DEBUG: No chapters in TOC, scanning documents...")
            # Get all HTML documents
            docs = [item for item in self._book.get_items() 
                   if item.get_type() == epub.ITEM_DOCUMENT]
            
            print(f"DEBUG: Found {len(docs)} documents to scan")
            
            for doc in docs:
                soup = BeautifulSoup(doc.get_content(), 'html.parser')
                
                # Look for chapter headings
                headings = (
                    soup.find_all(['h1', 'h2']) + 
                    soup.find_all(class_=lambda x: x and ('chapter' in x.lower() or 'title' in x.lower()))
                )
                
                for heading in headings:
                    # Clean up the text
                    title = ' '.join(heading.get_text().split())
                    if title:  # Only add if we have a title
                        chapters.append({
                            'title': title,
                            'item': doc
                        })
        
        if not chapters:
            print("DEBUG: No chapters found, using documents as chapters")
            # If still no chapters found, treat each document as a chapter
            for doc in self._book.get_items():
                if doc.get_type() == epub.ITEM_DOCUMENT:
                    chapters.append({
                        'title': f"Chapter {len(chapters) + 1}",
                        'item': doc
                    })
        
        return chapters
    
    def _get_chapter_text(self, item) -> str:
        """Extract text content from a chapter."""
        try:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()
            
            # Get main content area (usually in body or main tags)
            content_area = soup.find('body') or soup.find('main') or soup
            
            # Get all text blocks, excluding navigation elements
            text_blocks = []
            for element in content_area.find_all(text=True, recursive=True):
                if (element.parent.name not in ['script', 'style', 'nav', 'header'] and
                    element.strip()):
                    text_blocks.append(element.strip())
            
            return '\n\n'.join(text_blocks)
            
        except Exception as e:
            print(f"DEBUG: Error extracting text: {str(e)}")
            # Fallback to simple text extraction
            return soup.get_text(separator='\n\n', strip=True)
    
    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_file and os.path.exists(self._temp_file):
            os.unlink(self._temp_file)
            self._temp_file = None
        self._book = None
        self._chapters = [] 
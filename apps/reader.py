import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import gradio as gr
import asyncio
import os
import json
import requests
from anthropic import Anthropic
from utils.document_parser import DocumentParser
from dotenv import load_dotenv

# Load environment variables
env_path = Path(project_root) / ".env"
load_dotenv(env_path)

# Mochi deck IDs
DECK_CATEGORIES = {
    "CS/Hardware": "rhGqR9SK",
    "Math/Physics": "Dm5vczZg",
    "AI": "SS9QEfiy",
    "History/Military": "3nJYp7Zh",
    "Quotes/Random": "rWUzSu8t",
    "Bio": "BspzxaUJ",
    "Econ/Finance": "mvvJ27Q1"
}

class CardGenerator:
    """Handles card generation and Mochi integration."""
    
    def __init__(self):
        self.parser = DocumentParser()
        self.claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.mochi_key = os.getenv("MOCHI_API_KEY")
        
        # Load prompts
        self.prompts = {
            key: Path(f"prompts/{key}.txt").read_text()
            for key in ["card_generation", "commentary"]
        }
        
        # State
        self.current_cards = []
        self.current_index = 0
        self.approved_cards = []
    
    def get_chapter_list(self, file_data) -> list[str]:
        """Get list of chapters from document.
        
        Args:
            file_data: File data from Gradio
        """
        try:
            if not file_data:
                return []
            
            # Attempt to extract filename from file_data
            filename = getattr(file_data, 'name', None)
            if not filename:
                filename = "uploaded_file"
                print("DEBUG: No filename attribute found, using default.")
            else:
                print(f"DEBUG: Filename extracted: {filename}")
            
            # Check file extension
            file_ext = Path(filename).suffix.lower()
            if not file_ext:
                print("DEBUG: No file extension found, checking content type.")
                # Attempt to determine file type from content
                if file_data.startswith(b'%PDF-'):
                    file_ext = '.pdf'
                elif file_data.startswith(b'PK'):
                    file_ext = '.epub'
                else:
                    raise ValueError("Unsupported file type")
            print(f"DEBUG: File extension: {file_ext}")
            
            return self.parser.load_document(file_data, filename)
        except Exception as e:
            return [f"Error: {str(e)}"]
    
    async def process_chapter(self, file_data, chapter_idx: int) -> tuple:
        """Process chapter and generate cards + commentary.
        
        Args:
            file_data: File data from Gradio
            chapter_idx: Index of chapter to process
        """
        try:
            if not file_data:
                return None, "No file provided"
                
            # Get chapter content
            content = self.parser.get_chapter_content(chapter_idx)
            
            # Generate cards and commentary
            cards, commentary = await asyncio.gather(
                self._generate_cards(content),
                self._generate_commentary(content)
            )
            
            # Parse and store cards
            self.current_cards = json.loads(cards)
            self.current_index = 0
            self.approved_cards = []
            
            # Return first card and commentary
            return self._get_current_card(), commentary
            
        except Exception as e:
            return None, f"Error: {str(e)}"
        finally:
            self.parser.cleanup()
    
    async def _generate_cards(self, content: str) -> str:
        """Generate flashcards using Claude."""
        response = await self.claude.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            system=self.prompts["card_generation"],
            messages=[{"role": "user", "content": content}]
        )
        return response.content[0].text
    
    async def _generate_commentary(self, content: str) -> str:
        """Generate commentary using Claude."""
        response = await self.claude.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            system=self.prompts["commentary"],
            messages=[{"role": "user", "content": content}]
        )
        return response.content[0].text
    
    def _get_current_card(self) -> dict:
        """Get current card with UI state."""
        if not self.current_cards or self.current_index >= len(self.current_cards):
            return {
                'front': "",
                'back': "",
                'category': "",
                'status': "No more cards to review",
                'show_buttons': False,
                'show_upload': True
            }
        
        card = self.current_cards[self.current_index]
        return {
            'front': card['front'],
            'back': card['back'],
            'category': card['category'],
            'status': f"Card {self.current_index + 1} of {len(self.current_cards)}",
            'show_buttons': True,
            'show_upload': False
        }
    
    def accept_card(self, front: str, back: str, category: str) -> dict:
        """Accept current card and move to next."""
        if self.current_index < len(self.current_cards):
            self.approved_cards.append({
                'front': front,
                'back': back,
                'category': category
            })
        
        self.current_index += 1
        return self._get_current_card()
    
    def reject_card(self) -> dict:
        """Reject current card and move to next."""
        if self.current_index < len(self.current_cards):
            self.current_cards.pop(self.current_index)
        return self._get_current_card()
    
    def upload_to_mochi(self) -> str:
        """Upload approved cards to Mochi."""
        if not self.approved_cards:
            return "No cards to upload!"
        
        results = []
        for card in self.approved_cards:
            try:
                # Format card for Mochi
                mochi_card = {
                    "deck-id": DECK_CATEGORIES[card["category"]],
                    "fields": {
                        "name": {"id": "name", "value": card["front"]},
                        "back": {"id": "back", "value": card["back"]}
                    }
                }
                
                # Upload to Mochi
                response = requests.post(
                    "https://app.mochi.cards/api/cards",
                    json=mochi_card,
                    auth=(self.mochi_key, "")
                )
                
                if response.status_code != 200:
                    results.append(f"Error: {response.text}")
                
            except Exception as e:
                results.append(f"Error: {str(e)}")
        
        # Clear approved cards
        success_count = len(self.approved_cards) - len(results)
        self.approved_cards = []
        
        if results:
            return f"Uploaded {success_count} cards with {len(results)} errors:\n" + "\n".join(results)
        return f"Successfully uploaded {success_count} cards to Mochi!"

def create_interface():
    """Create the Gradio interface."""
    generator = CardGenerator()
    
    with gr.Blocks(title="Document Reader & Card Generator") as app:
        # Document upload and chapter selection
        with gr.Row():
            file_input = gr.File(
                label="Upload EPUB Document",
                type="binary",
                file_types=[".epub"]
            )
        
        chapter_select = gr.Dropdown(
            label="Select Chapter",
            choices=[],
            interactive=True,
            visible=False
        )
        
        def update_chapters(file):
            if not file:
                return gr.update(choices=[], visible=False)
            chapters = generator.get_chapter_list(file)
            return gr.update(choices=chapters, visible=True, value=chapters[0] if chapters else None)
        
        file_input.change(
            fn=update_chapters,
            inputs=[file_input],
            outputs=[chapter_select]
        )
        
        process_btn = gr.Button("Process Chapter")
        
        # Commentary section
        commentary = gr.Textbox(
            label="Commentary",
            lines=10,
            interactive=False
        )
        
        # Card review section
        gr.Markdown("## Review Cards")
        
        with gr.Row():
            card_front = gr.Textbox(
                label="Front",
                lines=3,
                interactive=True
            )
            card_back = gr.Textbox(
                label="Back",
                lines=3,
                interactive=True
            )
        
        with gr.Row():
            deck_category = gr.Dropdown(
                choices=list(DECK_CATEGORIES.keys()),
                label="Deck Category",
                value="AI"
            )
            card_status = gr.Textbox(
                label="Status",
                interactive=False
            )
        
        with gr.Row():
            accept_btn = gr.Button("Accept & Next", visible=False)
            reject_btn = gr.Button("Reject & Next", visible=False)
            upload_btn = gr.Button("Upload to Mochi", visible=False)
        
        upload_status = gr.Textbox(
            label="Upload Status",
            interactive=False
        )
        
        # Event handlers
        async def process_chapter(file, chapter_idx):
            card, comment = await generator.process_chapter(file, chapter_idx)
            if not card:  # Error occurred
                return [
                    "", "", comment, gr.update(visible=False),
                    gr.update(visible=False), "", gr.update(visible=False)
                ]
            
            return [
                card['front'],
                card['back'],
                comment,
                gr.update(visible=card['show_buttons']),
                gr.update(visible=card['show_buttons']),
                card['status'],
                gr.update(visible=card['show_upload'])
            ]
        
        def handle_card_action(action, front, back, category):
            card = (generator.accept_card(front, back, category) 
                   if action == 'accept' else 
                   generator.reject_card())
            
            return [
                card['front'],
                card['back'],
                card['status'],
                gr.update(visible=card['show_buttons']),
                gr.update(visible=card['show_buttons']),
                card['category'],
                gr.update(visible=card['show_upload'])
            ]
        
        # Connect events
        process_btn.click(
            fn=process_chapter,
            inputs=[file_input, chapter_select],
            outputs=[
                card_front, card_back, commentary,
                accept_btn, reject_btn, card_status, upload_btn
            ]
        )
        
        accept_btn.click(
            fn=lambda f, b, c: handle_card_action('accept', f, b, c),
            inputs=[card_front, card_back, deck_category],
            outputs=[
                card_front, card_back, card_status,
                accept_btn, reject_btn, deck_category, upload_btn
            ]
        )
        
        reject_btn.click(
            fn=lambda: handle_card_action('reject', None, None, None),
            outputs=[
                card_front, card_back, card_status,
                accept_btn, reject_btn, deck_category, upload_btn
            ]
        )
        
        upload_btn.click(
            fn=generator.upload_to_mochi,
            outputs=[upload_status]
        )
    
    return app

if __name__ == "__main__":
    create_interface().launch() 
"""
Vocabulary class for TinyLLM.
Defines the closed vocabulary and implements tokenization.
"""

import pickle
from typing import List, Dict


class Vocabulary:
    """
    Manages the closed vocabulary for TinyLLM.
    Maps tokens to integer IDs and vice versa.
    """
    
    # Core conversational and arithmetic words
    WORDS = [
        'hey', 'hello', 'i', 'am', 'a', 'tinyllm', 'what', 'is', 'your', 'name',
        'my', 'the', 'answer', 'compute', 'solve', 'please', 'tell', 'me'
    ]
    
    # Utility words to handle richer prompts (math synonyms + simple tasks)
    UTILITY_WORDS = [
        'add', 'subtract', 'multiply', 'divide', 'plus', 'minus', 'times', 'over', 'by',
        'and', 'from', 'todo', 'task', 'list', 'create', 'remove', 'complete', 'set',
        'reminder', 'for', 'on', 'at', 'in', 'today', 'tomorrow', 'date', 'time', 'now',
        'status', 'updated', 'added', 'removed', 'done', 'ok', 'sure',
        'email', 'report', 'code', 'read', 'milk', 'water', 'meeting', 'call', 'note'
    ]
    
    # Extra factual/phrase tokens to reduce UNKs for lightweight facts
    KNOWLEDGE_WORDS = [
        'programming', 'language', 'attention', 'based', 'model', 'chat', 'math',
        'who', 'are', 'can', 'do', 'made', 'help', 'demo'
    ]
    
    # Common/general-purpose words to let the model form richer sentences
    COMMON_WORDS = [
        'this', 'that', 'these', 'those', 'because', 'example', 'simple', 'text', 'answer',
        'question', 'explain', 'how', 'why', 'when', 'where', 'does', 'work', 'use',
        'provide', 'response', 'two', 'lines', 'line', 'first', 'second', 'also', 'more',
        'detail', 'details', 'knowledge', 'info', 'information', 'number', 'numbers',
        'integer', 'result', 'sum', 'product', 'difference', 'quotient', 'greeting',
        'thanks', 'thank', 'you', 'welcome', 'good', 'bad', 'fine', 'great', 'awesome',
        'nice', 'cool', 'okay', 'yes', 'no', 'maybe', 'later', 'morning', 'afternoon',
        'night', 'day', 'week', 'month', 'year', 'today', 'tomorrow', 'yesterday',
        'time', 'date', 'now', 'soon', 'later', 'small', 'large', 'bigger', 'powerful',
        'strong', 'capable', 'knowledgeable', 'sentence', 'sentences', 'word', 'words',
        'useful', 'helpful', 'short', 'long', 'multi', 'line', 'lines', 'double', 'break',
        'newline', 'two', 'up', 'down', 'left', 'right', 'forward', 'backward',
        'start', 'end', 'begin', 'finish', 'complete', 'summary', 'story', 'idea', 'plan',
        'goal', 'ready', 'set', 'go', 'please', 'sure', 'ok'
    ]
    
    # Daily conversation and Q&A words
    DAILY_WORDS = [
        'weather', 'rain', 'sunny', 'cloudy', 'hot', 'cold', 'warm', 'cool', 'temperature',
        'food', 'eat', 'breakfast', 'lunch', 'dinner', 'hungry', 'thirsty', 'drink',
        'sleep', 'tired', 'awake', 'wake', 'bed', 'rest', 'energy', 'feel', 'feeling',
        'happy', 'sad', 'excited', 'bored', 'interested', 'fun', 'enjoy', 'like', 'love',
        'hate', 'dislike', 'prefer', 'favorite', 'best', 'worst', 'better', 'worse',
        'color', 'red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple',
        'animal', 'dog', 'cat', 'bird', 'fish', 'pet', 'friend', 'friends', 'family',
        'home', 'house', 'room', 'door', 'window', 'chair', 'table', 'desk', 'computer',
        'phone', 'book', 'read', 'write', 'study', 'learn', 'teach', 'school', 'class',
        'work', 'job', 'office', 'meeting', 'project', 'task', 'done', 'finished',
        'sport', 'game', 'play', 'watch', 'movie', 'music', 'song', 'dance', 'sing',
        'travel', 'trip', 'vacation', 'visit', 'place', 'city', 'country', 'world',
        'money', 'buy', 'sell', 'price', 'cost', 'cheap', 'expensive', 'free', 'pay',
        'health', 'sick', 'well', 'doctor', 'medicine', 'exercise', 'run', 'walk',
        'shopping', 'store', 'buy', 'shop', 'clothes', 'shoes', 'shirt', 'pants'
    ]
    
    # Numeric coverage expanded to multi-digit math (0-199)
    NUMBERS = [str(i) for i in range(0, 200)]
    
    OPERATORS = ['+', '-', '*', '/', '=']
    
    # Special newline token to allow multi-line outputs
    NEWLINE = '\n'
    
    # Special tokens
    PAD = '[PAD]'
    UNK = '[UNK]'
    BOS = '[BOS]'
    EOS = '[EOS]'
    
    def __init__(self):
        """Initialize vocabulary with all tokens."""
        # Build complete token list: special tokens first, then unique lexical tokens
        lexical_tokens = (
            self.WORDS +
            self.UTILITY_WORDS +
            self.KNOWLEDGE_WORDS +
            self.COMMON_WORDS +
            self.DAILY_WORDS +
            self.NUMBERS +
            self.OPERATORS +
            [self.NEWLINE]
        )
        tokens: List[str] = []
        for token in [self.PAD, self.UNK, self.BOS, self.EOS] + lexical_tokens:
            if token not in tokens:
                tokens.append(token)
        self.tokens = tokens
        
        # Create token to ID mapping
        self.token_to_id: Dict[str, int] = {token: idx for idx, token in enumerate(self.tokens)}
        
        # Create ID to token mapping
        self.id_to_token: Dict[int, str] = {idx: token for token, idx in self.token_to_id.items()}
        
        # Store vocabulary size
        self.size = len(self.tokens)
        
        # Get special token IDs for easy access
        self.pad_id = self.token_to_id[self.PAD]
        self.unk_id = self.token_to_id[self.UNK]
        self.bos_id = self.token_to_id[self.BOS]
        self.eos_id = self.token_to_id[self.EOS]
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode a text string into a list of token IDs.
        
        Args:
            text: Input text string
            add_bos: Whether to add [BOS] token at the beginning
            add_eos: Whether to add [EOS] token at the end
        
        Returns:
            List of integer token IDs
        """
        # Split text by whitespace for word-level tokenization
        words = text.strip().lower().split()
        
        token_ids = []
        
        # Add BOS token if requested
        if add_bos:
            token_ids.append(self.bos_id)
        
        # Convert each word/token to ID
        for word in words:
            # Check if word is in vocabulary
            if word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
            else:
                # If not found, use UNK token
                token_ids.append(self.unk_id)
        
        # Add EOS token if requested
        if add_eos:
            token_ids.append(self.eos_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode a list of token IDs back into a text string.
        
        Args:
            token_ids: List of integer token IDs
            skip_special: Whether to skip special tokens (PAD, BOS, EOS, UNK) in output
        
        Returns:
            Decoded text string
        """
        tokens = []
        
        for token_id in token_ids:
            # Get token from ID
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special:
                    if token in [self.PAD, self.BOS, self.EOS, self.UNK]:
                        continue
                
                # Handle newline token specially
                if token == self.NEWLINE:
                    tokens.append('\n')
                else:
                    tokens.append(token)
        
        # Join tokens with spaces, but preserve newlines
        result = []
        for i, token in enumerate(tokens):
            if token == '\n':
                result.append('\n')
            else:
                # Add space before token if not first and previous wasn't newline
                if i > 0 and result and result[-1] != '\n':
                    result.append(' ')
                result.append(token)
        
        return ''.join(result)
    
    def save(self, filepath: str):
        """Save vocabulary to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from a pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    # Test vocabulary
    vocab = Vocabulary()
    print(f"Vocabulary size: {vocab.size}")
    print(f"Sample tokens: {vocab.tokens[:10]}")
    
    # Test encoding
    text = "hello what is your name"
    encoded = vocab.encode(text)
    print(f"Encoded '{text}': {encoded}")
    
    # Test decoding
    decoded = vocab.decode(encoded)
    print(f"Decoded: '{decoded}'")


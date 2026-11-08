"""
Synthetic data generator for TinyLLM.
Generates training examples using only the closed vocabulary.
"""

import random
from typing import Tuple, List


class SyntheticDataGenerator:
    """
    Generates synthetic training data for TinyLLM.
    Creates dialogues, arithmetic problems, and lightweight utility intents.
    """
    
    def __init__(self, vocab, max_number: int = 199):
        """
        Initialize the data generator.
        
        Args:
            vocab: Vocabulary instance to ensure only valid tokens are used
            max_number: Maximum integer to sample for arithmetic problems
        """
        self.vocab = vocab
        self.max_number = max_number
        
        # Extract valid tokens
        self.operators = ['+', '-', '*', '/']  # Exclude '=' from prompts
        self.number_tokens = [int(tok) for tok in vocab.NUMBERS]
        self.safe_max_answer = max(self.number_tokens)
        
        # Define greeting patterns
        self.greeting_inputs = [
            "hey",
            "hello",
            "hey hello"
        ]
        
        self.greeting_responses = [
            "hello",
            "hey",
            "hello i am tinyllm",
            "hello",
            "hey"  # More weight on simple responses
        ]
        
        # Define name query patterns
        self.name_inputs = [
            "what is your name",
            "tell me your name",
            "what is your name please",
            "please tell me your name"
        ]
        
        self.name_responses = [
            "i am tinyllm",
            "my name is tinyllm",
            "i am a tinyllm",
            "tinyllm"
        ]
        
        # Define general Q&A patterns (non-mathematical)
        self.qa_patterns = [
            ("what is the answer", "the answer"),
            ("tell me the answer", "the answer"),
            ("what is the answer please", "the answer"),
            ("please tell me the answer", "the answer"),
            ("compute the answer", "the answer"),
            ("solve the answer", "the answer"),
        ]
        
        # Simple factual snippets to stretch beyond arithmetic
        self.fact_patterns = [
            ("what is python", "python is a programming language"),
            ("what is a transformer", "a transformer is an attention based model"),
            ("who are you", "i am tinyllm"),
            ("what can you do", "i can chat and solve math and tasks"),
            ("who made you", "i am a tiny demo model"),
            ("what is machine learning", "machine learning uses data to learn patterns"),
            ("what is attention", "attention lets models focus on important tokens"),
            ("what is code", "code is a set of instructions for computers"),
            ("what is data", "data is information that models can learn from"),
            ("what is a model", "a model maps inputs to outputs using learned weights"),
        ]
        
        # Daily conversation Q&A patterns
        self.daily_qa_patterns = [
            ("how are you", "i am fine thanks for asking"),
            ("how is the weather", "the weather is nice today"),
            ("what time is it", "it is time to check your clock"),
            ("what day is it", "today is a good day"),
            ("are you hungry", "i do not eat but thanks for asking"),
            ("do you like music", "yes i like music it is fun"),
            ("what is your favorite color", "i like blue and green colors"),
            ("do you have friends", "yes you are my friend"),
            ("what do you do", "i chat and help with math and questions"),
            ("can you help me", "yes i can help what do you need"),
            ("what should i eat", "eat healthy food like fruits and vegetables"),
            ("where do you live", "i live in the computer"),
            ("do you sleep", "i do not sleep i am always ready"),
            ("what is fun", "learning and playing games is fun"),
            ("do you like books", "yes books have lots of knowledge"),
            ("what is good", "helping others and learning is good"),
            ("how do you feel", "i feel great and ready to help"),
            ("what is your hobby", "my hobby is chatting and solving math"),
            ("do you exercise", "i do not exercise but you should"),
            ("what makes you happy", "helping you makes me happy"),
            ("tell me a joke", "why did the math book look sad because it had too many problems"),
            ("what is the best food", "healthy food is the best food"),
            ("do you like animals", "yes animals are cute and fun"),
            ("what is your dream", "my dream is to help people learn"),
            ("are you smart", "i try to be helpful and knowledgeable"),
            ("what is night for", "night is time to sleep"),
            ("what is your favorite movie", "my favorite movie is tinyllm"),
        ]
        
        # More complex math patterns (larger numbers, multi-step hints)
        self.advanced_math_patterns = [
            ("calculate", "the answer is"),
            ("solve this", "the answer is"),
            ("find the result", "the answer is"),
            ("what does this equal", "the answer is"),
            ("give me the answer", "the answer is"),
        ]
        
        # Task-style intents (toy examples, deterministic responses)
        self.todo_items = ["email", "report", "code", "read", "call", "meeting", "note", "milk"]
        
        # Multi-line response bank (two short lines)
        self.multiline_responses = [
            ("here is a short note", "second line adds more info"),
            ("first line gives the answer", "second line gives a hint"),
            ("summary line one", "details line two"),
            ("hello this is line one", "and this is line two"),
            ("tinyllm can answer", "and also show two lines"),
        ]
    
    def _safe_sample_number(self, upper: int = None) -> int:
        """Sample a number that is guaranteed to be in-vocabulary and <= upper."""
        if upper is None:
            upper = self.max_number
        upper = min(upper, self.safe_max_answer)
        return random.randint(0, upper)
    
    def generate_arithmetic_problem(self) -> Tuple[str, str]:
        """
        Generate a multi-digit arithmetic problem with an in-vocab answer.
        
        Returns:
            Tuple of (prompt_string, answer_string)
        """
        op = random.choice(self.operators)
        
        # Sample operands with op-specific limits to keep answer in-vocab
        if op == '+':
            a = self._safe_sample_number(self.max_number)
            b = self._safe_sample_number(self.max_number)
            answer = a + b
            while answer > self.safe_max_answer:
                a = self._safe_sample_number(self.max_number // 2)
                b = self._safe_sample_number(self.max_number // 2)
                answer = a + b
        elif op == '-':
            a = self._safe_sample_number(self.max_number)
            b = self._safe_sample_number(a)  # ensure non-negative
            answer = a - b
        elif op == '*':
            # Keep products small enough for the vocab range
            a = self._safe_sample_number(20)
            b = self._safe_sample_number(20)
            answer = a * b
            while answer > self.safe_max_answer:
                a = self._safe_sample_number(15)
                b = self._safe_sample_number(15)
                answer = a * b
        else:  # division
            # Generate integer division results
            target = self._safe_sample_number(min(self.max_number // 2, 50))
            b = random.randint(1, 9)
            a = target * b
            answer = target
            if a > self.safe_max_answer:
                a = self._safe_sample_number(90)
                factors = [i for i in range(1, a + 1) if a % i == 0]
                b = random.choice(factors) if factors else 1
                answer = a // b
        
        prompt = f"what is {a} {op} {b}"
        # Format answer as "the answer is [number]" to fix output formatting
        answer_str = f"the answer is {answer}"
        
        return prompt, answer_str
    
    def generate_greeting_example(self) -> Tuple[str, str]:
        """
        Generate a greeting dialogue example.
        
        Returns:
            Tuple of (input_string, response_string)
        """
        input_text = random.choice(self.greeting_inputs)
        response = random.choice(self.greeting_responses)
        return input_text, response
    
    def generate_name_example(self) -> Tuple[str, str]:
        """
        Generate a name query example.
        
        Returns:
            Tuple of (input_string, response_string)
        """
        input_text = random.choice(self.name_inputs)
        response = random.choice(self.name_responses)
        return input_text, response
    
    def generate_intro_example(self) -> Tuple[str, str]:
        """
        Generate an introductory example combining greetings and name.
        
        Returns:
            Tuple of (input_string, response_string)
        """
        # Combine greeting and name query
        greeting = random.choice(["hey", "hello"])
        name_query = random.choice(["what is your name", "tell me your name"])
        input_text = f"{greeting} {name_query}"
        response = "hello i am tinyllm"
        return input_text, response
    
    def generate_arithmetic_query_example(self) -> Tuple[str, str]:
        """
        Generate an arithmetic query with different phrasings.
        Always formats answer as "the answer is [number]"
        
        Returns:
            Tuple of (input_string, answer_string)
        """
        # Generate the arithmetic problem (already formatted)
        prompt, answer = self.generate_arithmetic_problem()
        
        # Sometimes add "please" or "tell me" or use advanced patterns
        rand = random.random()
        if rand < 0.2:
            prompt = f"please {prompt}"
        elif rand < 0.4:
            prompt = f"tell me {prompt}"
        elif rand < 0.6:
            # Use advanced math phrasing
            prefix = random.choice(["calculate", "solve this", "find the result"])
            # Extract just the math part (e.g., "42 + 17")
            math_part = prompt.replace("what is ", "").replace("please ", "").replace("tell me ", "")
            prompt = f"{prefix} {math_part}"
        
        return prompt, answer
    
    def generate_worded_math_example(self) -> Tuple[str, str]:
        """
        Generate worded arithmetic like 'add 12 and 7' or 'subtract 3 from 18'.
        """
        op = random.choice(['add', 'subtract', 'multiply', 'divide', 'plus', 'minus', 'times'])
        a = self._safe_sample_number(self.max_number)
        b = self._safe_sample_number(self.max_number)
        
        if op in ['add', 'plus']:
            answer = a + b
        elif op in ['subtract', 'minus']:
            if b > a:
                a, b = b, a
            answer = a - b
        elif op in ['multiply', 'times']:
            a = self._safe_sample_number(20)
            b = self._safe_sample_number(20)
            answer = a * b
            while answer > self.safe_max_answer:
                a = self._safe_sample_number(12)
                b = self._safe_sample_number(12)
                answer = a * b
        else:  # divide
            answer = self._safe_sample_number(min(self.max_number // 2, 50))
            b = random.randint(1, 9)
            a = answer * b
        
        # Build prompt variants
        variant = random.random()
        if op in ['add', 'plus']:
            prompt = f"{op} {a} and {b}"
        elif op in ['subtract', 'minus']:
            if variant < 0.5:
                prompt = f"{op} {b} from {a}"
            else:
                prompt = f"{a} {op} {b}"
        elif op in ['multiply', 'times']:
            prompt = f"{op} {a} by {b}"
        else:
            prompt = f"divide {a} by {b}"
        
        # Format answer as "the answer is [number]"
        return prompt, f"the answer is {answer}"
    
    def generate_arithmetic_compute_example(self) -> Tuple[str, str]:
        """
        Generate an arithmetic problem using "compute" or "solve".
        Answer is already formatted as "the answer is [number]"
        
        Returns:
            Tuple of (input_string, answer_string)
        """
        prompt, answer = self.generate_arithmetic_problem()
        
        # Replace "what is" with "compute" or "solve"
        if random.random() < 0.5:
            prompt = prompt.replace("what is", "compute")
        else:
            prompt = prompt.replace("what is", "solve")
        
        return prompt, answer
    
    def generate_qa_example(self) -> Tuple[str, str]:
        """
        Generate a general Q&A example (non-mathematical).
        
        Returns:
            Tuple of (input_string, response_string)
        """
        question, response = random.choice(self.qa_patterns)
        return question, response
    
    def generate_fact_example(self) -> Tuple[str, str]:
        """
        Generate a small factual response to broaden coverage.
        """
        return random.choice(self.fact_patterns)
    
    def generate_daily_qa_example(self) -> Tuple[str, str]:
        """
        Generate daily conversation Q&A examples.
        """
        return random.choice(self.daily_qa_patterns)
    
    def generate_larger_math_example(self) -> Tuple[str, str]:
        """
        Generate math problems with larger numbers (up to 199 range).
        """
        op = random.choice(self.operators)
        
        if op == '+':
            # Can go up to 199 + 199 = 398, but we cap at 199
            a = self._safe_sample_number(100)
            b = self._safe_sample_number(100)
            answer = a + b
            while answer > self.safe_max_answer:
                a = self._safe_sample_number(80)
                b = self._safe_sample_number(80)
                answer = a + b
        elif op == '-':
            a = self._safe_sample_number(150)
            b = self._safe_sample_number(a)
            answer = a - b
        elif op == '*':
            a = self._safe_sample_number(30)
            b = self._safe_sample_number(30)
            answer = a * b
            while answer > self.safe_max_answer:
                a = self._safe_sample_number(20)
                b = self._safe_sample_number(20)
                answer = a * b
        else:  # division
            target = self._safe_sample_number(80)
            b = random.randint(1, 15)
            a = target * b
            answer = target
            if a > self.safe_max_answer:
                a = self._safe_sample_number(150)
                factors = [i for i in range(1, min(a + 1, 20)) if a % i == 0]
                b = random.choice(factors) if factors else 1
                answer = a // b
        
        prompt = f"what is {a} {op} {b}"
        answer_str = f"the answer is {answer}"
        
        return prompt, answer_str
    
    def generate_todo_example(self) -> Tuple[str, str]:
        """
        Generate lightweight task/todo style prompts with deterministic responses.
        """
        action = random.choice(["create", "add", "set", "complete", "remove"])
        item = random.choice(self.todo_items)
        time_hint = random.choice(["today", "tomorrow", "now", "at 9", "in 10"])
        
        if action in ["create", "add", "set"]:
            prompt = f"{action} todo {item} {time_hint}"
            response = f"todo {item} added"
        elif action == "complete":
            prompt = f"complete task {item}"
            response = f"task {item} done"
        else:
            prompt = f"remove todo {item}"
            response = f"todo {item} removed"
        return prompt, response
    
    def generate_multiline_example(self) -> Tuple[str, str]:
        """
        Generate a two-line response using the newline token.
        """
        line1, line2 = random.choice(self.multiline_responses)
        response = f"{line1} {self.vocab.NEWLINE} {line2}"
        # Keep the prompt simple so model learns to emit two lines when asked
        prompt = random.choice([
            "give me two lines",
            "respond in two lines",
            "provide double line answer",
            "two line response please",
            "show two lines"
        ])
        return prompt, response
    
    def generate_greeting_variation(self) -> Tuple[str, str]:
        """
        Generate greeting with more consistent responses.
        
        Returns:
            Tuple of (input_string, response_string)
        """
        input_text = random.choice(self.greeting_inputs)
        # More consistent: always respond with "hello" or "hey" for greetings
        if input_text == "hey":
            response = random.choice(["hello", "hey", "hello i am tinyllm"])
        elif input_text == "hello":
            response = random.choice(["hello", "hey", "hello i am tinyllm"])
        else:
            response = "hello i am tinyllm"
        return input_text, response
    
    def generate_example(self) -> Tuple[str, str]:
        """
        Generate a random training example.
        
        Returns:
            Tuple of (prompt_string, target_string)
        """
        # Randomly select example type with weighted probabilities
        rand = random.random()
        
        if rand < 0.06:
            # 6% greeting examples
            return self.generate_greeting_variation()
        elif rand < 0.11:
            # 5% name examples
            return self.generate_name_example()
        elif rand < 0.15:
            # 4% intro examples
            return self.generate_intro_example()
        elif rand < 0.20:
            # 5% basic Q&A examples
            return self.generate_qa_example()
        elif rand < 0.30:
            # 10% daily Q&A examples
            return self.generate_daily_qa_example()
        elif rand < 0.65:
            # 35% arithmetic query examples (multi-digit, formatted answers)
            return self.generate_arithmetic_query_example()
        else:
            # Remaining 35% split among advanced patterns
            branch = random.random()
            if branch < 0.20:
                return self.generate_arithmetic_compute_example()
            elif branch < 0.35:
                return self.generate_worded_math_example()
            elif branch < 0.50:
                return self.generate_larger_math_example()
            elif branch < 0.65:
                return self.generate_fact_example()
            elif branch < 0.80:
                return self.generate_todo_example()
            else:
                return self.generate_multiline_example()
    
    def generate_dataset(self, num_examples: int) -> List[Tuple[str, str]]:
        """
        Generate a dataset of training examples.
        
        Args:
            num_examples: Number of examples to generate
        
        Returns:
            List of (prompt, target) tuples
        """
        dataset = []
        for _ in range(num_examples):
            example = self.generate_example()
            dataset.append(example)
        return dataset


if __name__ == '__main__':
    # Test data generator
    from vocab import Vocabulary
    
    vocab = Vocabulary()
    generator = SyntheticDataGenerator(vocab)
    
    print("Sample generated examples:")
    for i in range(10):
        prompt, target = generator.generate_example()
        print(f"{i+1}. Input: '{prompt}' -> Target: '{target}'")


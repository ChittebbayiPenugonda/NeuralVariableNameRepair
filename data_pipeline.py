"""
Data Pipeline for Neural Variable Name Repair
This script processes the extracted function data and prepares it for LLM inference.
"""

import json
from typing import List, Dict, Tuple
from pathlib import Path


class DataPipeline:
    """
    Pipeline for processing variable name repair data for LLM inference.
    """
    
    def __init__(self, input_file: str):
        """
        Initialize the pipeline with an input JSONL file.
        
        Args:
            input_file: Path to the JSONL file containing extracted functions
        """
        self.input_file = Path(input_file)
        self.data = []
        
    def load_data(self) -> None:
        """Load data from the JSONL file."""
        self.data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} examples from {self.input_file}")
    
    def get_separate_arrays(self) -> Tuple[List[str], List[str]]:
        """
        Extract input_text and target_text as separate arrays.
        
        Returns:
            Tuple of (input_texts, target_texts)
        """
        input_texts = [item['input_text'] for item in self.data]
        target_texts = [item['target_text'] for item in self.data]
        return input_texts, target_texts
    
    def get_paired_format(self) -> List[Dict[str, str]]:
        """
        Extract data as an array of JSON objects with input_text and target_text pairs.
        
        Returns:
            List of dictionaries with 'input_text' and 'target_text' keys
        """
        paired_data = []
        for item in self.data:
            paired_data.append({
                'input_text': item['input_text'],
                'target_text': item['target_text']
            })
        return paired_data
    
    def get_full_format(self) -> List[Dict[str, str]]:
        """
        Get all fields from the original data.
        
        Returns:
            List of dictionaries with all original fields
        """
        return self.data
    
    def prepare_llm_prompts(self, prompt_template: str = None) -> List[str]:
        """
        Prepare prompts for LLM by combining input_text with a template.
        
        Args:
            prompt_template: Template string with {input_text} placeholder.
                           If None, uses a default template.
        
        Returns:
            List of formatted prompts ready for LLM input
        """
        if prompt_template is None:
            prompt_template = """You are tasked with predicting variable names in code. Given a code snippet with masked variable names (e.g., <ID_1>, <ID_2>), predict the original variable names.

Code:
{input_text}

Provide your answer as a JSON object mapping each placeholder to its predicted variable name, like this: {{"<ID_1>": "variableName", "<ID_2>": "anotherName"}}

Answer:"""
        
        prompts = []
        for item in self.data:
            prompt = prompt_template.format(input_text=item['input_text'])
            prompts.append(prompt)
        
        return prompts
    
    def parse_target_text(self, target_text: str) -> Dict[str, str]:
        """
        Parse the target_text JSON string into a dictionary.
        
        Args:
            target_text: JSON string mapping placeholders to variable names
            
        Returns:
            Dictionary of placeholder -> variable name mappings
        """
        return json.loads(target_text)
    
    def get_parsed_targets(self) -> List[Dict[str, str]]:
        """
        Get all target texts parsed as dictionaries.
        
        Returns:
            List of dictionaries with placeholder -> variable name mappings
        """
        return [self.parse_target_text(item['target_text']) for item in self.data]
    
    def save_prepared_data(self, output_file: str, format: str = 'paired') -> None:
        """
        Save prepared data to a file.
        
        Args:
            output_file: Path to output file
            format: One of 'paired', 'separate', 'prompts', 'full'
        """
        output_path = Path(output_file)
        
        if format == 'paired':
            data = self.get_paired_format()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'separate':
            input_texts, target_texts = self.get_separate_arrays()
            data = {
                'input_texts': input_texts,
                'target_texts': target_texts
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'prompts':
            prompts = self.prepare_llm_prompts()
            targets = [item['target_text'] for item in self.data]
            data = {
                'prompts': prompts,
                'targets': targets
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'full':
            data = self.get_full_format()
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ValueError(f"Unknown format: {format}. Choose from 'paired', 'separate', 'prompts', or 'full'")
        
        print(f"Saved prepared data to {output_path}")


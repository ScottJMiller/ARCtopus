import os
from pathlib import Path
from typing import Dict, Any

# Import utilities from our common.py
from src.common import (
    load_arc_challenges, grid_to_text,
    get_grid_dimensions, get_unique_colors
)

class ARCTopusOrchestrator:
    def __init__(self, data_dir: str = "data/arc-prize-2025"):
        self.data_dir = Path(data_dir)
        self.training_challenges = load_arc_challenges(self.data_dir / "arc-agi_training_challenges.json")
        self.evaluation_challenges = load_arc_challenges(self.data_dir / "arc-agi_evaluation_challenges.json")
        # Note: We won't load solutions here to avoid accidental peeking during development
        # For actual evaluation, solutions would be on the Kaggle side.

    def process_task(self, task_id: str, challenge_set: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single ARC task: loads, preprocesses, and extracts basic features.
        """
        task_data = challenge_set[task_id]
        
        # --- Task Ingestion & Preprocessing ---
        # Convert all train and test examples to text format for LLM consumption
        processed_examples = []
        for example in task_data['train']:
            processed_examples.append({
                'input_text': grid_to_text(example['input']),
                'output_text': grid_to_text(example['output']),
                'input_grid': example['input'], # Keep original for feature extraction / validation
                'output_grid': example['output'] # Keep original for validation
            })
        
        test_inputs_text = [grid_to_text(test_input['input']) for test_input in task_data['test']]
        test_inputs_grid = [test_input['input'] for test_input in task_data['test']]

        # --- Basic Feature Extraction ---
        # Features from the first training example's input
        first_train_input_dims = get_grid_dimensions(processed_examples[0]['input_grid'])
        first_train_output_dims = get_grid_dimensions(processed_examples[0]['output_grid'])
        unique_colors_train = get_unique_colors(processed_examples[0]['input_grid']).union(
                              get_unique_colors(processed_examples[0]['output_grid']))

        # Aggregate processed data and features
        processed_task_data = {
            'task_id': task_id,
            'train_examples_processed': processed_examples,
            'test_inputs_text': test_inputs_text,
            'test_inputs_grid': test_inputs_grid, # Original grid for potential later use
            'features': {
                'first_train_input_dimensions': first_train_input_dims,
                'first_train_output_dimensions': first_train_output_dims,
                'unique_colors_overall': list(unique_colors_train), # Convert set to list for JSON compatibility
                'num_train_examples': len(task_data['train'])
            }
        }
        
        return processed_task_data

    def run_single_task(self, task_id: str, is_evaluation_task: bool = False) -> Dict[str, Any]:
        """
        Runs the full ARCtopus pipeline for a single task.
        (This will be expanded significantly later)
        """
        challenge_set = self.training_challenges if not is_evaluation_task else self.evaluation_challenges
        
        # 1. Task Ingestion & Preprocessing + Basic Feature Extraction
        processed_task = self.process_task(task_id, challenge_set)
        print(f"Processing Task: {processed_task['task_id']}")
        print(f"Train Examples (text):")
        for i, ex in enumerate(processed_task['train_examples_processed']):
            print(f"  Example {i+1} Input:\n{ex['input_text']}")
            print(f"  Example {i+1} Output:\n{ex['output_text']}")
        print(f"Test Input (text):\n{processed_task['test_inputs_text'][0]}") # Assuming one test input for now
        print(f"Extracted Features: {processed_task['features']}")
        
        # --- Dummy Strategy Selector / Routing Logic (Placeholder) ---
        print("\nStrategy Selector: Choosing Program Synthesis Tentacle (Dummy Logic)...")
        # In the future, this is where we'd call a Tentacle and orchestrate its work
        
        # For now, just a placeholder of what a solution might look like
        dummy_prediction = {
            'task_id': task_id,
            'prediction': [[0,0],[0,0]] # A dummy 2x2 black grid as a placeholder prediction
        }
        
        print(f"\nDummy Prediction for Task {task_id}: {dummy_prediction['prediction']}")
        
        return dummy_prediction

# Example usage (for testing)
if __name__ == "__main__":
    # Ensure you are running this from the 'arctopus' directory
    # e.g., if your script is at arctopus/src/orchestrator/main_orchestrator.py
    # you would run: python -m src.orchestrator.main_orchestrator
    
    orchestrator = ARCTopusOrchestrator()
    
    # Pick a random training task ID to test
    # (You can inspect the JSON files to pick a specific one)
    import random
    all_train_ids = list(orchestrator.training_challenges.keys())
    
    # Let's pick a fixed simple one for consistent testing, like '007bbfb7' or '050a417b' if available
    test_task_id = '007bbfb7' if '007bbfb7' in all_train_ids else random.choice(all_train_ids)
    
    # Run the orchestrator for this task
    orchestrator.run_single_task(test_task_id, is_evaluation_task=False)
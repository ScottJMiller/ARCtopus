import os
from pathlib import Path
from typing import Dict, Any

# Import utilities from our common.py
from src.common import (
    load_arc_challenges, grid_to_text,
    get_grid_dimensions, get_unique_colors
)

from src.tentacles.program_synthesis_tentacle import ProgramSynthesisTentacle

class ARCtopus:
    def __init__(self, data_dir: str = "data/arc-prize-2025"):
        self.data_dir = Path(data_dir)
        self.training_challenges = load_arc_challenges(self.data_dir / "arc-agi_training_challenges.json")
        self.evaluation_challenges = load_arc_challenges(self.data_dir / "arc-agi_evaluation_challenges.json")
        
        # Initialize our primary Program Synthesis Tentacle
        # In the future, other tentacles might be initialized here too
        self.program_synthesis_tentacle = ProgramSynthesisTentacle()

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
        """
        challenge_set = self.training_challenges if not is_evaluation_task else self.evaluation_challenges
        
        # 1. Task Ingestion & Preprocessing + Basic Feature Extraction
        processed_task = self.process_task(task_id, challenge_set)
        print(f"Processing Task: {processed_task['task_id']}")
        print(f"Train Examples (text):")
        for i, ex in enumerate(processed_task['train_examples_processed']):
            print(f"  Example {i+1} Input:\n{ex['input_text']}")
            print(f"  Example {i+1} Output:\n{ex['output_text']}")
        print(f"Test Input (text):\n{processed_task['test_inputs_text'][0]}")
        print(f"Extracted Features: {processed_task['features']}")
        
        # --- Strategy Selector / Routing Logic (NOW calling Tentacle) ---
        print("\nStrategy Selector: Routing to Program Synthesis Tentacle (Dummy Logic)...")
        
        # Call the Program Synthesis Tentacle
        predicted_test_outputs = self.program_synthesis_tentacle.solve(processed_task)
        
        submission_entry = {
            'task_id': task_id,
            'prediction': [] # Default to empty if no prediction is made
        }

        if predicted_test_outputs:
            print(f"[ARCTopus] Program Synthesis Tentacle provided prediction.")
            # For submission, we need to provide a list of predictions for each test input.
            # The competition allows up to 2 attempts, but for now we'll just use the first prediction.
            submission_entry['prediction'] = predicted_test_outputs[0] # Take first prediction from list
        else:
            print(f"[ARCTopus] Program Synthesis Tentacle failed to provide prediction for task {task_id}.")
        
        print(f"\nFinal Prediction for Task {task_id}: {submission_entry['prediction']}")
        
        return submission_entry
    
    def run_all_tasks(self, is_evaluation_set: bool = False, output_filepath: str = "submission.json"):
        """
        Runs the ARCtopus on all tasks in the specified set and saves a submission file.
        """
        challenges_to_run = self.evaluation_challenges if is_evaluation_set else self.training_challenges
        submission_data = {"predictions": []}
        
        print(f"\n--- Running ARCTopus on {'EVALUATION' if is_evaluation_set else 'TRAINING'} set ({len(challenges_to_run)} tasks) ---")
        
        for i, task_id in enumerate(challenges_to_run.keys()):
            print(f"\n===== Task {i+1}/{len(challenges_to_run)}: {task_id} =====")
            task_prediction = self.run_single_task(task_id, is_evaluation_task=is_evaluation_set)
            submission_data["predictions"].append(task_prediction)
            
            # For brevity, let's just run 1 task for now in the main block
            # In actual competition, you'd run all.
            # For the basic skeleton, we'll stop after the first for a quick test.
            if i == 0:
                print("\n(Stopping after first task for basic skeleton test. Remove this break for full run.)")
                break 
        
        # Save dummy submission (or actual predictions later)
        # save_submission(submission_data, output_filepath)
        # print(f"\nSubmission saved to {output_filepath}")

# Example usage (for testing)
if __name__ == "__main__":
    
    orchestrator = ARCtopus()
    
    # Run a single training task as a test for end-to-end flow
    # It will call the dummy LLM and always return [[0,0],[0,0]]
    import random
    all_train_ids = list(orchestrator.training_challenges.keys())
    
    # Pick a fixed simple one for consistent testing, like '007bbfb7' or '050a417b' if available
    test_task_id = '007bbfb7' if '007bbfb7' in all_train_ids else random.choice(all_train_ids)
    
    print(f"\n--- Initiating ARCTopus run for single task: {test_task_id} ---")
    orchestrator.run_single_task(test_task_id, is_evaluation_task=False)
    print("\n--- Single task run complete. ---")

    # To run on all tasks (after Program Synthesis Tentacle is more robust):
    # orchestrator.run_all_tasks(is_evaluation_set=False, output_filepath="dummy_train_submission.json")
    # print("\n--- Full training set run complete. Check dummy_train_submission.json ---")
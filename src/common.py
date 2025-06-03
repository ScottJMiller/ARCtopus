from pathlib import Path
import numpy as np
import json

# Define a simple mapping for colors (integers 0-9 to basic names)
# You can customize these names later if a multimodal LLM prefers different names.
COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "gray",
    6: "fuchsia", # Often pink/purple, choosing fuchsia for distinctness
    7: "orange",
    8: "teal",
    9: "maroon",
}

# Reverse mapping for converting text back to numbers
COLOR_NUMBERS = {name: num for num, name in COLOR_NAMES.items()}

def grid_to_text(grid: list[list[int]]) -> str:
    """
    Converts a numerical ARC grid (list of lists of ints) to a text string
    suitable for an LLM.
    Each number is replaced by its color name, rows are separated by newlines,
    and cells by spaces.
    """
    if not grid:
        return ""
    
    text_rows = []
    for row in grid:
        text_cells = [COLOR_NAMES.get(cell, str(cell)) for cell in row] # Use str(cell) as fallback
        text_rows.append(" ".join(text_cells))
    return "\n".join(text_rows)

def text_to_grid(text_grid: str) -> list[list[int]]:
    """
    Converts a text string representation of an ARC grid back to numerical format.
    """
    if not text_grid:
        return []

    grid = []
    for line in text_grid.strip().split('\n'):
        row = []
        for cell_name in line.strip().split(' '):
            row.append(COLOR_NUMBERS.get(cell_name.lower(), 0)) # Default to 0 (black) if name not found
        grid.append(row)
    return grid

def load_arc_challenges(filepath: str | Path) -> dict:
    """
    Loads ARC challenges from a JSON file.
    """
    with open(filepath, 'r') as f:
        challenges = json.load(f)
    return challenges

def save_submission(submission_data: dict, filepath: str):
    """
    Saves submission data to a JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(submission_data, f, indent=4)

# Basic Feature Extraction utilities (will be expanded)
def get_grid_dimensions(grid: list[list[int]]) -> tuple[int, int]:
    """Returns (height, width) of a grid."""
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]))

def get_unique_colors(grid: list[list[int]]) -> set[int]:
    """Returns a set of unique color integers used in a grid."""
    colors = set()
    for row in grid:
        for cell in row:
            colors.add(cell)
    return colors
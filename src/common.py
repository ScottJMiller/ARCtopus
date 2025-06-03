from pathlib import Path
import numpy as np
import json
from collections import deque # For flood_fill

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

# --- Core Grid Manipulation Primitives ---

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

def create_grid(height: int, width: int, fill_color: int = 0) -> list[list[int]]:
    """Creates a new grid of specified dimensions filled with a given color."""
    return [[fill_color for _ in range(width)] for _ in range(height)]

def copy_grid(grid: list[list[int]]) -> list[list[int]]:
    """Creates a deep copy of a grid."""
    return [row[:] for row in grid] # Simple list comprehension for 2D lists

def get_pixels_with_color(grid: list[list[int]], color: int) -> list[tuple[int, int]]:
    """Returns a list of (row, col) coordinates for all pixels of a specific color."""
    pixels = []
    for r_idx, row in enumerate(grid):
        for c_idx, cell in enumerate(row):
            if cell == color:
                pixels.append((r_idx, c_idx))
    return pixels

def set_pixel(grid: list[list[int]], row: int, col: int, color: int):
    """Sets the color of a specific pixel in the grid."""
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        grid[row][col] = color

# --- Basic Object Detection (Simplified Connected Components) ---

def find_connected_components(grid: list[list[int]], background_color: int = 0, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    """
    Finds connected components (objects) in a grid.
    Returns a list of components, where each component is a list of (row, col) coordinates.
    connectivity: 4 (cardinal directions) or 8 (including diagonals).
    """
    height, width = get_grid_dimensions(grid)
    visited = [[False for _ in range(width)] for _ in range(height)]
    components = []

    def is_valid(r, c):
        return 0 <= r < height and 0 <= c < width and not visited[r][c] and grid[r][c] != background_color

    # Define directions based on connectivity
    if connectivity == 4:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up
    elif connectivity == 8:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] # All 8 directions
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    for r in range(height):
        for c in range(width):
            if is_valid(r, c):
                component = []
                q = deque([(r, c)])
                visited[r][c] = True
                
                while q:
                    curr_r, curr_c = q.popleft()
                    component.append((curr_r, curr_c))

                    for dr, dc in directions:
                        next_r, next_c = curr_r + dr, curr_c + dc
                        if is_valid(next_r, next_c):
                            visited[next_r][next_c] = True
                            q.append((next_r, next_c))
                components.append(component)
    return components

# --- Basic Geometric Transformations ---

def get_bounding_box(pixels: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """
    Calculates the bounding box (min_r, min_c, max_r, max_c) for a list of pixels.
    Returns (min_row, min_col, max_row, max_col)
    """
    if not pixels:
        return (0, 0, -1, -1) # Indicate empty bounding box

    min_r = min(p[0] for p in pixels)
    max_r = max(p[0] for p in pixels)
    min_c = min(p[1] for p in pixels)
    max_c = max(p[1] for p in pixels)

    return (min_r, min_c, max_r, max_c)

def crop_object(grid: list[list[int]], object_pixels: list[tuple[int, int]]) -> list[list[int]]:
    """
    Crops a subgrid containing the object defined by its pixels.
    The background color of the cropped grid will be 0 (black).
    """
    if not object_pixels:
        return []

    min_r, min_c, max_r, max_c = get_bounding_box(object_pixels)
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    cropped_grid = create_grid(height, width, 0) # Fill with black background

    for r, c in object_pixels:
        # Map global coordinates to local cropped grid coordinates
        local_r = r - min_r
        local_c = c - min_c
        cropped_grid[local_r][local_c] = grid[r][c]
        
    return cropped_grid

def blit_grid(target_grid: list[list[int]], source_grid: list[list[int]], row_offset: int, col_offset: int):
    """
    Copies (blits) a source_grid onto a target_grid at a specified offset.
    Pixels with color 0 (black) in source_grid are considered transparent.
    """
    target_height, target_width = get_grid_dimensions(target_grid)
    source_height, source_width = get_grid_dimensions(source_grid)

    for r in range(source_height):
        for c in range(source_width):
            if source_grid[r][c] != 0: # Only blit non-black pixels
                target_r = row_offset + r
                target_c = col_offset + c
                if 0 <= target_r < target_height and 0 <= target_c < target_width:
                    target_grid[target_r][target_c] = source_grid[r][c]

# --- Example of a simple transform that could be generated by LLM (for testing) ---
# This function is here just as an example of what an LLM might generate using these primitives.
# It's not part of the common library itself, but shows how it would be used.
def example_llm_generated_transform(input_grid: list[list[int]]) -> list[list[int]]:
    """
    Example transform: Finds the largest object, crops it, and places it in the center
    of a new grid twice the size of the original input.
    """
    input_h, input_w = get_grid_dimensions(input_grid)
    
    # Use common.py primitives
    objects = find_connected_components(input_grid, background_color=0, connectivity=8)
    
    if not objects:
        return create_grid(input_h * 2, input_w * 2, 0) # Return empty scaled grid if no objects

    # Find largest object
    largest_object_pixels = []
    for obj_pixels in objects:
        if len(obj_pixels) > len(largest_object_pixels):
            largest_object_pixels = obj_pixels
            
    cropped_obj_grid = crop_object(input_grid, largest_object_pixels)
    
    # Create new output grid
    output_h, output_w = input_h * 2, input_w * 2
    output_grid = create_grid(output_h, output_w, 0)
    
    # Blit object to center
    obj_h, obj_w = get_grid_dimensions(cropped_obj_grid)
    center_r = (output_h // 2) - (obj_h // 2)
    center_c = (output_w // 2) - (obj_w // 2)
    
    blit_grid(output_grid, cropped_obj_grid, center_r, center_c)
    
    return output_grid
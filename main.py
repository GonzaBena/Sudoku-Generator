import os
import random
from typing import List, Optional
from enum import Enum
import copy
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn


class Difficult(Enum):
    EASY = 15
    MEDIUM = 25
    HARD = 35
    VERYHARD = 45


class Sudoku:

    def __init__(
        self,
        matrix: Optional[list[list[int]]],
        difficult: Optional[Difficult],
        solution,
    ):
        self.matrix = matrix
        self.solution = solution
        self.difficult = difficult


def exist_in_col(num: int, col: int, grid: list[list[Optional[int]]]) -> bool:
    """Check if a number exists in a given column."""
    for row in grid:
        if row[col] == num:
            return True
    return False


def exist_in_quadrant(
    num: int, row: int, col: int, grid: list[list[Optional[int]]]
) -> bool:
    """
    Check if a number exists in the 3x3 quadrant of a given cell.

       1 2 3   4 5 6   7 8 9
    1 [- - -   - - -   - - -],
    2 [- - -   - - -   - - -], 1  2  3
    3 [- - -   - - -   - - -],

    4 [- - -   - - -   - - -],
    5 [- - -   - - -   - - -], 4  5  6
    6 [- - -   - - -   - - -],

    7 [- - -   - - -   - - -],
    8 [- - -   - - -   - - -], 7  8  9
    9 [- - -   - - -   - - -],
    """
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if grid[r][c] == num:
                return True
    return False


def shuffle_rows_and_columns(sudoku: Sudoku) -> None:
    """Shuffle rows and columns within each 3x3 block to randomize the grid."""
    block_size = 3
    grid: list[list[int]] = sudoku.solution

    # Shuffle rows within each block
    for block_start in range(0, len(grid), block_size):
        rows = grid[block_start : block_start + block_size]
        random.shuffle(rows)
        grid[block_start : block_start + block_size] = rows

    # Transpose the grid to shuffle columns (rows become columns)
    grid[:] = list(map(list, zip(*grid)))
    for block_start in range(0, len(grid), block_size):
        cols = grid[block_start : block_start + block_size]
        random.shuffle(cols)
        grid[block_start : block_start + block_size] = cols

    # Transpose back to original orientation
    grid[:] = list(map(list, zip(*grid)))

    sudoku.solution = grid


def generate_sudoku(size: int = 9) -> Sudoku:
    """Generate a randomized Sudoku grid of the given size using backtracking."""
    if size % 3 != 0:
        raise ValueError("The grid size must be a multiple of 3 (e.g., 9x9).")

    grid: list[list[Optional[int]]] = [[None for _ in range(size)] for _ in range(size)]

    def is_valid(num: int, row: int, col: int) -> bool:
        """Check if placing num at grid[row][col] is valid."""
        return (
            num not in grid[row]
            and not exist_in_col(num, col, grid)
            and not exist_in_quadrant(num, row, col, grid)
        )

    def solve(row: int, col: int) -> bool:
        """Backtracking solver to fill the grid."""
        if row == size:  # If we've reached beyond the last row, the grid is complete
            return True

        next_row, next_col = (row, col + 1) if col + 1 < size else (row + 1, 0)

        if grid[row][col] is not None:  # Skip pre-filled cells
            return solve(next_row, next_col)

        numbers = list(range(1, 10))  # Possible numbers for Sudoku
        random.shuffle(numbers)  # Randomize the order of numbers

        for num in numbers:
            if is_valid(num, row, col):
                grid[row][col] = num  # Try placing the number
                if solve(next_row, next_col):  # Recurse to the next cell
                    return True
                grid[row][col] = None  # Backtrack if the placement fails

        return False  # No valid numbers found; trigger backtracking

    if not solve(0, 0):
        raise ValueError("Sudoku grid generation failed.")

    return Sudoku(None, None, grid)


def count_solutions(grid: List[List[Optional[int]]]) -> int:
    """Count the number of solutions for a given Sudoku grid."""
    size = len(grid)
    solution_count = 0

    def is_valid(num: int, row: int, col: int) -> bool:
        """Check if placing num at grid[row][col] is valid."""
        return (
            num not in grid[row]
            and not exist_in_col(num, col, grid)
            and not exist_in_quadrant(num, row, col, grid)
        )

    def solve(row: int, col: int) -> bool:
        """Backtracking solver to count solutions."""
        nonlocal solution_count
        if row == size:
            solution_count += 1
            return solution_count > 1  # Stop if more than 1 solution is found

        next_row, next_col = (row, col + 1) if col + 1 < size else (row + 1, 0)

        if grid[row][col] is not None:
            return solve(next_row, next_col)

        for num in range(1, 10):
            if is_valid(num, row, col):
                grid[row][col] = num
                if solve(next_row, next_col):
                    grid[row][col] = None
                    return True
                grid[row][col] = None

        return False

    solve(0, 0)
    return solution_count


def remove_numbers(mygrid, difficult: Difficult) -> list[list[Optional[int]]]:
    """Remove a specified number of cells from a complete grid to create a unique puzzle."""
    grid = copy.deepcopy(mygrid)  # Create a deep copy of the grid
    size = len(grid)
    total_cells = size * size
    num_cells_to_remove = difficult.value
    if num_cells_to_remove > total_cells:
        raise ValueError(
            f"Cannot remove more than {total_cells} cells in a {size}x{size} grid."
        )

    cells = [(row, col) for row in range(size) for col in range(size)]
    random.shuffle(cells)

    removed_cells = 0
    for row, col in cells:
        if removed_cells >= num_cells_to_remove:
            break

        # Backup the current value
        backup = grid[row][col]
        grid[row][col] = None

        # Check if the grid still has a unique solution
        if count_solutions(grid) != 1:
            # Restore the value if it breaks uniqueness
            grid[row][col] = backup
        else:
            removed_cells += 1

    return grid


def main():
    sudoku = generate_sudoku(9)
    shuffle_rows_and_columns(sudoku)
    grid = sudoku.solution
    sudoku.difficult = Difficult.MEDIUM
    sudoku.matrix = remove_numbers(grid, sudoku.difficult)  # Assign the returned grid
    # print(" \n\n\n\n")
    # print("Puzzle:")
    # for row in sudoku.matrix:
    #     print(row)
    # print("\nSolution:")
    # for row in sudoku.solution:
    #     print(row)


app = FastAPI()

# Configurar los orígenes permitidos
origins = [
    "*"  # Permite todos los orígenes. Cambia esto a una lista de orígenes específicos si lo prefieres.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Puedes especificar dominios específicos aquí en lugar de "*".
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.).
    allow_headers=["*"],  # Permite todos los encabezados.
)


@app.get("/")
async def root(difficult: str = ""):
    print(difficult)
    sudoku = generate_sudoku(9)
    shuffle_rows_and_columns(sudoku)
    grid = sudoku.solution
    difficults = list(Difficult.__dict__["_member_map_"].keys())
    if difficult != "" and difficult.upper() in difficults:
        sudoku.difficult = Difficult[difficult.upper()]
    else:
        sudoku.difficult = Difficult[random.choice(difficults)]
        print(random.choice(difficults))
    sudoku.matrix = remove_numbers(grid, sudoku.difficult)  # Assign the returned grid
    return {
        "puzzle": sudoku.matrix,
        "difficult": sudoku.difficult.name,
        "difficultLevel": sudoku.difficult.value,
        "solution": sudoku.solution,
    }


if __name__ == "__main__":
    main()
    port = int(os.environ.get("PORT", 8000))  # Usa el puerto especificado por Render
    uvicorn.run(app, host="0.0.0.0", port=port)

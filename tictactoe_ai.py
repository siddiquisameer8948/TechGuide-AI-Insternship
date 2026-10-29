"""
Tic-Tac-Toe AI - Unbeatable Opponent using Minimax + Alpha-Beta Pruning
TechGuide AI Internship | Project 2

Run in terminal :  python tictactoe_ai.py
Run with a window:  python tictactoe_ai.py --gui
Compare pruning  :  python tictactoe_ai.py --compare

How it works
- The board is a list of 9 cells (index 0-8). Human = "X", AI = "O".
- Minimax looks at EVERY possible future game. It assumes the AI always picks
  the move with the highest score and the human always picks the move with the
  lowest score.
- Scores: AI win = +10 - depth, human win = depth - 10, draw = 0.
  Subtracting depth makes the AI prefer faster wins and slower losses.
- Alpha-Beta Pruning skips branches that cannot change the final decision,
  giving the same answer with far fewer positions searched.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List, Optional, Tuple

HUMAN, AI, EMPTY = "X", "O", " "

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]

# Trying centre and corners first makes pruning more effective
MOVE_ORDER = [4, 0, 2, 6, 8, 1, 3, 5, 7]


# --------------------------------------------------------------------------
# Game logic
# --------------------------------------------------------------------------
def winner(board: List[str]) -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
    """Return (winning symbol, winning line) or (None, None)."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a], (a, b, c)
    return None, None


def is_full(board: List[str]) -> bool:
    return EMPTY not in board


def available_moves(board: List[str]) -> List[int]:
    return [i for i in MOVE_ORDER if board[i] == EMPTY]


# --------------------------------------------------------------------------
# Minimax (with optional Alpha-Beta Pruning)
# --------------------------------------------------------------------------
def minimax(board: List[str], depth: int, maximizing: bool,
            alpha: float, beta: float, counter: list, prune: bool) -> int:
    counter[0] += 1  # count positions examined

    won, _ = winner(board)
    if won == AI:
        return 10 - depth
    if won == HUMAN:
        return depth - 10
    if is_full(board):
        return 0

    if maximizing:                      # AI's turn: pick the highest score
        best = -math.inf
        for move in available_moves(board):
            board[move] = AI
            score = minimax(board, depth + 1, False, alpha, beta, counter, prune)
            board[move] = EMPTY
            best = max(best, score)
            alpha = max(alpha, best)
            if prune and beta <= alpha:  # remaining moves cannot matter
                break
        return best

    best = math.inf                     # Human's turn: pick the lowest score
    for move in available_moves(board):
        board[move] = HUMAN
        score = minimax(board, depth + 1, True, alpha, beta, counter, prune)
        board[move] = EMPTY
        best = min(best, score)
        beta = min(beta, best)
        if prune and beta <= alpha:
            break
    return best


def best_move(board: List[str], prune: bool = True) -> Tuple[int, int]:
    """Return (best cell index for the AI, number of positions searched)."""
    counter = [0]
    best_score, move = -math.inf, -1
    for m in available_moves(board):
        board[m] = AI
        score = minimax(board, 1, False, -math.inf, math.inf, counter, prune)
        board[m] = EMPTY
        if score > best_score:
            best_score, move = score, m
    return move, counter[0]


# --------------------------------------------------------------------------
# Terminal interface
# --------------------------------------------------------------------------
def print_board(board: List[str]) -> None:
    def cell(i: int) -> str:
        return board[i] if board[i] != EMPTY else str(i + 1)

    print()
    for r in range(3):
        print(f"   {cell(3*r)} | {cell(3*r+1)} | {cell(3*r+2)}")
        if r < 2:
            print("  ---+---+---")
    print()


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please type y or n.")


def get_human_move(board: List[str]) -> int:
    while True:
        raw = input("Your move (1-9): ").strip()
        if not raw.isdigit() or not 1 <= int(raw) <= 9:
            print("Enter a number from 1 to 9.")
        elif board[int(raw) - 1] != EMPTY:
            print("That cell is already taken.")
        else:
            return int(raw) - 1


def play_cli_game(human_first: bool) -> str:
    board = [EMPTY] * 9
    human_turn = human_first
    print_board(board)

    while True:
        if human_turn:
            board[get_human_move(board)] = HUMAN
        else:
            start = time.perf_counter()
            move, nodes = best_move(board)
            ms = (time.perf_counter() - start) * 1000
            board[move] = AI
            print(f"AI plays {move + 1}  (searched {nodes} positions in {ms:.1f} ms)")
        print_board(board)

        won, _ = winner(board)
        if won:
            print("You win!" if won == HUMAN else "AI wins! Better luck next time.")
            return won
        if is_full(board):
            print("It's a draw!")
            return "Draw"
        human_turn = not human_turn


def run_cli() -> None:
    print("=" * 50)
    print("  Tic-Tac-Toe vs Unbeatable AI (Minimax + Alpha-Beta)")
    print("  You are X, the AI is O. Cells are numbered 1-9.")
    print("=" * 50)
    scores = {HUMAN: 0, AI: 0, "Draw": 0}
    while True:
        result = play_cli_game(ask_yes_no("Do you want to go first? (y/n): "))
        scores[result] += 1
        print(f"Score -> You: {scores[HUMAN]} | AI: {scores[AI]} | Draws: {scores['Draw']}")
        if not ask_yes_no("Play again? (y/n): "):
            print("Thanks for playing!")
            break


# --------------------------------------------------------------------------
# Compare Minimax with and without pruning
# --------------------------------------------------------------------------
def run_compare() -> None:
    board = [EMPTY] * 9
    print("Searching the empty board (AI to move)...\n")
    for prune in (False, True):
        start = time.perf_counter()
        move, nodes = best_move(board, prune=prune)
        ms = (time.perf_counter() - start) * 1000
        label = "With Alpha-Beta   " if prune else "Plain Minimax      "
        print(f"{label}: best move = {move + 1}, positions searched = {nodes:>7,}, time = {ms:8.1f} ms")


# --------------------------------------------------------------------------
# GUI (tkinter ships with Python, nothing to install)
# --------------------------------------------------------------------------
def run_gui() -> None:
    import tkinter as tk

    class TicTacToeGUI:
        def __init__(self, root: tk.Tk):
            self.root = root
            root.title("Tic-Tac-Toe AI - Minimax + Alpha-Beta")
            root.configure(bg="#111827")
            root.resizable(False, False)

            self.board = [EMPTY] * 9
            self.scores = {HUMAN: 0, AI: 0, "Draw": 0}
            self.game_over = False
            self.locked = False
            self.ai_first = tk.BooleanVar(value=False)

            tk.Label(root, text="Tic-Tac-Toe AI", bg="#111827", fg="#f9fafb",
                     font=("Segoe UI", 20, "bold")).pack(pady=(14, 2))
            tk.Label(root, text="You are X  |  AI is O  |  Unbeatable",
                     bg="#111827", fg="#9ca3af", font=("Segoe UI", 10)).pack()

            self.score_label = tk.Label(root, bg="#111827", fg="#fbbf24",
                                        font=("Segoe UI", 11, "bold"))
            self.score_label.pack(pady=8)

            grid = tk.Frame(root, bg="#111827")
            grid.pack(padx=20)
            self.buttons = []
            for i in range(9):
                btn = tk.Button(grid, text="", width=4, height=1,
                                font=("Segoe UI", 34, "bold"),
                                bg="#1f2937", fg="#f9fafb", activebackground="#374151",
                                bd=0, command=lambda i=i: self.on_click(i))
                btn.grid(row=i // 3, column=i % 3, padx=4, pady=4)
                self.buttons.append(btn)

            self.status = tk.Label(root, bg="#111827", fg="#f9fafb",
                                   font=("Segoe UI", 12))
            self.status.pack(pady=(10, 2))

            controls = tk.Frame(root, bg="#111827")
            controls.pack(pady=(4, 16))
            tk.Checkbutton(controls, text="AI plays first", variable=self.ai_first,
                           bg="#111827", fg="#d1d5db", selectcolor="#1f2937",
                           activebackground="#111827", activeforeground="#d1d5db"
                           ).pack(side=tk.LEFT, padx=8)
            tk.Button(controls, text="New Game", command=self.new_game,
                      bg="#2563eb", fg="white", font=("Segoe UI", 11, "bold"),
                      bd=0, padx=14, pady=5).pack(side=tk.LEFT, padx=8)

            self.new_game()

        # ---- helpers -------------------------------------------------
        def refresh(self) -> None:
            for i, btn in enumerate(self.buttons):
                symbol = self.board[i]
                colour = "#60a5fa" if symbol == HUMAN else "#f87171"
                btn.config(text=symbol if symbol != EMPTY else "", fg=colour)

        def update_scores(self) -> None:
            self.score_label.config(
                text=f"You: {self.scores[HUMAN]}    AI: {self.scores[AI]}    Draws: {self.scores['Draw']}")

        def new_game(self) -> None:
            self.board = [EMPTY] * 9
            self.game_over = False
            self.locked = False
            for btn in self.buttons:
                btn.config(bg="#1f2937")
            self.refresh()
            self.update_scores()
            if self.ai_first.get():
                self.locked = True
                self.status.config(text="AI is thinking...")
                self.root.after(350, self.ai_turn)
            else:
                self.status.config(text="Your turn (X)")

        # ---- game flow -----------------------------------------------
        def on_click(self, i: int) -> None:
            if self.game_over or self.locked or self.board[i] != EMPTY:
                return
            self.board[i] = HUMAN
            self.refresh()
            if self.check_end():
                return
            self.locked = True
            self.status.config(text="AI is thinking...")
            self.root.after(350, self.ai_turn)

        def ai_turn(self) -> None:
            move, nodes = best_move(self.board)
            self.board[move] = AI
            self.refresh()
            self.locked = False
            if not self.check_end():
                self.status.config(
                    text=f"Your turn (X)  |  AI searched {nodes:,} positions")

        def check_end(self) -> bool:
            won, line = winner(self.board)
            if won:
                self.game_over = True
                for i in line:
                    self.buttons[i].config(bg="#166534" if won == HUMAN else "#7f1d1d")
                self.scores[won] += 1
                self.status.config(text="You win!" if won == HUMAN else "AI wins!")
                self.update_scores()
                return True
            if is_full(self.board):
                self.game_over = True
                self.scores["Draw"] += 1
                self.status.config(text="It's a draw!")
                self.update_scores()
                return True
            return False

    root = tk.Tk()
    TicTacToeGUI(root)
    root.mainloop()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unbeatable Tic-Tac-Toe AI")
    parser.add_argument("--gui", action="store_true", help="open the game window")
    parser.add_argument("--compare", action="store_true",
                        help="compare Minimax with and without Alpha-Beta pruning")
    args = parser.parse_args()

    if args.compare:
        run_compare()
    elif args.gui:
        run_gui()
    else:
        run_cli()
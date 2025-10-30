import chess
import random
n = 1000
board = chess.Board()
print(board)
for _ in range(n):
    moves = list(board.legal_moves)
    move = random.choice(moves)
    board.push(move)
    print(board)
    print()
    if board.is_game_over():
        print("Game over:", board.result())
        break
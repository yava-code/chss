#logic is: we want to translate from a chess.Board() to a numpy array for future pytorch tensor
import chess
import numpy as np
#at first i thought that was simple, but turns out that board.pieces is not array but "Bitboard" object
board = chess.Board()
x = np.zeros((8, 8), dtype=int)
pawn_squares = board.pieces(chess.PAWN, chess.WHITE)

#pretty hard part, because we need to convert from chess.Square to row/col
for square in pawn_squares:
    # We get "coordinates" from 0 to 7
    row = chess.square_rank(square)
    col = chess.square_file(square)

    # Update our numpy array
    x[row][col] = 1
#after half and our we have the numpy array
print(x)
# Output:
# [[0 0 0 0 0 0 0 0]
#  [1 1 1 1 1 1 1 1]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0]]
#where 1 means white pawn presence
#similarly we can do for other pieces and black pieces too🙄🙄🙄
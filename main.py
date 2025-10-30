import chess
import numpy as np
from translator import create_piece_layer

# We will need to go through all 6 types of pieces:
# • chess.PAWN
# • chess.KNIGHT
# • chess.BISHOP
# • chess.ROOK
# • chess.QUEEN
# • chess.KING
# ...and each of the two colors (chess.WHITE, chess.BLACK).
board = chess.Board()
tensor8 = []
piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
colors = [chess.WHITE, chess.BLACK]

for color in colors:
    for piece_type in piece_types:
        tensor8.append(create_piece_layer(board, piece_type, color))

np_array = np.array(tensor8)
print(np_array.shape)  # Should print (12, 8, 8)
turn_layer = np.ones((8, 8), dtype=int) if board.turn == chess.WHITE else np.zeros((8, 8), dtype=int)
np_array = np.concatenate((np_array, turn_layer[np.newaxis, :, :]), axis=0)
print(np_array.shape)  # Should print (13, 8, 8)
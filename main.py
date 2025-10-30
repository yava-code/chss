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


# Now we need to add 4 more layers for castling rights
castling_checks = [
    board.has_kingside_castling_rights(chess.WHITE),
    board.has_queenside_castling_rights(chess.WHITE),
    board.has_kingside_castling_rights(chess.BLACK),
    board.has_queenside_castling_rights(chess.BLACK)
]

# For each of these, we will create an 8x8 layer filled with 1s if the right exists, or 0s if it does not.
for has_right in castling_checks:
    # Создаем слой 8x8 из 1 или 0
    layer = np.ones((8, 8), dtype=int) if has_right else np.zeros((8, 8), dtype=int)


    np_array = np.concatenate((np_array, layer[np.newaxis, :, :]), axis=0)

print(np_array.shape)  # (17, 8, 8)
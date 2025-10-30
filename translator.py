#logic is: we want to translate from a chess.Board() to a numpy array for future pytorch tensor
import chess
import numpy as np

#lets try to create a function
def create_piece_layer(board, piece_type, color):
    layer = np.zeros((8, 8), dtype=int)
    piece_squares = board.pieces(piece_type, color)
    for square in piece_squares:
        row = chess.square_rank(square)
        col = chess.square_file(square)
        layer[row][col] = 1
    return layer
#how i get from start to function: i created a code for white pawns and researched how to make "Bitboard" to get all squares with white pawns
#then i created a "universal" function that takes piece type and color as arguments
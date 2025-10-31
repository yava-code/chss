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


# how i get from start to function: i created a code for white pawns and researched how to make "Bitboard" to get all
# squares with white pawns then i created a "universal" function that takes piece type and color as arguments
def move_to_index(move: chess.Move) -> int:
    """
    Converts a chess move into a unique index from 0 to 4095.
    (from_square * 64) + to_square
    """
    # move.from_square (от 0 до 63)
    # move.to_square (от 0 до 63)
    return (move.from_square * 64) + move.to_square


def index_to_move(index: int) -> chess.Move:
    """
    Converts an index (0-4095) back into a chess move.
    This is the inverse operation of move_to_index.
    """
    # 1. Find the "column" (where)
    # Use the "remainder of division" (%)
    to_square = index % 64

    # 2. Find the "row" (from where)
    # Use "integer division" (//)
    from_square = index // 64

    # 3. Create a Move object from these two numbers
    return chess.Move(from_square, to_square)


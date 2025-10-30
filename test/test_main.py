import unittest
import numpy as np
import chess
from main import create_piece_layer

class TestMain(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
        self.piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        self.colors = [chess.WHITE, chess.BLACK]

    def test_tensor8_shape(self):
        tensor8 = []
        for color in self.colors:
            for piece_type in self.piece_types:
                tensor8.append(create_piece_layer(self.board, piece_type, color))
        np_array = np.array(tensor8)
        self.assertEqual(np_array.shape, (12, 8, 8))

    def test_turn_layer(self):
        turn_layer = np.ones((8, 8), dtype=int) if self.board.turn == chess.WHITE else np.zeros((8, 8), dtype=int)
        self.assertEqual(turn_layer.shape, (8, 8))
        self.assertTrue(np.all(turn_layer == 1) if self.board.turn == chess.WHITE else np.all(turn_layer == 0))

    def test_castling_layers(self):
        castling_checks = [
            self.board.has_kingside_castling_rights(chess.WHITE),
            self.board.has_queenside_castling_rights(chess.WHITE),
            self.board.has_kingside_castling_rights(chess.BLACK),
            self.board.has_queenside_castling_rights(chess.BLACK)
        ]
        for has_right in castling_checks:
            layer = np.ones((8, 8), dtype=int) if has_right else np.zeros((8, 8), dtype=int)
            self.assertEqual(layer.shape, (8, 8))
            self.assertTrue(np.all(layer == 1) if has_right else np.all(layer == 0))

    def test_final_array_shape(self):
        tensor8 = []
        for color in self.colors:
            for piece_type in self.piece_types:
                tensor8.append(create_piece_layer(self.board, piece_type, color))
        np_array = np.array(tensor8)
        turn_layer = np.ones((8, 8), dtype=int) if self.board.turn == chess.WHITE else np.zeros((8, 8), dtype=int)
        np_array = np.concatenate((np_array, turn_layer[np.newaxis, :, :]), axis=0)

        castling_checks = [
            self.board.has_kingside_castling_rights(chess.WHITE),
            self.board.has_queenside_castling_rights(chess.WHITE),
            self.board.has_kingside_castling_rights(chess.BLACK),
            self.board.has_queenside_castling_rights(chess.BLACK)
        ]
        for has_right in castling_checks:
            layer = np.ones((8, 8), dtype=int) if has_right else np.zeros((8, 8), dtype=int)
            np_array = np.concatenate((np_array, layer[np.newaxis, :, :]), axis=0)

        self.assertEqual(np_array.shape, (17, 8, 8))

if __name__ == '__main__':
    unittest.main()
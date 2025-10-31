import unittest
import numpy as np
import chess
import torch
from translator import create_piece_layer, move_to_index, index_to_move
from ChessNet import ChessNet


class TestCreatePieceLayer(unittest.TestCase):
    """Tests for the create_piece_layer function from translator.py"""
    
    def setUp(self):
        self.board = chess.Board()
    
    def test_returns_8x8_array(self):
        """Test that create_piece_layer returns an 8x8 numpy array"""
        layer = create_piece_layer(self.board, chess.PAWN, chess.WHITE)
        self.assertEqual(layer.shape, (8, 8))
        self.assertEqual(layer.dtype, int)
    
    def test_white_pawns_starting_position(self):
        """Test that white pawns are correctly positioned at start"""
        layer = create_piece_layer(self.board, chess.PAWN, chess.WHITE)
        # White pawns should be on rank 1 (index 1)
        self.assertEqual(np.sum(layer[1, :]), 8)  # 8 pawns on rank 1
        self.assertEqual(np.sum(layer), 8)  # Only 8 ones total
    
    def test_black_pawns_starting_position(self):
        """Test that black pawns are correctly positioned at start"""
        layer = create_piece_layer(self.board, chess.PAWN, chess.BLACK)
        # Black pawns should be on rank 6 (index 6)
        self.assertEqual(np.sum(layer[6, :]), 8)  # 8 pawns on rank 6
        self.assertEqual(np.sum(layer), 8)  # Only 8 ones total
    
    def test_white_king_position(self):
        """Test that white king is correctly positioned"""
        layer = create_piece_layer(self.board, chess.KING, chess.WHITE)
        self.assertEqual(np.sum(layer), 1)  # Only 1 king
        self.assertEqual(layer[0, 4], 1)  # King at e1 (rank 0, file 4)
    
    def test_black_king_position(self):
        """Test that black king is correctly positioned"""
        layer = create_piece_layer(self.board, chess.KING, chess.BLACK)
        self.assertEqual(np.sum(layer), 1)  # Only 1 king
        self.assertEqual(layer[7, 4], 1)  # King at e8 (rank 7, file 4)
    
    def test_empty_squares_for_nonexistent_pieces(self):
        """Test empty board positions have no pieces"""
        layer = create_piece_layer(self.board, chess.PAWN, chess.WHITE)
        # Ranks 2-7 should be empty for white pawns at start
        for rank in range(2, 8):
            self.assertEqual(np.sum(layer[rank, :]), 0)
    
    def test_after_move(self):
        """Test that layer updates after a move"""
        # Move white pawn from e2 to e4
        self.board.push_san("e4")
        layer = create_piece_layer(self.board, chess.PAWN, chess.WHITE)
        # Pawn should no longer be at e2 (rank 1, file 4)
        self.assertEqual(layer[1, 4], 0)
        # Pawn should be at e4 (rank 3, file 4)
        self.assertEqual(layer[3, 4], 1)


class TestMoveToIndex(unittest.TestCase):
    """Tests for the move_to_index function from translator.py"""
    
    def test_returns_integer(self):
        """Test that move_to_index returns an integer"""
        move = chess.Move(chess.E2, chess.E4)
        index = move_to_index(move)
        self.assertIsInstance(index, int)
    
    def test_range_0_to_4095(self):
        """Test that index is within valid range"""
        move = chess.Move(chess.E2, chess.E4)
        index = move_to_index(move)
        self.assertGreaterEqual(index, 0)
        self.assertLessEqual(index, 4095)
    
    def test_e2_to_e4(self):
        """Test specific move e2->e4"""
        move = chess.Move(chess.E2, chess.E4)  # from_square=12, to_square=28
        index = move_to_index(move)
        expected = 12 * 64 + 28  # 768 + 28 = 796
        self.assertEqual(index, expected)
    
    def test_a1_to_a1(self):
        """Test move from a1 to a1 (null move concept)"""
        move = chess.Move(chess.A1, chess.A1)  # from_square=0, to_square=0
        index = move_to_index(move)
        self.assertEqual(index, 0)
    
    def test_h8_to_h8(self):
        """Test move from h8 to h8"""
        move = chess.Move(chess.H8, chess.H8)  # from_square=63, to_square=63
        index = move_to_index(move)
        expected = 63 * 64 + 63  # 4032 + 63 = 4095
        self.assertEqual(index, expected)
    
    def test_different_moves_different_indices(self):
        """Test that different moves produce different indices"""
        move1 = chess.Move(chess.E2, chess.E4)
        move2 = chess.Move(chess.D2, chess.D4)
        index1 = move_to_index(move1)
        index2 = move_to_index(move2)
        self.assertNotEqual(index1, index2)


class TestIndexToMove(unittest.TestCase):
    """Tests for the index_to_move function from translator.py"""
    
    def test_returns_move(self):
        """Test that index_to_move returns a chess.Move object"""
        move = index_to_move(796)
        self.assertIsInstance(move, chess.Move)
    
    def test_index_0(self):
        """Test index 0 converts to a1->a1"""
        move = index_to_move(0)
        self.assertEqual(move.from_square, 0)
        self.assertEqual(move.to_square, 0)
    
    def test_index_4095(self):
        """Test index 4095 converts to h8->h8"""
        move = index_to_move(4095)
        self.assertEqual(move.from_square, 63)
        self.assertEqual(move.to_square, 63)
    
    def test_index_796(self):
        """Test index 796 converts to e2->e4"""
        move = index_to_move(796)
        self.assertEqual(move.from_square, 12)  # e2
        self.assertEqual(move.to_square, 28)    # e4
    
    def test_inverse_of_move_to_index(self):
        """Test that index_to_move is inverse of move_to_index"""
        original_move = chess.Move(chess.E2, chess.E4)
        index = move_to_index(original_move)
        reconstructed_move = index_to_move(index)
        self.assertEqual(original_move, reconstructed_move)
    
    def test_multiple_round_trips(self):
        """Test round trip conversion for multiple moves"""
        test_moves = [
            chess.Move(chess.A1, chess.H8),
            chess.Move(chess.E2, chess.E4),
            chess.Move(chess.G1, chess.F3),
            chess.Move(chess.D7, chess.D5),
        ]
        for original_move in test_moves:
            index = move_to_index(original_move)
            reconstructed_move = index_to_move(index)
            self.assertEqual(original_move, reconstructed_move)


class TestChessNet(unittest.TestCase):
    """Tests for the ChessNet neural network class"""
    
    def setUp(self):
        self.model = ChessNet()
    
    def test_initialization(self):
        """Test that ChessNet initializes properly"""
        self.assertIsInstance(self.model, ChessNet)
        self.assertIsInstance(self.model, torch.nn.Module)
    
    def test_has_conv_layer(self):
        """Test that model has conv1 layer"""
        self.assertTrue(hasattr(self.model, 'conv1'))
        self.assertIsInstance(self.model.conv1, torch.nn.Conv2d)
    
    def test_has_value_head(self):
        """Test that model has value_head layer"""
        self.assertTrue(hasattr(self.model, 'value_head'))
        self.assertIsInstance(self.model.value_head, torch.nn.Linear)
    
    def test_has_policy_head(self):
        """Test that model has policy_head layer"""
        self.assertTrue(hasattr(self.model, 'policy_head'))
        self.assertIsInstance(self.model.policy_head, torch.nn.Linear)
    
    def test_conv_layer_parameters(self):
        """Test conv1 layer has correct parameters"""
        self.assertEqual(self.model.conv1.in_channels, 17)
        self.assertEqual(self.model.conv1.out_channels, 20)
        self.assertEqual(self.model.conv1.kernel_size, (5, 5))
    
    def test_value_head_parameters(self):
        """Test value_head has correct parameters"""
        self.assertEqual(self.model.value_head.in_features, 320)
        self.assertEqual(self.model.value_head.out_features, 1)
    
    def test_policy_head_parameters(self):
        """Test policy_head has correct parameters"""
        self.assertEqual(self.model.policy_head.in_features, 320)
        self.assertEqual(self.model.policy_head.out_features, 4096)
    
    def test_forward_pass_single_input(self):
        """Test forward pass with single input"""
        # Create a random input tensor (batch_size=1, channels=17, height=8, width=8)
        x = torch.randn(1, 17, 8, 8)
        value, policy = self.model(x)
        
        # Check output shapes
        self.assertEqual(value.shape, (1, 1))
        self.assertEqual(policy.shape, (1, 4096))
    
    def test_forward_pass_batch_input(self):
        """Test forward pass with batch input"""
        # Create a batch of random inputs (batch_size=4, channels=17, height=8, width=8)
        x = torch.randn(4, 17, 8, 8)
        value, policy = self.model(x)
        
        # Check output shapes
        self.assertEqual(value.shape, (4, 1))
        self.assertEqual(policy.shape, (4, 4096))
    
    def test_value_output_range(self):
        """Test that value output is in range [-1, 1] due to tanh"""
        x = torch.randn(1, 17, 8, 8)
        value, _ = self.model(x)
        
        self.assertTrue(torch.all(value >= -1))
        self.assertTrue(torch.all(value <= 1))
    
    def test_policy_output_is_probability_distribution(self):
        """Test that policy output sums to 1 (probability distribution)"""
        x = torch.randn(1, 17, 8, 8)
        _, policy = self.model(x)
        
        # Policy should sum to ~1.0 for each batch item (due to softmax)
        policy_sum = torch.sum(policy, dim=1)
        self.assertTrue(torch.allclose(policy_sum, torch.ones_like(policy_sum), atol=1e-6))
    
    def test_policy_output_all_positive(self):
        """Test that all policy outputs are positive (probabilities)"""
        x = torch.randn(1, 17, 8, 8)
        _, policy = self.model(x)
        
        self.assertTrue(torch.all(policy >= 0))
    
    def test_forward_returns_two_outputs(self):
        """Test that forward returns exactly two outputs"""
        x = torch.randn(1, 17, 8, 8)
        outputs = self.model(x)
        
        self.assertEqual(len(outputs), 2)


if __name__ == '__main__':
    unittest.main()
# Copilot Instructions for CHSS

## Project Overview
This is a chess reinforcement learning (RL) project inspired by AlphaGo, implemented in Python. The goal is to create a neural network model that can learn to play chess through self-play and reinforcement learning.

## Tech Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch
- **Chess Engine**: python-chess library
- **Numerical Computing**: NumPy
- **Testing**: pytest
- **Linting**: flake8

## Project Structure
- `ChessNet.py`: Neural network architecture with value head and policy head
- `translator.py`: Functions to convert chess board states to numpy arrays and moves to indices
- `main.py`: Main script demonstrating board state encoding
- `test/`: Test directory containing unit tests
- `.github/workflows/python-app.yml`: CI/CD pipeline for testing and linting

## Code Architecture

### Board Representation
The chess board is represented as a 17-channel tensor (17, 8, 8):
- 12 channels for pieces (6 piece types × 2 colors)
- 1 channel for current turn (1 for white, 0 for black)
- 4 channels for castling rights (kingside/queenside for both colors)

### Neural Network
The `ChessNet` class implements a dual-head architecture:
- **Value Head**: Outputs a single value (win probability) using Tanh activation
- **Policy Head**: Outputs 4096 move probabilities (64 from squares × 64 to squares) using Softmax

### Move Encoding
Moves are encoded as indices from 0-4095 using the formula: `(from_square * 64) + to_square`

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Maximum line length: 127 characters
- Use type hints where appropriate
- Keep functions focused and well-documented

### Testing
- Write unit tests for all new functionality
- Use pytest for running tests
- Test files should be placed in the `test/` directory
- Run tests with: `pytest`

### Linting
- Use flake8 for code quality checks
- Critical errors (syntax, undefined names) must be fixed
- Run linting with: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`

### Dependencies
Install required packages:
```bash
pip install chess numpy torch pytest flake8
```

## Workflow
1. Make code changes
2. Run linting: `flake8 .`
3. Run tests: `pytest`
4. Ensure all checks pass before committing

## Important Notes
- The project uses chess coordinate system: row (rank) and col (file) from 0-7
- Board representation follows the python-chess library conventions
- The network expects input shape (batch_size, 17, 8, 8)
- Moves are encoded as integers from 0-4095 for the policy head output

# CHSS - Chess Reinforcement Learning

[![Python application](https://github.com/yava-code/chss/actions/workflows/python-app.yml/badge.svg)](https://github.com/yava-code/chss/actions/workflows/python-app.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)

My attempt at creating a chess RL model, an analog of AlphaGo. I know that there are already many projects like this, but I tried to make it on my own for experience and fun.

I hope for your feedback and contributions. If you want to collaborate or have any questions, feel free to reach out - vyaseno1@stu.vistula.edu.pl

## Features

- Neural network architecture with value and policy heads
- Board state encoding to 17-channel tensor
- Move encoding/decoding system
- Comprehensive test suite
- Docker support for development
- CI/CD pipeline with GitHub Actions
- Project documentation on GitHub Pages

## Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch
- **Chess Engine**: python-chess library
- **Numerical Computing**: NumPy
- **Testing**: pytest
- **Linting**: flake8

## Quick Start

### Using Docker Compose

```bash
# Run tests
docker-compose up
```

### Using Docker

```bash
# Build image
docker build -t chss .

# Run tests
docker run chss
```

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
PYTHONPATH=. pytest -v

# Lint code
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Architecture

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

## Project Structure

```
chss/
├── ChessNet.py         # Neural network architecture
├── translator.py       # Board state and move encoding
├── main.py            # Example usage
├── test/              # Test suite
│   └── test_main.py
├── requirements.txt   # Python dependencies
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Docker Compose setup
├── docs/             # GitHub Pages documentation
│   └── index.html
└── .github/
    └── workflows/
        ├── python-app.yml # CI/CD pipeline
        └── pages.yml      # GitHub Pages deployment
```

## Documentation

Full documentation is available at [https://yava-code.github.io/chss/](https://yava-code.github.io/chss/)

## Development

```bash
# Clone the repository
git clone https://github.com/yava-code/chss.git
cd chss

# Install dependencies
pip install -r requirements.txt

# Run tests
PYTHONPATH=. pytest -v

# Run linting
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Run main example
PYTHONPATH=. python main.py
```

## CI/CD

The project uses GitHub Actions for continuous integration:
- Automated testing on push and pull requests
- Code linting with flake8
- Automatic deployment of documentation to GitHub Pages

## License

This project is open source and available for educational purposes.

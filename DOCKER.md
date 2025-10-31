# CHSS Docker Setup

This directory contains Docker configuration for the CHSS project.

## Quick Start

### Using Docker Compose

Run tests:
```bash
docker-compose up
```

### Using Docker Directly

Build the image:
```bash
docker build -t chss .
```

Run tests:
```bash
docker run chss
```

Run with custom command:
```bash
docker run -it chss bash
```

## Development

Mount your local directory for live development:
```bash
docker run -v $(pwd):/app -it chss bash
```

## Environment Variables

- `PYTHONPATH=/app` - Set automatically to ensure imports work correctly

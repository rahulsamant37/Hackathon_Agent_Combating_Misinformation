# AI-Powered Misinformation Detection Tool

An intelligent tool for analyzing and detecting misinformation in text, images, and URLs using advanced AI models and LangChain/LangGraph workflows.

## Features

- **Multi-format Analysis**: Analyze text content, images, and URLs for misinformation indicators
- **AI-Powered Detection**: Uses Gemini and Groq models through LangChain/LangGraph orchestration
- **Risk Assessment**: Provides detailed risk scores and explanations
- **Educational Resources**: Integrated learning modules to improve misinformation detection skills
- **Real-time Processing**: Fast analysis with progress tracking and user feedback

## Architecture

- **Frontend**: Streamlit web application
- **Backend**: FastAPI REST API
- **LLM Integration**: LangChain pipelines with LangGraph workflow orchestration
- **Models**: Google Gemini and Groq support with provider abstraction

## Quick Start

### Prerequisites

- Python 3.9+
- API keys for Gemini or Groq (or both)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd misinformation-detection-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application:
```bash
# Start FastAPI backend
python -m api.main

# Start Streamlit frontend (in another terminal)
streamlit run streamlit_app/app.py
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Configuration

The application uses both environment variables and a YAML configuration file:

- **Environment Variables**: API keys, deployment settings (see `.env.example`)
- **Configuration File**: `config/config.yaml` for application settings

## Project Structure

```
misinformation-detection-tool/
├── api/                    # FastAPI backend
├── streamlit_app/          # Streamlit frontend
├── llm/                    # LLM provider integrations
├── graph/                  # LangGraph workflows
├── services/               # Business logic services
├── utils/                  # Utility functions
├── config/                 # Configuration management
├── static/                 # Static assets
└── templates/              # HTML templates
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## API Documentation

Once the FastAPI backend is running, visit:
- API Documentation: http://localhost:8000/docs
- Alternative Documentation: http://localhost:8000/redoc

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on the GitHub repository.
# Agentic Knowledge System

A powerful multi-agent system for knowledge retrieval and question answering, built with AutoGen and FastAPI.

## Features

- 🤖 Multi-agent system with specialized agents for:
  - Confluence knowledge base
  - Databricks SQL and system tables
  - GraphQL APIs
- 🔍 Vector-based semantic search
- 💬 Natural language to SQL/GraphQL conversion
- 📝 Persistent conversation history
- 🚀 Production-ready FastAPI backend
- 🔄 AutoGen group chat patterns
- 🛠️ Tool-calling capabilities
- 📊 Structured database storage

## Quick Start

### Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) (recommended) or pip
- PostgreSQL (for production) or SQLite (for local development)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-knowledge-system.git
cd agentic-knowledge-system
```

2. Install dependencies using UV (recommended):
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

Or using pip:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure the environment:
```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
```

### Configuration

The system supports two modes:

1. **Local Development** (`config/local_config.yaml`):
   - Uses SQLite for conversation storage
   - In-memory embeddings
   - Local file-based logging
   - Development-friendly settings

2. **Production** (`config/production_config.yaml`):
   - PostgreSQL with pgvector for storage
   - Bedrock or Gemini for LLM
   - Structured logging
   - Production-ready settings

### Running the Application

1. Start the FastAPI server:
```bash
python src/run.py
```

The API will be available at `http://localhost:8000`

## API Usage

### Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the recent changes in the sales data?",
    "user_id": "user123",
    "conversation_id": "conv456"
  }'
```

### Conversation Management
```bash
# Get recent conversations
curl http://localhost:8000/conversations/user123

# Get conversation history
curl http://localhost:8000/conversations/conv456/history

# Clear conversation
curl -X DELETE http://localhost:8000/conversations/conv456
```

### Search
```bash
curl "http://localhost:8000/search?query=sales%20data&user_id=user123&limit=5"
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Development

### Project Structure
```
.
├── config/
│   ├── local_config.yaml
│   └── production_config.yaml
├── src/
│   ├── api/
│   │   └── app.py
│   ├── core/
│   │   ├── agents/
│   │   │   ├── base_agent.py
│   │   │   ├── supervisor_agent.py
│   │   │   ├── confluence_agent.py
│   │   │   ├── databricks_agent.py
│   │   │   └── graphql_agent.py
│   │   ├── memory.py
│   │   └── embeddings.py
│   └── run.py
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

### Adding New Agents

1. Create a new agent class in `src/core/agents/`
2. Inherit from `BaseAgent`
3. Implement required methods:
   - `_get_system_message()`
   - `_register_tools()`
   - `process_message()`

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 
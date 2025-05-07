# Agentic Knowledge System

A powerful multi-agent system built with AutoGen and LangChain for intelligent knowledge retrieval and analysis across multiple data sources.

## Features

- **Multi-Agent Architecture**: Specialized agents for different data sources (Confluence, Databricks, GraphQL)
- **Intelligent Retrieval**: Vector-based semantic search using pgvector/SQLite
- **Group Chat Pattern**: Collaborative problem-solving using AutoGen's group chat
- **State Management**: Robust state tracking using dataclasses
- **Production Ready**: Configurable for both local development and production environments

## Components

### Core Agents

1. **Supervisor Agent**
   - Coordinates between specialized agents
   - Manages conversation flow
   - Synthesizes responses from multiple sources

2. **Confluence Agent**
   - Searches and analyzes Confluence content
   - Uses LangChain retrievers for efficient content retrieval
   - Maintains vector embeddings of content

3. **Databricks Agent**
   - Queries and analyzes Databricks data
   - Schema-aware query generation
   - Vector-based schema search

4. **GraphQL Agent**
   - Queries and analyzes GraphQL APIs
   - Schema validation and optimization
   - Intelligent query construction

### Storage

- **Vector Storage**:
  - Production: pgvector (PostgreSQL)
  - Local: SQLite with vector support
- **Conversation Storage**:
  - Production: PostgreSQL
  - Local: SQLite

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/agentic-knowledge-system.git
cd agentic-knowledge-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

The system supports two configuration modes:

### Local Development
- Uses SQLite for storage
- OpenAI for LLM
- Local file-based logging
- Debug mode enabled

### Production
- Uses PostgreSQL with pgvector
- AWS Bedrock for LLM
- Production-grade monitoring
- Rate limiting and security features

## Usage

1. Start the API server:
```bash
python src/run.py
```

2. The API will be available at:
- Local: http://localhost:8000
- Production: https://your-domain.com

### API Endpoints

- `POST /chat`: Send messages to the agent system
- `GET /conversations`: List recent conversations
- `GET /search`: Search across all data sources
- `GET /health`: Health check endpoint

## Development

### Project Structure
```
.
├── config/
│   ├── local_config.yaml
│   └── production_config.yaml
├── src/
│   ├── core/
│   │   ├── agents/
│   │   ├── memory.py
│   │   └── config.py
│   ├── api/
│   └── run.py
├── tests/
├── data/
└── logs/
```

### Adding New Agents

1. Create a new agent class in `src/core/agents/`
2. Implement required methods:
   - `_create_group_chat()`
   - `_register_tools()`
   - `process_message()`
3. Add configuration in `config/*.yaml`

## Monitoring

### Local
- Basic health checks
- Log file monitoring

### Production
- Prometheus metrics
- Jaeger tracing
- Rate limiting
- Health checks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 
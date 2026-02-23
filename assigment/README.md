# Agentic SQL Query Generator

A modular, self-correcting AI agent that generates SQL queries from natural language questions based on database schema. Built for the Senior Applied Scientist take-home assignment.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│  Table Retriever │────▶│ SQL Generator   │
│ (Natural Lang)  │     │ (FAISS + Embed)  │     │ (LLM + Valid)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                         │
                                ▼                         ▼
                        ┌─────────────────────────────────────┐
                        │     Self-Correction Loop            │
                        │  (Validation + Retry Logic)         │
                        └─────────────────────────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │   FastAPI API   │
                                                │ /generate-sql   │
                                                └─────────────────┘
```

## System Design

### Part 1: Agent Architecture & Retrieval Design

- **Modular Design**: Separate components for retrieval, generation, and validation
- **Self-Correction**: Automatic retry with expanded context on failure
- **Confidence Scoring**: Multi-factor scoring based on retrieval relevance, schema coverage, and SQL validity

### Part 2: Production API

- **FastAPI** with clean request/response contracts
- **Observability**: Latency tracking, logging, and metrics
- **Performance Target**: <20s retrieval latency (excluding LLM inference)

### Part 3: Fine-Tuning Analysis

See [Fine-Tuning Analysis](#fine-tuning-analysis) section below.

## Quick Start

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

Edit `config.yaml` to set your preferences:

```yaml
llm:
  provider: "ollama"  # or "openai"
  model: "llama3"     # or "gpt-4o-mini"
```

For OpenAI, uncomment and set:
```yaml
llm:
  openai_api_key: "${OPENAI_API_KEY}"
```

### 4. Run the API

```bash
cd src
python -m api.main
```

Or with uvicorn:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate SQL
curl -X POST http://localhost:8000/generate-sql \
  -H "Content-Type: application/json" \
  -d '{"question": "How many open tickets are there?"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/metrics` | GET | Agent and API metrics |
| `/generate-sql` | POST | Generate SQL from question |
| `/reset-metrics` | POST | Reset metrics |

## Example Requests

### Generate SQL Query

```bash
curl -X POST http://localhost:8000/generate-sql \
  -H "Content-Type: application/json" \
  -d '{
    "question": "List all high priority tickets from the last week"
  }'
```

### Response

```json
{
  "sql": "SELECT * FROM tickets_tbl WHERE priority = 'high' AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)",
  "confidence": 0.8,
  "tables_used": ["tickets_tbl"],
  "retrieval_time_ms": 150.5,
  "generation_time_ms": 200.3,
  "total_time_ms": 350.8,
  "self_correction_attempts": 1,
  "is_valid": true,
  "warnings": []
}
```

## Sample Scenarios

The system is tested against these scenarios from the sample data:

1. "How many open tickets are there?"
2. "List all high priority tickets from the last week"
3. "Which customers have the most unresolved tickets?"
4. "Show me the average resolution time by agent"
5. "Find the most common ticket categories"
6. "What is the customer satisfaction rating for resolved tickets?"
7. "List agents with their department and active ticket count"
8. "Show tickets created this month by region"

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| AI/LLM | LangChain + Ollama/OpenAI |
| Embeddings | sentence-transformers |
| Vector Store | FAISS |
| Database | SQLite |
| Logging | Loguru |

## Self-Correction Mechanism

The agent implements self-correction through:

1. **Schema Validation**: Checks if SQL uses valid tables/columns
2. **Confidence Threshold**: Requires minimum confidence score (0.7 by default)
3. **Retry Logic**: Up to 2 retries with expanded table context
4. **Fallback**: Uses all tables as context if retrieval fails

### Failure Detection

| Signal | Detection Method |
|--------|------------------|
| Low retrieval relevance | similarity threshold check |
| Invalid column names | schema validator |
| Incomplete SQL | syntax checker |
| Ambiguous intent | confidence scorer |

## Observability

### Metrics Tracked

- Total/successful/failed requests
- Average retrieval/generation latency
- Self-correction attempts
- Confidence scores

### Logging

- Request/response logging
- Component-level timing
- Error tracking

## Configuration

All configuration is in `config.yaml`:

```yaml
app:
  name: "Agentic SQL Generator"
  version: "1.0.0"

llm:
  provider: "ollama"
  model: "llama3"
  temperature: 0.1

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"

retrieval:
  top_k: 5
  similarity_threshold: 0.5

agent:
  max_retries: 2
  confidence_threshold: 0.7

api:
  host: "0.0.0.0"
  port: 8000
```

## Assumptions & Limitations

### Assumptions
- CPU-only execution environment
- Read-only SQL queries (SELECT only)
- SQLite-compatible SQL syntax

### Limitations
- Simplified SQL generation (may not handle complex JOINs)
- Mock LLM used if Ollama/OpenAI unavailable
- No query execution (only SQL generation)
- Basic self-correction (not exhaustive)

## Fine-Tuning Analysis

### Can Fine-Tuning Improve Model Accuracy & Reduce Response Time?

**Yes, but with caveats:**

#### When Fine-Tuning Helps:
1. **Domain-Specific Vocabulary**: Fine-tuned models understand table/column names better
2. **Schema Grounding**: Reduced hallucinations for specific database schemas
3. **Output Format**: Consistent SQL formatting and structure

#### Trade-offs:
| Factor | Fine-Tuning | RAG + Prompt Engineering |
|--------|-------------|---------------------------|
| Development Time | High (data collection, training) | Low (immediate) |
| Maintenance | Higher (retraining needed) | Lower (update prompts) |
| Flexibility | Lower (schema-bound) | Higher (dynamic schemas) |
| Cost | Higher (GPU training) | Lower (inference only) |

#### Recommended Approach:
1. **Start with RAG + Prompt Engineering** - Most cost-effective
2. **Collect Production Query Logs** - After 1000+ queries
3. **Fine-tune if**: 
   - Consistent schema (not changing often)
   - High volume of similar queries
   - Latency critical (smaller fine-tuned model)

#### Alternative: Smaller Model
Using a smaller, fine-tuned model (e.g., CodeGen-2B) can reduce latency while maintaining accuracy for SQL-specific tasks.

## Future Improvements

With more time, we would add:

1. **Query Caching**: Cache similar queries for faster responses
2. **Async Parallel Retrieval**: Fetch multiple tables simultaneously
3. **Schema Graph Reasoning**: Understand table relationships better
4. **Feedback Learning Loop**: Learn from user corrections
5. **Evaluation Benchmark**: Automated testing against known queries
6. **Fine-tuned SQL Model**: Domain-specific model for better accuracy

## Docker Support

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t sql-agent .
docker run -p 8000:8000 sql-agent
```

## License

MIT License

## Contact

For questions about this implementation, please refer to the assignment documentation.

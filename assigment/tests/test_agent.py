"""Test script for the SQL Agent."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config
from src.data.schema_loader import SchemaLoader
from src.agent.sql_generator import MockLLM, SQLGenerator, SQLValidator
from src.agent.embeddings import EmbeddingModel, TableRetriever
from src.agent.sql_agent import SQLAgent


def test_schema_loader():
    """Test schema loader."""
    print("=" * 50)
    print("Testing Schema Loader...")
    print("=" * 50)
    
    schema_path = "data/sample_schema.json"
    loader = SchemaLoader(schema_path)
    
    print(f"Loaded {len(loader.get_table_names())} tables:")
    for name in loader.get_table_names():
        print(f"  - {name}")
    
    print(f"\nLoaded {len(loader.get_scenarios())} scenarios")
    
    return loader


def test_retriever(loader, embedding_model):
    """Test table retriever."""
    print("\n" + "=" * 50)
    print("Testing Table Retriever...")
    print("=" * 50)
    
    retriever = TableRetriever(embedding_model, loader)
    
    test_questions = [
        "How many open tickets are there?",
        "Which customers have the most tickets?",
        "Show me agent performance metrics"
    ]
    
    for question in test_questions:
        tables, context = retriever.retrieve_with_context(question, top_k=3)
        print(f"\nQuestion: {question}")
        print(f"Retrieved tables: {tables}")
    
    return retriever


def test_sql_generator(loader):
    """Test SQL generator."""
    print("\n" + "=" * 50)
    print("Testing SQL Generator...")
    print("=" * 50)
    
    # Use Mock LLM for testing
    llm = MockLLM()
    generator = SQLGenerator(llm, loader)
    
    test_questions = [
        "How many open tickets are there?",
        "List all high priority tickets",
        "Show customers with their ticket count"
    ]
    
    for question in test_questions:
        context = loader.get_full_context()
        result = generator.generate(question, context, loader.get_table_names())
        print(f"\nQuestion: {question}")
        print(f"SQL: {result.sql}")
        print(f"Valid: {result.is_valid}, Confidence: {result.confidence}")
    
    return generator


def test_full_agent(config, loader):
    """Test full agent."""
    print("\n" + "=" * 50)
    print("Testing Full Agent...")
    print("=" * 50)
    
    # Use Mock LLM
    from src.agent.sql_generator import MockLLM
    from src.agent.embeddings import EmbeddingModel
    
    llm = MockLLM()
    
    # Create a simple embedding model
    try:
        embedding_model = EmbeddingModel(
            model_name=config.embedding_model,
            device="cpu"
        )
    except Exception as e:
        print(f"Warning: Could not load embedding model: {e}")
        print("Using simple mock retriever...")
        return None
    
    from src.agent.sql_generator import SQLGenerator
    from src.agent.sql_agent import SQLAgent
    
    sql_gen = SQLGenerator(llm, loader)
    agent = SQLAgent(loader, embedding_model, sql_gen, config)
    
    # Test scenarios
    scenarios = loader.get_scenarios()[:3]  # Test first 3
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario.question}")
        result = agent.process(scenario.question)
        print(f"SQL: {result.sql}")
        print(f"Tables used: {result.tables_used}")
        print(f"Confidence: {result.confidence}")
        print(f"Valid: {result.is_valid}")
    
    # Print metrics
    print("\n" + "-" * 30)
    print("Agent Metrics:")
    metrics = agent.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return agent


def main():
    """Run all tests."""
    print("Starting SQL Agent Tests\n")
    
    # Load config
    config = Config("config.yaml")
    print(f"App: {config.app_name} v{config.app_version}")
    
    # Test 1: Schema Loader
    loader = test_schema_loader()
    
    # Test 2: Retriever (without full embedding model)
    print("\nSkipping embedding tests (requires model download)")
    print("Using MockLLM for testing...")
    
    # Test 3: SQL Generator
    test_sql_generator(loader)
    
    # Test 4: Full Agent
    try:
        test_full_agent(config, loader)
    except Exception as e:
        print(f"Full agent test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Tests Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

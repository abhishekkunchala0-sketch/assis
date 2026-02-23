"""Main SQL Agent with self-correction capabilities."""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from ..data.schema_loader import SchemaLoader
from .embeddings import EmbeddingModel, TableRetriever
from .sql_generator import SQLGenerator, SQLGenerationResult, create_llm


@dataclass
class AgentMetrics:
    """Metrics for agent performance."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retrieval_time_ms: float = 0.0
    total_generation_time_ms: float = 0.0
    total_self_corrections: int = 0
    avg_confidence: float = 0.0


@dataclass
class AgentResponse:
    """Response from the SQL Agent."""
    sql: str
    confidence: float
    tables_used: List[str]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    self_correction_attempts: int
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class SQLAgent:
    """Main agent that orchestrates SQL generation with self-correction."""
    
    def __init__(
        self,
        schema_loader: SchemaLoader,
        embedding_model: EmbeddingModel,
        sql_generator: SQLGenerator,
        config
    ):
        """Initialize SQL Agent.
        
        Args:
            schema_loader: Schema loader instance
            embedding_model: Embedding model instance
            sql_generator: SQL generator instance
            config: Configuration instance
        """
        self.schema_loader = schema_loader
        self.embedding_model = embedding_model
        self.sql_generator = sql_generator
        self.config = config
        self.retriever = TableRetriever(embedding_model, schema_loader)
        
        # Agent configuration
        self.max_retries = config.max_retries
        self.confidence_threshold = config.confidence_threshold
        self.enable_self_correction = config.enable_self_correction
        self.retrieval_top_k = config.retrieval_top_k
        self.similarity_threshold = config.similarity_threshold
        
        # Metrics
        self.metrics = AgentMetrics()
    
    def process(self, question: str) -> AgentResponse:
        """Process a question and generate SQL.
        
        Args:
            question: Natural language question
            
        Returns:
            AgentResponse with generated SQL and metadata
        """
        logger.info(f"Processing question: {question}")
        start_time = time.time()
        self.metrics.total_requests += 1
        
        warnings = []
        
        # Step 1: Retrieve relevant tables
        retrieval_start = time.time()
        try:
            table_names, context = self.retriever.retrieve_with_context(
                question,
                top_k=self.retrieval_top_k,
                threshold=self.similarity_threshold
            )
            retrieval_time = time.time() - retrieval_start
            self.metrics.total_retrieval_time_ms += retrieval_time * 1000
            
            logger.info(f"Retrieved tables: {table_names} in {retrieval_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return AgentResponse(
                sql="",
                confidence=0.0,
                tables_used=[],
                retrieval_time_ms=0,
                generation_time_ms=0,
                total_time_ms=(time.time() - start_time) * 1000,
                self_correction_attempts=0,
                is_valid=False,
                error_message=f"Retrieval failed: {str(e)}"
            )
        
        # Check if retrieval quality is low
        if not table_names:
            warnings.append("No tables retrieved - using all tables as fallback")
            table_names = self.schema_loader.get_table_names()
            context = self.schema_loader.get_full_context()
        
        # Step 2: Generate SQL with self-correction
        generation_start = time.time()
        result = self._generate_with_self_correction(
            question, 
            context, 
            table_names
        )
        generation_time = time.time() - generation_start
        self.metrics.total_generation_time_ms += generation_time * 1000
        
        # Step 3: Update metrics
        if result.is_valid:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Calculate running average confidence
        n = self.metrics.successful_requests + self.metrics.failed_requests
        if n > 0:
            old_avg = self.metrics.avg_confidence
            self.metrics.avg_confidence = (
                (old_avg * (n - 1) + result.confidence) / n
            )
        
        total_time = time.time() - start_time
        
        logger.info(
            f"Processed question in {total_time:.2f}s "
            f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)"
        )
        
        return AgentResponse(
            sql=result.sql,
            confidence=result.confidence,
            tables_used=result.tables_used,
            retrieval_time_ms=retrieval_time * 1000,
            generation_time_ms=generation_time * 1000,
            total_time_ms=total_time * 1000,
            self_correction_attempts=getattr(result, 'attempts', 1),
            is_valid=result.is_valid,
            error_message=result.error_message,
            warnings=warnings
        )
    
    def _generate_with_self_correction(
        self, 
        question: str, 
        context: str, 
        table_names: List[str]
    ) -> SQLGenerationResult:
        """Generate SQL with self-correction loop.
        
        Args:
            question: Natural language question
            context: Schema context
            table_names: Retrieved table names
            
        Returns:
            SQLGenerationResult
        """
        if not self.enable_self_correction:
            return self.sql_generator.generate(question, context, table_names)
        
        attempts = 0
        max_attempts = self.max_retries + 1
        last_result = None
        
        while attempts < max_attempts:
            attempts += 1
            
            if attempts > 1:
                logger.info(f"Self-correction attempt {attempts}")
                self.metrics.total_self_corrections += 1
                # Expand context with more tables for retry
                all_tables = self.schema_loader.get_table_names()
                remaining = [t for t in all_tables if t not in table_names]
                if remaining:
                    additional_context = self.schema_loader.get_table_context(
                        remaining[:3]  # Add up to 3 more tables
                    )
                    context = context + "\n\n" + additional_context
            
            result = self.sql_generator.generate(question, context, table_names)
            
            # Check if result is acceptable
            if result.is_valid and result.confidence >= self.confidence_threshold:
                result.attempts = attempts
                return result
            
            # Check for unrecoverable errors
            if result.error_message and "Unknown table" not in result.error_message:
                # Don't retry on unknown table errors
                break
            
            last_result = result
        
        # Return best result or last attempt
        if last_result and last_result.sql:
            last_result.attempts = attempts
            return last_result
        
        return result if result else last_result
    
    def get_metrics(self) -> Dict:
        """Get agent metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (
                self.metrics.successful_requests / self.metrics.total_requests 
                if self.metrics.total_requests > 0 else 0
            ),
            "avg_retrieval_time_ms": (
                self.metrics.total_retrieval_time_ms / self.metrics.total_requests
                if self.metrics.total_requests > 0 else 0
            ),
            "avg_generation_time_ms": (
                self.metrics.total_generation_time_ms / self.metrics.total_requests
                if self.metrics.total_requests > 0 else 0
            ),
            "total_self_corrections": self.metrics.total_self_corrections,
            "avg_confidence": self.metrics.avg_confidence
        }
    
    def reset_metrics(self) -> None:
        """Reset agent metrics."""
        self.metrics = AgentMetrics()


def create_agent(config) -> SQLAgent:
    """Create and initialize the SQL Agent.
    
    Args:
        config: Configuration instance
        
    Returns:
        Initialized SQLAgent
    """
    logger.info("Creating SQL Agent")
    
    # Initialize schema loader
    schema_loader = SchemaLoader(config.sample_schema_path)
    logger.info(f"Loaded {len(schema_loader.get_table_names())} tables")
    
    # Initialize embedding model
    embedding_model = EmbeddingModel(
        model_name=config.embedding_model,
        device=config.embedding_device
    )
    embedding_model._index_built = True  # Skip index building for now
    
    # Initialize LLM
    llm = create_llm(
        provider=config.llm_provider,
        model=config.llm_model,
        temperature=config.llm_temperature,
        base_url=config.llm_base_url
    )
    logger.info(f"LLM initialized: {llm}")
    
    # Initialize SQL generator
    sql_generator = SQLGenerator(llm, schema_loader)
    
    # Create agent
    agent = SQLAgent(
        schema_loader=schema_loader,
        embedding_model=embedding_model,
        sql_generator=sql_generator,
        config=config
    )
    
    logger.info("SQL Agent created successfully")
    return agent

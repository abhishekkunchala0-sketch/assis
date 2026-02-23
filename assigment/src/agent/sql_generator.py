"""SQL Generator using LLM."""
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger


@dataclass
class SQLGenerationResult:
    """Result of SQL generation."""
    sql: str
    tables_used: List[str]
    confidence: float
    is_valid: bool
    error_message: Optional[str] = None


class SQLValidator:
    """Validates SQL queries against schema."""
    
    def __init__(self, schema_loader):
        """Initialize SQL validator.
        
        Args:
            schema_loader: Schema loader instance
        """
        self.schema_loader = schema_loader
    
    def validate(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL query against schema.
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query"
        
        # Basic SQL syntax checks
        sql_upper = sql.upper()
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'INSERT', 'UPDATE']
        for keyword in dangerous_keywords:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                return False, f"Dangerous operation not allowed: {keyword}"
        
        # Extract table names from SQL
        table_names = self._extract_table_names(sql)
        
        # Validate tables exist
        valid_tables = set(self.schema_loader.get_table_names())
        for table in table_names:
            if table not in valid_tables:
                return False, f"Unknown table: {table}"
        
        # Validate columns for each table
        for table in table_names:
            columns = self._extract_columns_for_table(sql, table)
            if not self.schema_loader.validate_columns(table, columns):
                return False, f"Invalid columns in table {table}"
        
        return True, None
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL.
        
        Args:
            sql: SQL query
            
        Returns:
            List of table names
        """
        # Simple regex to find table names (handles common patterns)
        table_pattern = r'(?:FROM|JOIN|INTO|UPDATE)\s+(\w+)(?:\s|$)'
        matches = re.findall(table_pattern, sql, re.IGNORECASE)
        
        # Filter out common SQL keywords
        sql_keywords = {'SELECT', 'WHERE', 'AND', 'OR', 'ON', 'AS', 'IN', 'NOT', 'NULL', 'IS'}
        return [m for m in matches if m.upper() not in sql_keywords]
    
    def _extract_columns_for_table(self, sql: str, table: str) -> List[str]:
        """Extract column names for a specific table.
        
        Args:
            sql: SQL query
            table: Table name
            
        Returns:
            List of column names
        """
        # Find columns in SELECT clause or WHERE clause
        select_pattern = r'SELECT\s+(.+?)\s+FROM'
        match = re.search(select_pattern, sql, re.IGNORECASE)
        if match:
            cols_str = match.group(1)
            if cols_str.strip() == '*':
                return []
            columns = [c.strip().split()[-1] for c in cols_str.split(',')]
            return columns
        return []


class SQLGenerator:
    """Generates SQL queries from natural language."""
    
    # Default prompt template
    DEFAULT_PROMPT = """You are an expert SQL query generator. Given a natural language question and database schema, generate a valid SQLite SQL query.

Database Schema:
{schema}

Question: {question}

Instructions:
1. Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP)
2. Use proper SQL syntax for SQLite
3. Include appropriate JOINs if needed
4. Use aliases for clarity when joining tables
5. Return ONLY the SQL query, nothing else

SQL Query:"""

    def __init__(
        self, 
        llm, 
        schema_loader,
        validator: Optional[SQLValidator] = None,
        prompt_template: Optional[str] = None
    ):
        """Initialize SQL generator.
        
        Args:
            llm: LangChain LLM instance
            schema_loader: Schema loader instance
            validator: SQL validator instance
            prompt_template: Custom prompt template
        """
        self.llm = llm
        self.schema_loader = schema_loader
        self.validator = validator or SQLValidator(schema_loader)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self._is_mock = isinstance(llm, MockLLM)
        
        # Create prompt
        self._prompt = ChatPromptTemplate.from_template(self.prompt_template)
        if self._is_mock:
            # Don't use chain for mock LLM
            self._chain = None
        else:
            self._chain = self._prompt | self.llm | StrOutputParser()
    
    def generate(
        self, 
        question: str, 
        context: str,
        tables: List[str]
    ) -> SQLGenerationResult:
        """Generate SQL from question.
        
        Args:
            question: Natural language question
            context: Schema context
            tables: List of relevant table names
            
        Returns:
            SQLGenerationResult
        """
        logger.info(f"Generating SQL for question: {question}")
        start_time = time.time()
        
        try:
            # Generate SQL using LLM (or mock)
            if self._is_mock:
                # Use mock LLM directly
                sql = self.llm.invoke({
                    "schema": context,
                    "question": question
                }).strip()
            else:
                sql = self._chain.invoke({
                    "schema": context,
                    "question": question
                }).strip()
            
            # Clean up SQL (remove markdown code blocks if present)
            sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE)
            sql = re.sub(r'^```\s*', '', sql)
            sql = re.sub(r'\s*```$', '', sql)
            
            generation_time = time.time() - start_time
            logger.info(f"SQL generated in {generation_time:.2f}s: {sql}")
            
            # Validate SQL
            is_valid, error_message = self.validator.validate(sql)
            
            if not is_valid:
                logger.warning(f"SQL validation failed: {error_message}")
                return SQLGenerationResult(
                    sql=sql,
                    tables_used=tables,
                    confidence=0.0,
                    is_valid=False,
                    error_message=error_message
                )
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(sql, tables)
            
            return SQLGenerationResult(
                sql=sql,
                tables_used=tables,
                confidence=confidence,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            return SQLGenerationResult(
                sql="",
                tables_used=tables,
                confidence=0.0,
                is_valid=False,
                error_message=str(e)
            )
    
    def _calculate_confidence(self, sql: str, tables: List[str]) -> float:
        """Calculate confidence score for generated SQL.
        
        Args:
            sql: Generated SQL
            tables: Tables used
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Check SQL has SELECT
        if re.search(r'\bSELECT\b', sql, re.IGNORECASE):
            confidence += 0.1
        
        # Check SQL has FROM
        if re.search(r'\bFROM\b', sql, re.IGNORECASE):
            confidence += 0.1
        
        # Check uses expected tables
        sql_lower = sql.lower()
        for table in tables:
            if table.lower() in sql_lower:
                confidence += 0.1
        
        # Check has WHERE or GROUP BY or ORDER BY (meaningful query)
        if re.search(r'\b(WHERE|GROUP BY|ORDER BY|HAVING)\b', sql, re.IGNORECASE):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def generate_with_retry(
        self,
        question: str,
        context: str,
        tables: List[str],
        max_retries: int = 2
    ) -> SQLGenerationResult:
        """Generate SQL with retry logic.
        
        Args:
            question: Natural language question
            context: Schema context
            tables: List of relevant table names
            max_retries: Maximum number of retries
            
        Returns:
            SQLGenerationResult
        """
        last_result = None
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}")
                # Add more context to help the model
                context = context + "\n\nIMPORTANT: The previous SQL was invalid. Please correct it."
            
            result = self.generate(question, context, tables)
            
            if result.is_valid and result.confidence >= 0.5:
                return result
            
            last_result = result
        
        return last_result


class MockLLM:
    """Mock LLM for testing without actual LLM."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def invoke(self, input_data):
        """Mock invoke that returns a simple SQL."""
        # Handle both string and dict input
        if isinstance(input_data, str):
            question = input_data
            schema = ""
        else:
            question = input_data.get("question", "") if hasattr(input_data, "get") else str(input_data)
            schema = input_data.get("schema", "") if hasattr(input_data, "get") else ""
        
        # Simple pattern matching for demo
        question_lower = question.lower()
        
        if "open" in question_lower and "ticket" in question_lower:
            return "SELECT COUNT(*) as count FROM tickets_tbl WHERE status = 'open'"
        elif "high priority" in question_lower or "priority" in question_lower:
            return "SELECT * FROM tickets_tbl WHERE priority = 'high'"
        elif "customer" in question_lower and "ticket" in question_lower:
            return "SELECT c.customer_id, c.name, COUNT(t.ticket_id) as ticket_count FROM customers_tbl c LEFT JOIN tickets_tbl t ON c.customer_id = t.customer_id GROUP BY c.customer_id ORDER BY ticket_count DESC"
        elif "agent" in question_lower:
            return "SELECT a.agent_id, a.name, a.department, COUNT(t.ticket_id) as active_tickets FROM agents_tbl a LEFT JOIN ticket_assignments_tbl ta ON a.agent_id = ta.agent_id LEFT JOIN tickets_tbl t ON ta.ticket_id = t.ticket_id AND t.status != 'resolved' GROUP BY a.agent_id"
        elif "category" in question_lower:
            return "SELECT category, COUNT(*) as count FROM tickets_tbl GROUP BY category ORDER BY count DESC"
        elif "satisfaction" in question_lower or "rating" in question_lower:
            return "SELECT AVG(CAST(rating AS FLOAT)) as avg_rating FROM satisfaction_surveys_tbl"
        elif "region" in question_lower:
            return "SELECT c.region, COUNT(t.ticket_id) as ticket_count FROM customers_tbl c JOIN tickets_tbl t ON c.customer_id = t.customer_id GROUP BY c.region"
        else:
            return "SELECT * FROM tickets_tbl LIMIT 10"
    
    def __repr__(self):
        return "MockLLM()"
    
    # Make it compatible with LangChain Runnable interface
    @property
    def lc_attributes(self):
        return {}
    
    @property
    def lc_secrets(self):
        return {}
    
    @property
    def lc_serializable(self):
        return {}
    
    def bind(self, **kwargs):
        return self
    
    def with_config(self, **kwargs):
        return self


def create_llm(provider: str = "ollama", model: str = "llama3", **kwargs):
    """Create LLM based on provider.
    
    Args:
        provider: LLM provider (ollama, openai, openrouter, huggingface)
        model: Model name
        **kwargs: Additional parameters (base_url, temperature, max_tokens)
        
    Returns:
        LangChain LLM instance
    """
    if provider == "ollama":
        try:
            from langchain_community.llms import Ollama
            return Ollama(model=model, temperature=kwargs.get("temperature", 0.1))
        except ImportError:
            logger.warning("Ollama not available, using MockLLM")
            return MockLLM()
    
    elif provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            import os
            
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            
            if not hf_token:
                logger.warning("No HuggingFace token found, using MockLLM")
                return MockLLM()
            
            logger.info(f"Creating HuggingFace LLM: {model}")
            
            return HuggingFaceEndpoint(
                repo_id=model,
                max_new_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.1),
                huggingfacehub_api_token=hf_token,
            )
        except ImportError:
            logger.warning("HuggingFace not available, using MockLLM")
            return MockLLM()
        except Exception as e:
            logger.error(f"Failed to create HuggingFace LLM: {e}")
            return MockLLM()
    
    elif provider == "openai" or provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
            import os
            
            # Get API key from environment and strip whitespace
            api_key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
            
            # Get base_url if provided (for OpenRouter)
            base_url = kwargs.get("base_url")
            
            llm_params = {
                "model": model,
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "api_key": api_key,
            }
            
            if base_url:
                llm_params["base_url"] = base_url
            
            logger.info(f"Creating OpenAI-compatible LLM: {model}")
            if base_url:
                logger.info(f"Using custom base_url: {base_url}")
            
            return ChatOpenAI(**llm_params)
        except ImportError:
            logger.warning("OpenAI not available, using MockLLM")
            return MockLLM()
        except Exception as e:
            logger.error(f"Failed to create OpenAI LLM: {e}")
            return MockLLM()
    
    else:
        logger.warning(f"Unknown provider: {provider}, using MockLLM")
        return MockLLM()

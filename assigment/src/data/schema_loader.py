"""Schema loader for table metadata."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Column:
    """Represents a database column."""
    name: str
    type: str
    primary_key: bool = False
    foreign_key: Optional[str] = None
    nullable: bool = True
    unique: bool = False
    default: Optional[Any] = None
    values: Optional[List[str]] = None


@dataclass
class TableMetadata:
    """Represents a database table metadata."""
    name: str
    description: str
    columns: List[Column] = field(default_factory=list)
    
    def to_text(self) -> str:
        """Convert to text representation for LLM context."""
        cols = []
        for col in self.columns:
            col_info = f"  - {col.name} ({col.type})"
            if col.primary_key:
                col_info += " PRIMARY KEY"
            if col.foreign_key:
                col_info += f" REFERENCES {col.foreign_key}"
            if col.values:
                col_info += f" VALUES: {col.values}"
            if not col.nullable:
                col_info += " NOT NULL"
            cols.append(col_info)
        
        return f"Table: {self.name}\nDescription: {self.description}\nColumns:\n" + "\n".join(cols)
    
    def get_column_names(self) -> List[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]


@dataclass
class Scenario:
    """Represents a test scenario."""
    id: int
    question: str
    expected_tables: List[str]
    expected_sql_pattern: str


class SchemaLoader:
    """Loads and manages database schema metadata."""
    
    def __init__(self, schema_path: str):
        """Initialize schema loader.
        
        Args:
            schema_path: Path to schema JSON file
        """
        self.schema_path = Path(schema_path)
        self.tables: Dict[str, TableMetadata] = {}
        self.scenarios: List[Scenario] = []
        self._load_schema()
    
    def _load_schema(self) -> None:
        """Load schema from JSON file."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path, 'r') as f:
            data = json.load(f)
        
        # Load tables
        for table_name, table_data in data.get('tables', {}).items():
            columns = []
            for col_data in table_data.get('columns', []):
                col = Column(
                    name=col_data['name'],
                    type=col_data['type'],
                    primary_key=col_data.get('primary_key', False),
                    foreign_key=col_data.get('foreign_key'),
                    nullable=col_data.get('nullable', True),
                    unique=col_data.get('unique', False),
                    default=col_data.get('default'),
                    values=col_data.get('values')
                )
                columns.append(col)
            
            table = TableMetadata(
                name=table_name,
                description=table_data.get('description', ''),
                columns=columns
            )
            self.tables[table_name] = table
        
        # Load scenarios
        for scenario_data in data.get('scenarios', []):
            scenario = Scenario(
                id=scenario_data['id'],
                question=scenario_data['question'],
                expected_tables=scenario_data['expected_tables'],
                expected_sql_pattern=scenario_data['expected_sql_pattern']
            )
            self.scenarios.append(scenario)
    
    def get_table(self, table_name: str) -> Optional[TableMetadata]:
        """Get table metadata by name.
        
        Args:
            table_name: Name of the table
            
        Returns:
            TableMetadata or None if not found
        """
        return self.tables.get(table_name)
    
    def get_all_tables(self) -> List[TableMetadata]:
        """Get all table metadata.
        
        Returns:
            List of all tables
        """
        return list(self.tables.values())
    
    def get_table_names(self) -> List[str]:
        """Get all table names.
        
        Returns:
            List of table names
        """
        return list(self.tables.keys())
    
    def get_scenarios(self) -> List[Scenario]:
        """Get all test scenarios.
        
        Returns:
            List of scenarios
        """
        return self.scenarios
    
    def get_table_context(self, table_names: List[str]) -> str:
        """Get context text for specified tables.
        
        Args:
            table_names: List of table names to include
            
        Returns:
            Formatted context string
        """
        contexts = []
        for name in table_names:
            table = self.tables.get(name)
            if table:
                contexts.append(table.to_text())
        
        return "\n\n".join(contexts)
    
    def get_full_context(self) -> str:
        """Get context for all tables.
        
        Returns:
            Formatted context for all tables
        """
        all_tables = [table.name for table in self.tables.values()]
        return self.get_table_context(all_tables)
    
    def validate_columns(self, table_name: str, columns: List[str]) -> bool:
        """Validate that columns exist in table.
        
        Args:
            table_name: Name of the table
            columns: List of column names to validate
            
        Returns:
            True if all columns are valid
        """
        table = self.tables.get(table_name)
        if not table:
            return False
        
        valid_columns = set(table.get_column_names())
        return all(col in valid_columns for col in columns)
    
    def get_foreign_key_references(self, table_name: str) -> Dict[str, str]:
        """Get foreign key references for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dict mapping column to referenced table
        """
        table = self.tables.get(table_name, TableMetadata(name="", description=""))
        references = {}
        for col in table.columns:
            if col.foreign_key:
                references[col.name] = col.foreign_key
        return references

"""FastAPI application for SQL Agent."""
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import yaml

from ..utils.config import get_config
from ..utils.logging import setup_logging
from ..agent.sql_agent import create_agent, SQLAgent


# Request/Response Models
class SQLRequest(BaseModel):
    """Request model for SQL generation."""
    question: str = Field(..., description="Natural language question")


class SQLResponse(BaseModel):
    """Response model for SQL generation."""
    sql: str = Field(..., description="Generated SQL query")
    confidence: float = Field(..., description="Confidence score (0-1)")
    tables_used: list = Field(..., description="List of tables used")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")
    self_correction_attempts: int = Field(..., description="Number of self-correction attempts")
    is_valid: bool = Field(..., description="Whether SQL is valid")
    error_message: Optional[str] = Field(None, description="Error message if any")
    warnings: list = Field(default_factory=list, description="Warnings if any")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    agent_metrics: Dict[str, Any]
    api_metrics: Dict[str, Any]


# Global state
agent: Optional[SQLAgent] = None
start_time: float = time.time()
api_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency_ms": 0.0
}


def get_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    # Load config
    config = get_config()
    
    # Setup logging
    setup_logging(config.log_level)
    
    # Create FastAPI app
    app = FastAPI(
        title="Agentic SQL Generator",
        description="AI-powered SQL query generator with self-correction",
        version=config.app_version
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        global agent
        
        # Startup
        logger.info("Starting application")
        
        # Create agent
        agent = create_agent(config)
        
        logger.info("Application started successfully")
        yield
        
        # Shutdown
        logger.info("Shutting down application")
    
    app.router.lifespan_context = lifespan
    
    # Middleware for logging and metrics
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time and track metrics."""
        start = time.time()
        response = await call_next(request)
        
        process_time = (time.time() - start) * 1000
        response.headers["X-Process-Time"] = str(process_time)
        
        # Track API metrics
        if request.url.path in ["/generate-sql", "/generate-sql/"]:
            api_metrics["total_requests"] += 1
            api_metrics["total_latency_ms"] += process_time
            
            if response.status_code == 200:
                api_metrics["successful_requests"] += 1
            else:
                api_metrics["failed_requests"] += 1
        
        return response
    
    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )
    
    # Routes
    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint."""
        return {
            "name": config.app_name,
            "version": config.app_version,
            "status": "running"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version=config.app_version,
            uptime_seconds=time.time() - start_time
        )
    
    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics():
        """Metrics endpoint."""
        agent_metrics = agent.get_metrics() if agent else {}
        
        return MetricsResponse(
            agent_metrics=agent_metrics,
            api_metrics={
                "total_requests": api_metrics["total_requests"],
                "successful_requests": api_metrics["successful_requests"],
                "failed_requests": api_metrics["failed_requests"],
                "avg_latency_ms": (
                    api_metrics["total_latency_ms"] / api_metrics["total_requests"]
                    if api_metrics["total_requests"] > 0 else 0
                )
            }
        )
    
    @app.post("/generate-sql", response_model=SQLResponse)
    async def generate_sql(request: SQLRequest):
        """Generate SQL from natural language question."""
        if not agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        try:
            # Process question through agent
            result = agent.process(request.question)
            
            return SQLResponse(
                sql=result.sql,
                confidence=result.confidence,
                tables_used=result.tables_used,
                retrieval_time_ms=result.retrieval_time_ms,
                generation_time_ms=result.generation_time_ms,
                total_time_ms=result.total_time_ms,
                self_correction_attempts=result.self_correction_attempts,
                is_valid=result.is_valid,
                error_message=result.error_message,
                warnings=result.warnings
            )
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/reset-metrics")
    async def reset_metrics():
        """Reset agent metrics."""
        if agent:
            agent.reset_metrics()
        
        api_metrics["total_requests"] = 0
        api_metrics["successful_requests"] = 0
        api_metrics["failed_requests"] = 0
        api_metrics["total_latency_ms"] = 0.0
        
        return {"status": "metrics reset"}
    
    return app


# Create app instance
app = get_app()


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(app, host=config.api_host, port=config.api_port)

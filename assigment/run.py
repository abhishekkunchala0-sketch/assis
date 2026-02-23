"""Entry point for running the application."""
import uvicorn
from src.utils.config import get_config


def main():
    """Run the application."""
    config = get_config()
    
    print(f"Starting {config.app_name} v{config.app_version}")
    print(f"API will be available at http://{config.api_host}:{config.api_port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.get('api.reload', True)
    )


if __name__ == "__main__":
    main()

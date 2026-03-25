from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configurações da aplicação carregadas do arquivo .env."""
    models_dir: str = "models_store"
    logs_dir: str = "logs"
    min_training_points: int = 10
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

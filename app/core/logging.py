import logging
import logging.handlers
import os
import structlog
from app.core.config import settings


def setup_logging() -> None:
    """Configura handlers de stdout e arquivo rotativo, e inicializa o structlog com saída JSON."""
    os.makedirs(settings.logs_dir, exist_ok=True)

    # Handler para stdout (Docker lê daqui)
    stream_handler = logging.StreamHandler()

    # Handler para arquivo com rotação diária, mantém 7 dias
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(settings.logs_dir, "app.log"),
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        handlers=[stream_handler, file_handler],
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Retorna um logger structlog vinculado ao nome do módulo informado."""
    return structlog.get_logger(name)

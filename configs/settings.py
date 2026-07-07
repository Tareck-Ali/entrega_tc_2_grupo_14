from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Environment variables are automatically converted to the appropriate
    Python types based on the field annotations.

    Attributes:
        host: Hostname or IP address of the service.
        port: Port number the service listens on.
        username: Username used for authentication.
        password: Password used for authentication.
        log_level: Logging level for the application.
        debug: Whether debug mode is enabled.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    event_weight_view: int = 1
    event_weight_addtocart: int = 5
    event_weight_transaction: int = 10
    top_k: int = 10
    csv_file_name: str = "settings.csv"
    ttsplit_test_size: float = 0.2
    ttsplit_random_state: int = 42
    batch_size: int = 256
    learning_rate: float = 1e-3
    embedding_dim: int = 64
    epochs: int = 10

EVENT_WEIGHTS = {
    "view": settings.event_weight_view,
    "addtocart": settings.event_weight_addtocart,
    "transaction": settings.event_weight_transaction,
}

settings = Settings()

#print(settings.host)
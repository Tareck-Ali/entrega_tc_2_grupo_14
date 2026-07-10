from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

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
    model_checkpoint_name: str = "checkpoint.pth"
    ttsplit_test_size: float = 0.2
    ttsplit_random_state: int = 42
    batch_size: int = 256
    learning_rate: float = 1e-3
    embedding_dim: int = 64
    epochs: int = 10
    user_encoder: str = "artifacts/user_encoder.pkl"
    item_encoder: str = "artifacts/item_encoder.pkl"

EVENT_WEIGHTS = {
    "view": settings.event_weight_view,
    "addtocart": settings.event_weight_addtocart,
    "transaction": settings.event_weight_transaction,
}

settings = Settings()

#print(settings.host)
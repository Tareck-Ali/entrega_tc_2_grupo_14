from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """# App config
    APP_NAME: str = "DefaultApp"
    DEBUG: bool = False

    # Database config
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str

    # Security
    SECRET_KEY: str
    """
    # Tell Pydantic where to load .env from
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Create a single instance to use across the app
settings = Settings()
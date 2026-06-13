from settings import settings


def validate():
    errors = []
"""
    # Required checks
    if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 10:
        errors.append("SECRET_KEY is missing or too short")

    if settings.DB_PORT <= 0 or settings.DB_PORT > 65535:
        errors.append("DB_PORT is invalid")

    if not settings.DB_HOST:
        errors.append("DB_HOST is missing")

    if errors:
        print("Environment validation failed:")
        for e in errors:
            print(f" - {e}")
        exit(1)

    print("Valid")
"""

if __name__ == "__main__":
    validate()
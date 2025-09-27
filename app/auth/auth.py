from fastapi_users.authentication import BearerTransport, AuthenticationBackend, JWTStrategy
from dotenv import load_dotenv
import os

load_dotenv()

bearer_transport = BearerTransport(tokenUrl="/auth/jwt/login")

private_key = os.getenv('private_key')

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=private_key, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)
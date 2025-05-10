# generate_test_jwt.py
import jwt
token = jwt.encode(
    {"sub": "testuser"},
    "your-supabase-jwt-secret-for-testing",
    algorithm="HS256"
)
print(f"Bearer {token}")

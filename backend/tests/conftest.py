import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ["SEED_ON_STARTUP"] = "false"
os.environ["LLM_ENDPOINT"] = "https://api.example.com/v1"
os.environ["LLM_MODEL"] = "test-model"
os.environ["LLM_API_KEY"] = "test-api-key"
os.environ.setdefault("TEST_LLM_ENDPOINT", os.environ["LLM_ENDPOINT"])
os.environ.setdefault("TEST_LLM_MODEL", os.environ["LLM_MODEL"])
os.environ.setdefault("TEST_LLM_API_KEY", os.environ["LLM_API_KEY"])

from app.core.database import Base, get_db
from app.main import app

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)


@pytest.fixture(scope="function")
def db():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def test_llm_payload():
    return {
        "name": "Test LLM",
        "api_base_url": os.environ["TEST_LLM_ENDPOINT"],
        "api_key": os.environ["TEST_LLM_API_KEY"],
        "model_name": os.environ["TEST_LLM_MODEL"],
    }

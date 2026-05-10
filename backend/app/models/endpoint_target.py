from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func

from app.core.database import Base


class EndpointTarget(Base):
    __tablename__ = "endpoint_targets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    endpoint_url = Column(String(1000), nullable=False)
    transport_mode = Column(String(50), default="json")
    authorization = Column(Text, nullable=True)
    extra_headers = Column(Text, nullable=True)
    request_body_template = Column(Text, nullable=True)
    response_mapping = Column(JSON, nullable=True)
    default_test_input = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

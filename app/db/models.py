from sqlalchemy import Column, Integer
from .database import Base

class Counter(Base):
    """SQLAlchemy model for Counter table"""
    __tablename__ = "counter"

    id = Column(Integer, primary_key=True, index=True)
    value = Column(Integer, default=0)

from sqlalchemy import Column, Integer, String, Boolean, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from db import Base

class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True)
    plate_number = Column(String(20), unique=True, nullable=False)
    registered_at = Column(DateTime, server_default=func.now())


class GateLog(Base):
    __tablename__ = "gate_logs"

    id = Column(Integer, primary_key=True)
    plate_number = Column(String(20), nullable=False)
    status = Column(String(15))
    confidence_score = Column(Float)
    snapshot_path = Column(String)
    timestamp = Column(DateTime, server_default=func.now())
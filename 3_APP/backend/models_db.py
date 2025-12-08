# backend/models_db.py
from sqlalchemy import Column, BigInteger, Text, Numeric, Boolean, Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class VacancyDB(Base):
    __tablename__ = "vacancies"

    id = Column(BigInteger, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    salary = Column(Numeric(12, 2))
    skills = Column(ARRAY(Text))
    sectors = Column(ARRAY(Text))
    lat = Column(Float)
    lon = Column(Float)
    remote = Column(Boolean, default=False)
    embedding = Column(ARRAY(Float))  # embedding de la vacante


class CandidateDB(Base):
    __tablename__ = "candidates"

    id = Column(BigInteger, primary_key=True, index=True)
    title = Column(Text, nullable=False)         # rol del candidato
    experiences = Column(Text, nullable=False)   # texto de experiencia/CV
    salary = Column(Numeric(12, 2))
    skills = Column(ARRAY(Text))
    sectors = Column(ARRAY(Text))
    lat = Column(Float)
    lon = Column(Float)
    remote = Column(Boolean, default=False)
    embedding = Column(ARRAY(Float))  # embedding del candidato

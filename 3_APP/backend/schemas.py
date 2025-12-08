# schemas.py
from typing import List, Optional
from pydantic import BaseModel


# --------- VACANCIES ---------
class VacancyBase(BaseModel):
    title: str
    description: str
    salary: Optional[float] = None
    skills: List[str] = []
    sectors: List[str] = []
    lat: Optional[float] = None
    lon: Optional[float] = None
    remote: bool = False


class VacancyCreate(VacancyBase):
    pass


class VacancyUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    salary: Optional[float] = None
    skills: Optional[List[str]] = None
    sectors: Optional[List[str]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    remote: Optional[bool] = None
    embedding: Optional[List[float]] = None


class VacancyOut(VacancyBase):
    id: int
    embedding: Optional[List[float]] = None

    class Config:
        orm_mode = True


# --------- CANDIDATES ---------
class CandidateBase(BaseModel):
    title: str
    experiences: str
    salary: Optional[float] = None
    skills: List[str] = []
    sectors: List[str] = []
    lat: Optional[float] = None
    lon: Optional[float] = None
    remote: bool = False


class CandidateCreate(CandidateBase):
    pass


class CandidateUpdate(BaseModel):
    title: Optional[str] = None
    experiences: Optional[str] = None
    salary: Optional[float] = None
    skills: Optional[List[str]] = None
    sectors: Optional[List[str]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    remote: Optional[bool] = None
    embedding: Optional[List[float]] = None


class CandidateOut(CandidateBase):
    id: int
    embedding: Optional[List[float]] = None

    class Config:
        orm_mode = True


# --------- Embeddings & búsquedas ---------
class VacancyEmbeddingIn(BaseModel):
    embedding: List[float]


class VacancyEmbeddingOut(BaseModel):
    id: int
    embedding: List[float]


class VectorSearchRequest(BaseModel):
    cv_text: str
    candidate_lat: Optional[float] = None
    candidate_lon: Optional[float] = None
    top_k: int = 10


class VectorSearchMatch(BaseModel):
    vacancy: VacancyOut
    similarity: float


class VectorSearchResponse(BaseModel):
    matches: List[VectorSearchMatch]


# --------- Apply (afinidad CV-vacante) ---------
class ApplyRequest(BaseModel):
    vacancy_id: int
    cv_text: str
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    candidate_lat: Optional[float] = None
    candidate_lon: Optional[float] = None


class ApplyResponse(BaseModel):
    vacancy: VacancyOut
    affinity: float
    affinity_percent: float

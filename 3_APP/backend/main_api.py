# backend/main_api.py

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from db_connection import get_conn
from schemas import (
    VacancyCreate,
    VacancyUpdate,
    VacancyOut,
    CandidateCreate,
    CandidateUpdate,
    CandidateOut,
    ApplyRequest,
    ApplyResponse,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorSearchMatch,
    VacancyEmbeddingIn,
    VacancyEmbeddingOut,
)
from utils.affinity_calc import compute_affinity

app = FastAPI(
    title="Vacantes Matching API",
    version="2.0.0",
    description=(
        "API CRUD de vacantes/candidatos + matching de afinidad "
        "entre CV y vacante usando el modelo actual."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restringe en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================
#   CRUD VACANCIES
# ======================

@app.post("/vacancies", response_model=VacancyOut)
def create_vacancy(vac_in: VacancyCreate):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vacancies
                    (title, description, salary, skills, sectors, lat, lon, remote)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, title, description, salary, skills, sectors, lat, lon, remote, embedding
                """,
                (
                    vac_in.title,
                    vac_in.description,
                    vac_in.salary,
                    vac_in.skills,
                    vac_in.sectors,
                    vac_in.lat,
                    vac_in.lon,
                    vac_in.remote,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="No se pudo crear la vacante")

    return VacancyOut(**row)


@app.get("/vacancies", response_model=List[VacancyOut])
def list_vacancies(skip: int = 0, limit: int = 100):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, description, salary, skills, sectors, lat, lon, remote, embedding
                FROM vacancies
                ORDER BY id
                OFFSET %s LIMIT %s
                """,
                (skip, limit),
            )
            rows = cur.fetchall()
    return [VacancyOut(**r) for r in rows]


@app.get("/vacancies/{vacancy_id}", response_model=VacancyOut)
def get_vacancy(vacancy_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, description, salary, skills, sectors, lat, lon, remote, embedding
                FROM vacancies
                WHERE id = %s
                """,
                (vacancy_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Vacante no encontrada")

    return VacancyOut(**row)


@app.put("/vacancies/{vacancy_id}", response_model=VacancyOut)
def update_vacancy(vacancy_id: int, vac_in: VacancyCreate):
    # Para simplificar, PUT espera todos los campos (no patch parcial)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE vacancies
                SET title = %s,
                    description = %s,
                    salary = %s,
                    skills = %s,
                    sectors = %s,
                    lat = %s,
                    lon = %s,
                    remote = %s
                WHERE id = %s
                RETURNING id, title, description, salary, skills, sectors, lat, lon, remote, embedding
                """,
                (
                    vac_in.title,
                    vac_in.description,
                    vac_in.salary,
                    vac_in.skills,
                    vac_in.sectors,
                    vac_in.lat,
                    vac_in.lon,
                    vac_in.remote,
                    vacancy_id,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Vacante no encontrada")

    return VacancyOut(**row)


@app.delete("/vacancies/{vacancy_id}")
def delete_vacancy(vacancy_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM vacancies WHERE id = %s RETURNING id", (vacancy_id,))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Vacante no encontrada")

    return {"ok": True}


@app.put("/vacancies/{vacancy_id}/embedding", response_model=VacancyOut)
def update_vacancy_embedding(vacancy_id: int, emb: VacancyEmbeddingIn):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE vacancies
                SET embedding = %s
                WHERE id = %s
                RETURNING id, title, description, salary, skills, sectors, lat, lon, remote, embedding
                """,
                (emb.embedding, vacancy_id),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Vacante no encontrada")

    return VacancyOut(**row)


# ======================
#   CRUD CANDIDATES
# ======================

@app.post("/candidates", response_model=CandidateOut)
def create_candidate(cand_in: CandidateCreate):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidates
                    (title, experiences, salary, skills, sectors, lat, lon, remote)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, title, experiences, salary, skills, sectors, lat, lon, remote, embedding
                """,
                (
                    cand_in.title,
                    cand_in.experiences,
                    cand_in.salary,
                    cand_in.skills,
                    cand_in.sectors,
                    cand_in.lat,
                    cand_in.lon,
                    cand_in.remote,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="No se pudo crear el candidato")

    return CandidateOut(**row)


@app.get("/candidates", response_model=List[CandidateOut])
def list_candidates(skip: int = 0, limit: int = 100):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, experiences, salary, skills, sectors, lat, lon, remote, embedding
                FROM candidates
                ORDER BY id
                OFFSET %s LIMIT %s
                """,
                (skip, limit),
            )
            rows = cur.fetchall()
    return [CandidateOut(**r) for r in rows]


@app.get("/candidates/{candidate_id}", response_model=CandidateOut)
def get_candidate(candidate_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, experiences, salary, skills, sectors, lat, lon, remote, embedding
                FROM candidates
                WHERE id = %s
                """,
                (candidate_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Candidato no encontrado")

    return CandidateOut(**row)


@app.put("/candidates/{candidate_id}", response_model=CandidateOut)
def update_candidate(candidate_id: int, cand_in: CandidateCreate):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE candidates
                SET title = %s,
                    experiences = %s,
                    salary = %s,
                    skills = %s,
                    sectors = %s,
                    lat = %s,
                    lon = %s,
                    remote = %s
                WHERE id = %s
                RETURNING id, title, experiences, salary, skills, sectors, lat, lon, remote, embedding
                """,
                (
                    cand_in.title,
                    cand_in.experiences,
                    cand_in.salary,
                    cand_in.skills,
                    cand_in.sectors,
                    cand_in.lat,
                    cand_in.lon,
                    cand_in.remote,
                    candidate_id,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Candidato no encontrado")

    return CandidateOut(**row)


@app.delete("/candidates/{candidate_id}")
def delete_candidate(candidate_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM candidates WHERE id = %s RETURNING id", (candidate_id,))
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Candidato no encontrado")

    return {"ok": True}


# ======================
#   APPLY (afinidad)
# ======================

@app.post("/apply", response_model=ApplyResponse)
def apply_to_vacancy(req: ApplyRequest):
    # Cargamos la vacante desde la BD
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, description, salary, skills, sectors, lat, lon, remote, embedding
                FROM vacancies
                WHERE id = %s
                """,
                (req.vacancy_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Vacante no encontrada")

    vacancy = VacancyOut(**row)

    if not req.cv_text.strip():
        raise HTTPException(status_code=400, detail="El CV no puede estar vacío")

    # compute_affinity se encarga de construir textos y combinar con la localización
    affinity = compute_affinity(
        vacancy,
        req.cv_text,
        req.candidate_lat,
        req.candidate_lon,
    )

    return ApplyResponse(
        vacancy=vacancy,
        affinity=affinity,
        affinity_percent=affinity * 100.0,
    )


# ======================
#   BÚSQUEDA POR SIMILARIDAD
# ======================

@app.post("/vacancies/vector-search", response_model=VectorSearchResponse)
def vector_search_vacancies(req: VectorSearchRequest):
    """
    Búsqueda de vacantes similares a un CV usando compute_affinity.
    Recorrer todas las vacantes puede ser suficiente si el volumen es pequeño.
    """
    cv_text = (req.cv_text or "").strip()
    if not cv_text:
        raise HTTPException(status_code=400, detail="El CV no puede estar vacío")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, description, salary, skills, sectors, lat, lon, remote, embedding
                FROM vacancies
                ORDER BY id
                """
            )
            rows = cur.fetchall()

    matches: List[VectorSearchMatch] = []
    for row in rows:
        vacancy = VacancyOut(**row)

        try:
            score = compute_affinity(
                vacancy,
                cv_text,
                req.candidate_lat,
                req.candidate_lon,
            )
        except Exception:
            # Si hay algún problema con una vacante concreta, la saltamos
            continue

        matches.append(VectorSearchMatch(vacancy=vacancy, similarity=score))

    matches.sort(key=lambda m: m.similarity, reverse=True)
    top_matches = matches[: req.top_k]

    return VectorSearchResponse(matches=top_matches)

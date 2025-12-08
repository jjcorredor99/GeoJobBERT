-- db/init.sql

-- Opcional: crear el esquema
CREATE SCHEMA IF NOT EXISTS public;

-- Tabla de VACANTES
CREATE TABLE IF NOT EXISTS vacancies (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    salary      NUMERIC(12, 2),                -- puedes usar integer si prefieres
    skills      TEXT[] DEFAULT ARRAY[]::TEXT[],
    sectors     TEXT[] DEFAULT ARRAY[]::TEXT[],
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    remote      BOOLEAN DEFAULT FALSE,
    -- Espacio para el embedding precomputado de tu modelo siamesa
    -- por ejemplo, un vector de 256 dimensiones
    embedding   DOUBLE PRECISION[]
);

-- Tabla de CANDIDATOS
-- "mismo" esquema, pero description -> experiences
CREATE TABLE IF NOT EXISTS candidates (
    id           BIGSERIAL PRIMARY KEY,
    title        TEXT NOT NULL,                -- por ejemplo: "Data Scientist", "Backend Engineer"
    experiences  TEXT NOT NULL,                -- CV/experiencia del candidato
    salary       NUMERIC(12, 2),               -- expectativa salarial
    skills       TEXT[] DEFAULT ARRAY[]::TEXT[],
    sectors      TEXT[] DEFAULT ARRAY[]::TEXT[],
    lat          DOUBLE PRECISION,
    lon          DOUBLE PRECISION,
    remote       BOOLEAN DEFAULT FALSE,
    -- También dejamos columna de embedding para el candidato, útil para matching rápido
    embedding    DOUBLE PRECISION[]
);

-- Índices útiles para búsquedas
CREATE INDEX IF NOT EXISTS idx_vacancies_skills
    ON vacancies USING GIN (skills);

CREATE INDEX IF NOT EXISTS idx_vacancies_sectors
    ON vacancies USING GIN (sectors);

CREATE INDEX IF NOT EXISTS idx_candidates_skills
    ON candidates USING GIN (skills);

CREATE INDEX IF NOT EXISTS idx_candidates_sectors
    ON candidates USING GIN (sectors);


-- ============================
--   DATOS DE EJEMPLO
-- ============================

-- Vacantes ejemplo (alineadas con las que usabas en el API)
INSERT INTO vacancies (title, description, salary, skills, sectors, lat, lon, remote)
VALUES
  (
    'Data Scientist',
    'Responsable de construir modelos de machine learning, analizar grandes volúmenes de datos y comunicar hallazgos a stakeholders de negocio. Requiere experiencia con Python, Pandas, SQL y frameworks de deep learning.',
    80000,
    ARRAY['python', 'machine learning', 'pandas', 'sql'],
    ARRAY['Data', 'Tech'],
    19.4326,   -- CDMX
    -99.1332,
    TRUE
  ),
  (
    'Backend Engineer (Python/FastAPI)',
    'Diseñar y desarrollar APIs escalables en Python usando FastAPI, integración con bases de datos relacionales y NoSQL, despliegue en la nube (AWS/GCP) y buenas prácticas de testing.',
    90000,
    ARRAY['python', 'fastapi', 'rest', 'sql'],
    ARRAY['Backend', 'Tech'],
    19.4326,   -- CDMX
    -99.1332,
    FALSE
  ),
  (
    'ML Engineer',
    'Diseño de pipelines de ML de punta a punta: feature engineering, entrenamiento, evaluación y despliegue de modelos. Experiencia con MLOps, Docker, Kubernetes y herramientas de orquestación.',
    95000,
    ARRAY['ml', 'mlops', 'docker', 'kubernetes'],
    ARRAY['ML', 'Tech'],
    -34.6037,  -- Buenos Aires
    -58.3816,
    TRUE
  );

-- Candidatos ejemplo
INSERT INTO candidates (title, experiences, salary, skills, sectors, lat, lon, remote)
VALUES
  (
    'Data Scientist',
    '3 años de experiencia en modelos supervisados, Python, scikit-learn y SQL. Experiencia en banca y seguros.',
    82000,
    ARRAY['python', 'scikit-learn', 'sql'],
    ARRAY['Banca', 'Seguros'],
    19.4326,
    -99.1332,
    TRUE
  ),
  (
    'Backend Engineer',
    '5 años desarrollando APIs REST en Python y Node.js, experiencia con FastAPI, PostgreSQL y despliegues en AWS.',
    88000,
    ARRAY['python', 'fastapi', 'postgresql', 'aws'],
    ARRAY['Backend', 'Cloud'],
    40.4168,   -- Madrid
    -3.7038,
    FALSE
  );

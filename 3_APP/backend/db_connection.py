import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Ruta base de este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env.prod"))


# poc_user=os.getenv("poc_user")
# poc_password=os.getenv("poc_password")
# poc_host=os.getenv("poc_host")
# poc_database=os.getenv("poc_database")
# poc_port=os.getenv("poc_port")

poc_user="vector_store_jobs"
poc_password="jobs_password"
poc_host="localhost"
poc_database="vector_store_jobs"
poc_port=5432

DATABASE_URL = f"postgresql://{poc_user}:{poc_password}@{poc_host}:5432/{poc_database}"


def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def get_all_vacancies():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM vacancies ORDER BY id;")
            return cur.fetchall()


def get_vacancy_by_id(vacancy_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM vacancies WHERE id = %s;", (vacancy_id,))
            return cur.fetchone()

import os
import folium
import requests
import streamlit as st
from streamlit_folium import st_folium
# Puedes sobreescribir esto con BACKEND_URL en tu .env / entorno
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Portal de Vacantes",
    page_icon="💼",
    layout="centered",
)


# -----------------------------
# Helpers de backend
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_vacancies():
    """
    Llama a GET /vacancies del backend.

    El backend actual devuelve una lista de vacantes (no un dict con 'vacancies'),
    pero dejamos compatibilidad por si en algún momento cambia.
    """
    resp = requests.get(f"{BACKEND_URL}/vacancies")
    resp.raise_for_status()
    data = resp.json()

    # Compatibilidad con posible forma antigua {"vacancies": [...]}
    if isinstance(data, dict) and "vacancies" in data:
        return data["vacancies"]

    if not isinstance(data, list):
        raise RuntimeError("Respuesta inesperada desde /vacancies")

    return data


def format_location(vacancy: dict) -> str:
    """
    El backend no tiene un campo 'location'; sólo lat/lon y remote.
    Creamos una cadena legible para mostrar.
    """
    lat = vacancy.get("lat")
    lon = vacancy.get("lon")
    remote = vacancy.get("remote")

    has_coords = lat is not None and lon is not None

    if remote and has_coords:
        return f"Híbrido · lat {lat:.4f}, lon {lon:.4f}"
    if remote and not has_coords:
        return "Remoto"
    if has_coords:
        return f"Presencial · lat {lat:.4f}, lon {lon:.4f}"
    return "Ubicación no especificada"


# -----------------------------
# Picker de ubicación
# -----------------------------
def render_location_picker(vacancy: dict):
    """
    Widget completo para elegir la ubicación del candidato:
    - mapa interactivo (click para mover el pin)
    - lat / lon visibles y editables
    - botón para centrar en la vacante
    - intenta geolocalización del navegador (si se instala streamlit-js-eval)
    """
    st.markdown("### 2. Tu ubicación")

    st.caption(
        "Marca en el mapa dónde estás (o dónde te interesa trabajar). "
        "La ubicación se usa como señal adicional en el modelo."
    )

    # Valores iniciales en sesión
    default_lat = vacancy.get("lat") if vacancy.get("lat") is not None else 4.641518   # CDMX fallback
    default_lon = vacancy.get("lon") if vacancy.get("lon") is not None else -74.062047

    if "candidate_lat" not in st.session_state:
        st.session_state["candidate_lat"] = float(default_lat)
    if "candidate_lon" not in st.session_state:
        st.session_state["candidate_lon"] = float(default_lon)

    col_map, col_controls = st.columns([3, 2])

    # ----- Columna derecha: controles, geolocalización y números -----
    with col_controls:
        st.markdown("#### Coordenadas")

        lat = st.number_input(
            "Latitud",
            value=float(st.session_state["candidate_lat"]),
            format="%.6f",
        )
        lon = st.number_input(
            "Longitud",
            value=float(st.session_state["candidate_lon"]),
            format="%.6f",
        )
        st.session_state["candidate_lat"] = lat
        st.session_state["candidate_lon"] = lon

        st.caption("Puedes editar los valores o hacer clic en el mapa para actualizarlos.")

        # Botón para centrar en la vacante (si tiene coords)
        if vacancy.get("lat") is not None and vacancy.get("lon") is not None:
            if st.button("🏢 Centrar en la ubicación de la vacante"):
                st.session_state["candidate_lat"] = float(vacancy["lat"])
                st.session_state["candidate_lon"] = float(vacancy["lon"])
                st.rerun()

        # ------------------ Geolocalización navegador ------------------
        # Montamos el componente SIEMPRE fuera del botón para evitar
        # el bug de "unregistered ComponentInstance".
        loc = None
        geoloc_available = True
        try:
            from streamlit_js_eval import get_geolocation

            # Esto dibuja el enlace/botón del componente JS.
            # Cuando el usuario haga clic y acepte permisos, `loc`
            # empezará a venir con coords.
            loc = get_geolocation()
        except Exception:
            geoloc_available = False
            st.caption(
                "No pude usar la geolocalización automática. "
                "Instala `streamlit-js-eval` o marca tu posición en el mapa."
            )

        # Botón "Usar mi ubicación actual" que sólo aplica el último valor devuelto
        if st.button("📡 Usar mi ubicación actual", disabled=not geoloc_available):
            if loc and loc.get("coords"):
                st.session_state["candidate_lat"] = loc["coords"]["latitude"]
                st.session_state["candidate_lon"] = loc["coords"]["longitude"]
                st.success("Ubicación detectada, ajusta en el mapa si lo necesitas.")
                st.rerun()
            else:
                st.warning(
                    "Aún no tengo coordenadas del navegador. "
                    "Haz clic primero en el enlace generado por el componente "
                    "(por ejemplo 'Get Location') y acepta los permisos."
                )

    # ----- Columna izquierda: mapa interactivo -----
    with col_map:
        st.markdown("#### Mapa interactivo")



        center = [
            float(st.session_state["candidate_lat"]),
            float(st.session_state["candidate_lon"]),
        ]

        m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

        # Marker del candidato
        folium.Marker(
            center,
            tooltip="Tu ubicación",
            icon=folium.Icon(color="blue", icon="user"),
        ).add_to(m)

        # Marker de la vacante (si tiene coords)
        if vacancy.get("lat") is not None and vacancy.get("lon") is not None:
            folium.Marker(
                [vacancy["lat"], vacancy["lon"]],
                tooltip=f"Vacante: {vacancy['title']}",
                icon=folium.Icon(color="green", icon="briefcase"),
            ).add_to(m)

        map_data = st_folium(m, height=400, width=700)

        # Si el usuario hace clic en el mapa, actualizamos coords
        if map_data and map_data.get("last_clicked"):
            st.session_state["candidate_lat"] = map_data["last_clicked"]["lat"]
            st.session_state["candidate_lon"] = map_data["last_clicked"]["lng"]


    return float(st.session_state["candidate_lat"]), float(st.session_state["candidate_lon"])



# -----------------------------
# App principal
# -----------------------------
def main():
    st.title("💼 Portal de Vacantes con Matching de CV")

    # 1. Cargar vacantes desde la API
    try:
        vacancies = fetch_vacancies()
    except Exception as e:
        st.error(f"No se pudieron cargar las vacantes desde el backend: {e}")
        return

    if not vacancies:
        st.warning("No hay vacantes disponibles en este momento.")
        return

    # Selección de vacante
    st.subheader("1. Selecciona una vacante")

    options = {}
    for v in vacancies:
        loc_str = format_location(v)
        label = f"{v.get('id', '?')} · {v.get('title', 'Sin título')} — {loc_str}"
        options[label] = v

    selected_label = st.selectbox("Vacante:", list(options.keys()))
    vacancy = options[selected_label]

    # Tarjeta con detalle de vacante
    location_str = format_location(vacancy)
    salary = vacancy.get("salary")
    skills = vacancy.get("skills") or []
    sectors = vacancy.get("sectors") or []

    salary_html = ""
    if salary is not None:
        try:
            salary_float = float(salary)
            salary_str = f"{salary_float:,.0f}"
        except Exception:
            salary_str = str(salary)
        salary_html = f"<p style='margin: 0.2rem 0;'>💰 Salario objetivo: {salary_str}</p>"

    skills_html = ""
    if skills:
        skills_html = (
            "<p style='margin: 0.2rem 0;'>🧩 Skills: "
            + ", ".join(skills)
            + "</p>"
        )

    sectors_html = ""
    if sectors:
        sectors_html = (
            "<p style='margin: 0.2rem 0;'>🏷 Sectores: "
            + ", ".join(sectors)
            + "</p>"
        )

    with st.container():
        st.markdown(
            f"""
            <div style="padding: 1rem; border-radius: 0.5rem;
                        border: 1px solid #eee; background-color: #00000;">
              <h4 style="margin-bottom: 0.2rem;">{vacancy.get('title', 'Sin título')}</h4>
              <p style="margin: 0.2rem 0; color: #666;">📍 {location_str}</p>
              {salary_html}
              {skills_html}
              {sectors_html}
              <p style="margin-top: 0.5rem;">{vacancy.get('description', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 2. Ubicación en mapa
    cand_lat, cand_lon = render_location_picker(vacancy)

    st.markdown("---")
    st.subheader("3. Pega tu CV")

    cv_text = st.text_area(
        "Texto de tu CV:",
        height=250,
        placeholder="Ejemplo: Soy ingeniera de datos con experiencia en Python, Spark...",
    )

    col1, col2 = st.columns(2)
    with col1:
        candidate_name = st.text_input("Nombre (opcional)")
    with col2:
        candidate_email = st.text_input("Email (opcional)")

    st.markdown("---")
    st.subheader("4. Postular y ver afinidad")

    if st.button("Postularme a esta vacante"):
        if not cv_text.strip():
            st.error("Por favor, pega tu CV antes de postularte.")
            return

        payload = {
            "vacancy_id": vacancy.get("id"),
            "cv_text": cv_text,
            "candidate_name": candidate_name or None,
            "candidate_email": candidate_email or None,
            "candidate_lat": cand_lat,
            "candidate_lon": cand_lon,
        }

        try:
            with st.spinner("Calculando afinidad con tu CV y tu ubicación..."):
                resp = requests.post(f"{BACKEND_URL}/apply", json=payload)

            if resp.status_code != 200:
                st.error(f"Error desde el backend: {resp.text}")
                return

            data = resp.json()
            affinity = data.get("affinity")
            affinity_percent = data.get("affinity_percent", affinity * 100 if affinity is not None else None)

            if affinity is None or affinity_percent is None:
                st.error("La respuesta del backend no contiene campos de afinidad esperados.")
                st.json(data)
                return

            st.success("✅ Postulación enviada correctamente.")
            st.metric("Afinidad estimada", f"{affinity_percent:.1f} %")

            # Afinidad [0,1] -> barra de progreso
            progress_value = min(1.0, max(0.0, float(affinity)))
            st.progress(progress_value)

            st.caption(
                "La afinidad se calcula con un modelo siamesa de texto + localización, "
                "entrenado sobre pares vacante–candidato."
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Error de conexión con el backend: {e}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Aplicación Streamlit: predicción de deserción estudiantil (SVM).

Ejecutar:  streamlit run app_Desercion.py
Requiere en la misma carpeta: desercion_core.py y modelo-clas_Desercion.pkl
"""
from __future__ import annotations

import base64
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import desercion_core as core

MODEL_PATH = Path(__file__).with_name("modelo-clas_Desercion.pkl")
RECALL = 0.93  # métrica de validación del modelo (clase "SI")

# Paleta: azules oscuros mate + acentos apagados
BG, SURFACE, BORDER = "#0a1424", "#101e36", "#1f3358"
ACCENT, TEXT, MUTED = "#5b86c6", "#e4eaf4", "#8fa3c2"
RISK, OK = "#c4666d", "#4f9e88"

# Streamlit >= 1.50 usa width="stretch"; versiones previas, use_container_width.
try:
    _MINOR = int(st.__version__.split(".")[1]) if st.__version__.startswith("1.") else 99
except ValueError:
    _MINOR = 99
STRETCH = {"width": "stretch"} if _MINOR >= 50 else {"use_container_width": True}

st.set_page_config(page_title="Deserción estudiantil", page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
def _academic_background() -> str:
    """Patrón SVG sutil (birrete, libro, lápiz) embebido como data-URI."""
    svg = f"""
<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240' viewBox='0 0 240 240'
     fill='none' stroke='{ACCENT}' stroke-width='1.3' stroke-linecap='round' stroke-linejoin='round'>
  <g><polygon points='40,44 72,30 104,44 72,58'/>
     <path d='M56,51 v11 c0,7 32,7 32,0 v-11'/><path d='M104,44 v18'/><circle cx='104' cy='64' r='2'/></g>
  <g transform='translate(128,138)'>
     <path d='M0,10 C15,2 30,2 40,8 C50,2 65,2 80,10 V38 C65,30 50,30 40,36 C30,30 15,30 0,38 Z'/>
     <path d='M40,8 V36'/></g>
  <g transform='translate(34,166) rotate(-35)'>
     <rect x='0' y='0' width='48' height='9'/><path d='M48,0 L58,4.5 L48,9'/><path d='M9,0 V9'/></g>
  <g stroke-width='1'><circle cx='180' cy='40' r='9'/><ellipse cx='180' cy='40' rx='9' ry='3.5'/>
     <ellipse cx='180' cy='40' rx='3.5' ry='9'/></g>
  <g stroke-width='1' opacity='.8'><path d='M150,200 h40 M150,208 h30 M150,216 h36'/></g>
</svg>"""
    return base64.b64encode(svg.encode()).decode()


def inject_css() -> None:
    css = f"""
<style>
:root {{ --bg:{BG}; --surface:{SURFACE}; --border:{BORDER}; --accent:{ACCENT};
        --text:{TEXT}; --muted:{MUTED}; --risk:{RISK}; --ok:{OK}; }}

/* Fondo: degradado mate + textura académica a muy baja opacidad */
.stApp {{ background: radial-gradient(1100px 650px at 12% -8%, #13274d 0%, var(--bg) 62%);
         color: var(--text); isolation: isolate; }}
.stApp::before {{ content:""; position:fixed; inset:0; z-index:-1; pointer-events:none;
  background-image:url("data:image/svg+xml;base64,{_academic_background()}");
  background-size:240px 240px; opacity:.07; }}

#MainMenu, footer {{ visibility:hidden; }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:2.2rem; max-width:1280px; }}
h1, h2, h3 {{ letter-spacing:-.01em; font-weight:650; }}
h1 {{ font-size:2rem !important; }}

/* Barra lateral */
[data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--border);
  box-shadow:6px 0 28px rgba(0,0,0,.35); }}
[data-testid="stForm"] {{ border:1px solid var(--border); border-radius:14px; padding:1rem;
  background:rgba(10,20,36,.55); box-shadow:inset 0 1px 0 rgba(255,255,255,.03); }}
.group-title {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.09em;
  color:var(--muted); margin:.9rem 0 .1rem; }}

/* Paneles (st.container(border=True, key="panel_*")) */
[class*="st-key-panel"] {{ background:rgba(16,30,54,.82); border:1px solid var(--border) !important;
  border-radius:16px; padding:1.1rem 1.25rem; box-shadow:0 10px 30px rgba(0,0,0,.30);
  backdrop-filter:blur(2px); transition:box-shadow .25s ease, transform .25s ease; }}
[class*="st-key-panel"]:hover {{ box-shadow:0 14px 36px rgba(0,0,0,.38); }}

/* Tarjeta de resultado */
.result {{ border-radius:12px; padding:1rem 1.1rem; border:1px solid; animation:rise .45s ease-out; }}
.result.low  {{ background:rgba(79,158,136,.12); border-color:rgba(79,158,136,.45); }}
.result.high {{ background:rgba(196,102,109,.12); border-color:rgba(196,102,109,.50); }}
.result .kicker {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); }}
.result .verdict {{ font-size:1.35rem; font-weight:650; margin:.15rem 0 .6rem; }}
.result.low .verdict  {{ color:#8fd1bd; }}
.result.high .verdict {{ color:#e39aa0; }}
.meter {{ height:8px; border-radius:99px; background:rgba(143,163,194,.18); overflow:hidden; }}
.meter > span {{ display:block; height:100%; width:var(--w); border-radius:99px;
  animation:grow .9s cubic-bezier(.2,.8,.2,1); }}
.low .meter > span  {{ background:linear-gradient(90deg,#3f7f6d,var(--ok)); }}
.high .meter > span {{ background:linear-gradient(90deg,#a4505a,var(--risk)); }}
.meter-legend {{ display:flex; justify-content:space-between; font-size:.72rem; color:var(--muted); margin-top:.3rem; }}
@keyframes grow {{ from {{ width:0; }} }}
@keyframes rise {{ from {{ opacity:0; transform:translateY(6px); }} }}

/* Métricas, botones, pestañas */
[data-testid="stMetric"] {{ background:rgba(10,20,36,.6); border:1px solid var(--border);
  border-radius:12px; padding:.7rem .9rem; transition:transform .2s ease, border-color .2s ease; }}
[data-testid="stMetric"]:hover {{ transform:translateY(-2px); border-color:var(--accent); }}
[data-testid="stMetricLabel"] {{ color:var(--muted); }}
[data-testid="stFormSubmitButton"] button {{ background:#27508f; color:#fff; border:0; border-radius:10px;
  font-weight:600; box-shadow:0 4px 14px rgba(39,80,143,.35); transition:all .2s ease; }}
[data-testid="stFormSubmitButton"] button:hover {{ background:#3363ad; transform:translateY(-1px); }}
button[data-baseweb="tab"] {{ color:var(--muted); }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:var(--text); }}
[data-baseweb="tab-highlight"] {{ background:var(--accent) !important; }}
[data-testid="stDataFrame"] {{ border-radius:10px; overflow:hidden; }}
.note {{ color:var(--muted); font-size:.82rem; line-height:1.5; }}
</style>"""
    st.markdown(css, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Modelo (se carga una sola vez por proceso)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Cargando modelo…")
def get_artifacts() -> core.Artifacts:
    return core.load_artifacts(MODEL_PATH)


# --------------------------------------------------------------------------- #
# Entradas
# --------------------------------------------------------------------------- #
def sidebar_form(art: core.Artifacts) -> dict:
    """Formulario de captura. Solo recalcula al presionar «Predecir»."""
    rng = core.bounds(art)
    d = core.DEFAULTS
    L = core.LABELS
    opts = core.CAT_OPTIONS

    def group(title: str) -> None:
        st.markdown(f'<div class="group-title">{title}</div>', unsafe_allow_html=True)

    def num(col: str, step) -> float | int:
        lo, hi = rng[col]
        return st.slider(L[col], min_value=lo, max_value=hi, value=d[col], step=step)

    def cat(col: str, fmt=lambda s: s):
        return st.selectbox(L[col], opts[col], index=opts[col].index(d[col]), format_func=fmt)

    with st.sidebar:
        st.markdown("### Datos del estudiante")
        with st.form("perfil"):
            group("Periodo")
            lo, hi = rng["TERM_ID"]
            term = st.number_input(L["TERM_ID"], min_value=lo, max_value=hi, value=d["TERM_ID"], step=1)
            group("Desempeño del periodo")
            gpa_term = num("GPA_TERM", 0.05)
            cr_att = num("CREDITS_ATTEMPTED_TERM", 1)
            cr_fail = num("CREDITS_FAILED_TERM", 1)
            group("Historial acumulado")
            gpa_cum = num("GPA_CUM", 0.05)
            cr_apr = num("CREDITS_APPROVED_CUM_ALL", 1)
            cr_fail_cum = num("CREDITS_FAILED_CUM", 1)
            group("Contexto académico")
            escuela = cat("COD_ESCUELA")
            estado = cat("ACADEMIC_STATUS_TERM", lambda s: s.strip("'"))
            genero = cat("GENERO")
            group("Apoyos recibidos")
            fin = cat("RECIBIO_APOYO_FINANCIERO")
            aca = cat("RECIBIO_APOYO_ACADEMICO")
            otros = cat("RECIBIO_OTROS_APOYOS")
            st.form_submit_button("Predecir", **STRETCH)

    return {
        "TERM_ID": int(term), "GPA_TERM": float(gpa_term), "CREDITS_ATTEMPTED_TERM": int(cr_att),
        "CREDITS_FAILED_TERM": int(cr_fail), "GPA_CUM": float(gpa_cum),
        "CREDITS_APPROVED_CUM_ALL": int(cr_apr), "CREDITS_FAILED_CUM": int(cr_fail_cum),
        "COD_ESCUELA": escuela, "ACADEMIC_STATUS_TERM": estado, "GENERO": genero,
        "RECIBIO_APOYO_FINANCIERO": fin, "RECIBIO_APOYO_ACADEMICO": aca, "RECIBIO_OTROS_APOYOS": otros,
    }


# --------------------------------------------------------------------------- #
# Visualizaciones
# --------------------------------------------------------------------------- #
PLOT_CONFIG = {"displayModeBar": False}


def style_fig(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12), showlegend=False,
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font_color=TEXT),
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    fig.update_xaxes(gridcolor="rgba(143,163,194,.14)", zeroline=False, linecolor=BORDER)
    fig.update_yaxes(gridcolor="rgba(143,163,194,.14)", zeroline=False, linecolor=BORDER)
    return fig


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG, **STRETCH)


def tornado_chart(profile: dict, art: core.Artifacts) -> go.Figure:
    """Factores críticos: cuánto puede mover cada variable el índice de riesgo."""
    df = core.factor_impact(profile, art)
    fig = go.Figure()
    fig.add_bar(
        y=df["etiqueta"], x=df["rango"], base=df["bajo"], orientation="h",
        marker=dict(color=ACCENT, opacity=.75, line=dict(width=0)),
        customdata=df[["bajo", "alto"]],
        hovertemplate="<b>%{y}</b><br>Índice entre %{customdata[0]:.0f} y %{customdata[1]:.0f}<extra></extra>",
    )
    current = float(df["actual"].iloc[0])
    fig.add_vline(x=current, line=dict(color=TEXT, width=1.5), annotation_text="Estudiante",
                  annotation_font_color=TEXT, annotation_position="top")
    fig.add_vline(x=50, line=dict(color=RISK, width=1, dash="dot"), annotation_text="Frontera SVM",
                  annotation_font_color=RISK, annotation_position="bottom")
    fig.update_xaxes(range=[0, 100], title="Índice de riesgo (0–100)")
    fig.update_yaxes(title=None)
    return style_fig(fig, 420)


def sensitivity_chart(profile: dict, var: str, art: core.Artifacts) -> go.Figure:
    """Curva índice de riesgo vs. una variable numérica, con el perfil actual marcado."""
    curve = core.sensitivity_curve(profile, var, art)
    here = core.predict(profile, art).index * 100
    fig = go.Figure()
    fig.add_hrect(y0=50, y1=100, fillcolor=RISK, opacity=.08, line_width=0)
    fig.add_scatter(x=curve["x"], y=curve["indice"], mode="lines",
                    line=dict(color=ACCENT, width=3, shape="spline"),
                    hovertemplate=f"{core.LABELS[var]}: %{{x:.2f}}<br>Índice: %{{y:.1f}}<extra></extra>")
    fig.add_scatter(x=[profile[var]], y=[here], mode="markers",
                    marker=dict(color=TEXT, size=12, line=dict(color=ACCENT, width=3)),
                    hovertemplate="Perfil actual<br>Índice: %{y:.1f}<extra></extra>")
    fig.add_hline(y=50, line=dict(color=RISK, width=1, dash="dot"))
    fig.update_xaxes(title=core.LABELS[var])
    fig.update_yaxes(title="Índice de riesgo (0–100)", range=[0, 100])
    return style_fig(fig, 380)


# --------------------------------------------------------------------------- #
# Página
# --------------------------------------------------------------------------- #
def result_card(pred: core.Prediction) -> str:
    kind = "high" if pred.high_risk else "low"
    text = "Riesgo ALTO de deserción" if pred.high_risk else "Riesgo BAJO de deserción"
    pct = round(pred.index * 100)
    return f"""
<div class="result {kind}">
  <div class="kicker">Resultado del modelo</div>
  <div class="verdict">{text}</div>
  <div class="meter" style="--w:{pct}%"><span></span></div>
  <div class="meter-legend"><span>0</span><span>Índice de riesgo: {pct}/100</span><span>100</span></div>
</div>"""


def main() -> None:
    inject_css()
    if not MODEL_PATH.exists():
        st.error(f"No se encontró el modelo en {MODEL_PATH.name}. Colóquelo junto a la aplicación.")
        st.stop()
    art = get_artifacts()
    profile = sidebar_form(art)
    pred = core.predict(profile, art)

    st.title("Predicción de deserción estudiantil")
    st.caption("Modelo SVM (kernel RBF) · perfil académico y de apoyos · educación superior")

    col_left, col_right = st.columns([1, 1.25], gap="large")

    with col_left:
        with st.container(border=True, key="panel_resultado"):
            st.subheader("Resultado")
            st.markdown(result_card(pred), unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Predicción", "SI deserta" if pred.high_risk else "NO deserta")
            m2.metric("Margen SVM", f"{pred.score:+.2f}",
                      help="Distancia con signo a la frontera de decisión. > 0 → clase «SI».")
            st.dataframe(core.profile_table(profile, pred), hide_index=True, height=300, **STRETCH)
            st.markdown(
                f'<p class="note">Recall del modelo en validación: <b>{RECALL:.0%}</b> '
                "(detecta ~9 de cada 10 casos de deserción; puede generar falsas alarmas). "
                "El índice es una escala derivada del margen, <b>no</b> una probabilidad calibrada.</p>",
                unsafe_allow_html=True)

    with col_right:
        with st.container(border=True, key="panel_analisis"):
            st.subheader("¿Qué factores pesan más en este perfil?")
            tab_factors, tab_curve = st.tabs(["Factores críticos", "Sensibilidad"])
            with tab_factors:
                st.markdown(
                    '<p class="note">Cada barra muestra cuánto cambia el índice de riesgo al mover '
                    "<b>solo esa variable</b> a sus valores extremos, manteniendo el resto del perfil.</p>",
                    unsafe_allow_html=True)
                plot(tornado_chart(profile, art))
            with tab_curve:
                numeric = [c for c in core.NUM_COLS if c != "TERM_ID"]
                var = st.selectbox("Variable a explorar", numeric, format_func=core.LABELS.get,
                                   index=numeric.index("GPA_CUM"))
                plot(sensitivity_chart(profile, var, art))


main()

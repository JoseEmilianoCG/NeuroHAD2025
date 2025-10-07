# neuro_dashboard.py (scalp-only)
# Dash en proceso separado + comunicación por multiprocessing.Queue
# Uso (idéntico al anterior, pero ahora se empuja un dict "scalp" con TP9/AF7/AF8/TP10 en [0,1]):
#   main:
#       from multiprocessing import Process, Queue
#       from neuro_dashboard import run_dash, push, shutdown
#       dash_q = Queue()
#       p_dash = Process(target=run_dash, args=(dash_q,),
#                        kwargs=dict(port=8051, window_sec=180, offset=0.8,
#                                    title="Sincronía cerebral"),
#                        daemon=True)
#       p_dash.start()
#       ...
#       # Enviar una actualización de scalp (valores en 0..1):
#       push(dash_q, t=0.0, scalp={"TP9":0.2, "AF7":0.7, "AF8":0.6, "TP10":0.3})
#       # (radars es opcional y se sobrepone en el panel derecho como antes)
#       push(dash_q, t=0.0, scalp={...}, radars={
#           "Métrica A": {"Delta":0.2, "Theta":0.35, "Alpha":0.5, "Beta":0.4, "Gamma":0.3}
#       })
#       # cuando cierres:
#       shutdown(dash_q)  # o p_dash.terminate()

from collections import defaultdict, deque
import numpy as np
import dash
from dash import dcc, html
import plotly.graph_objects as go

PALETTE = {
    "bg": "#FAFBFF",
    "panel": "#FFFFFF",
    "border": "#E8ECF7",
    "text": "#2B2B2B",
    "subtext": "#5A6270",
    "grid": "#E9EDF5",
    # paleta fría→cálida, tonos suaves
    # (valores en 0..1 irán de azul claro a rosado cálido)
    "scalp_colorscale": [
        [0.00, "#EEF6FF"],
        [0.25, "#CFE8FF"],
        [0.50, "#FFDDE2"],
        [0.75, "#FFC2CC"],
        [1.00, "#FF8FA3"],
    ],
    "primary": "#8E77F0",
}


def apply_light_theme(fig):
    fig.update_layout(
        template=None,
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"]),
        legend=dict(font=dict(color=PALETTE["text"]), bgcolor="rgba(255,255,255,0.6)"),
    )
    return fig


# ------------------------------
# API minimalista para productores
# ------------------------------

def push(q, t, series_dict=None, metric1=None, metric2=None, radars=None, scalp=None):
    """
    Envía actualización al dashboard.

    - t: escalar o lista de tiempos (solo informativo; ya no se grafican líneas)
    - series_dict: DEPRECADO (ignoradas en UI)
    - radars: dict con entradas a superponer en el radar del panel derecho
    - scalp: dict {"TP9":v, "AF7":v, "AF8":v, "TP10":v} con valores en [0,1]
    """

    def _to_list(v):
        if isinstance(v, (list, tuple)):
            return list(v)
        try:
            return [float(v)]
        except Exception:
            return list(np.atleast_1d(v).astype(float))

    t_list = _to_list(t)
    clean_series = {}
    if series_dict:
        clean_series = {k: _to_list(v) for k, v in (series_dict or {}).items()}
        for k, y in list(clean_series.items()):
            if len(y) == 1 and len(t_list) > 1:
                clean_series[k] = y * len(t_list)
        if len(t_list) == 1:
            max_len = max((len(v) for v in clean_series.values()), default=1)
            t_list = t_list * max_len

    q.put(
        {
            "t": t_list,
            "series": clean_series,  # legado (no usado en UI)
            "metric1": None if metric1 is None else float(metric1),
            "metric2": None if metric2 is None else float(metric2),
            "radars": radars or None,
            "scalp": scalp or None,
        }
    )


def shutdown(q):
    try:
        q.put({"_shutdown": True})
    except Exception:
        pass


# ------------------------------
# Helper para el scalp (topo-like) con 4 electrodos: TP9, AF7, AF8, TP10
# ------------------------------

def _scalp_positions():
    # Coordenadas 2D aproximadas en un disco unitario.
    # AF7/AF8 hacia arriba (frontal), TP9/TP10 hacia abajo (temporo-parietal/mastoideas).
    return {
        "AF7": (-0.40, 0.72),
        "AF8": ( 0.40, 0.72),
        "TP9": (-0.62,-0.58),
        "TP10":( 0.62,-0.58),
    }


def _idw_grid(values_dict, grid_res=121, power=2.0, eps=1e-6):
    """Interpolación simple por inverse-distance weighting (IDW) en un disco unitario.
    values_dict: {name: value in [0,1]}
    Devuelve X, Y, Z con NaNs fuera del círculo.
    """
    pos = _scalp_positions()
    # matriz de puntos conocidos
    keys = [k for k in ("AF7","AF8","TP9","TP10") if k in values_dict]
    if len(keys) == 0:
        # sin datos: devolver todo NaN
        X = Y = np.linspace(-1, 1, grid_res)
        X, Y = np.meshgrid(X, Y)
        Z = np.full_like(X, np.nan, dtype=float)
        return X, Y, Z

    P = np.array([pos[k] for k in keys], dtype=float)  # (M,2)
    V = np.array([float(values_dict[k]) for k in keys], dtype=float)  # (M,)

    xs = ys = np.linspace(-1.0, 1.0, grid_res)
    X, Y = np.meshgrid(xs, ys)
    # máscara de disco (cabeza)
    R2 = X**2 + Y**2
    mask = R2 <= 1.0

    # IDW: para cada grid point, w_i = 1/(d_i^p + eps)
    Z = np.full_like(X, np.nan, dtype=float)
    XY = np.stack([X[mask], Y[mask]], axis=1)  # (K,2)
    # distancias a cada electrodo
    # dists shape: (K, M)
    dists = np.sqrt(((XY[:,None,:] - P[None,:,:])**2).sum(axis=2))
    w = 1.0 / (np.power(dists, power) + eps)
    w_sum = np.sum(w, axis=1, keepdims=True)
    vals = (w @ V.reshape(-1,1)) / w_sum

    Z[mask] = vals.ravel()
    return X, Y, Z


def make_scalp_figure(scalp_values):
    """Crea figura Plotly con mapa de calor sobre la cabeza y marcas de electrodos."""
    X, Y, Z = _idw_grid(scalp_values, grid_res=121)
    fig = go.Figure()
    # Heatmap suave
    fig.add_trace(
        go.Heatmap(
            x=X[0], y=Y[:,0], z=Z,
            colorscale=PALETTE["scalp_colorscale"],
            zmin=0.0, zmax=1.0,
            colorbar=dict(title="Sincronía",),
            hoverinfo="skip",
        )
    )

    # Contorno de cabeza (círculo)
    circle = dict(
        type="circle",
        xref="x", yref="y",
        x0=-1, y0=-1, x1=1, y1=1,
        line=dict(color=PALETTE["border"], width=2),
    )
    # Nariz (pequeño triángulo)
    nose = dict(type="path", xref="x", yref="y",
                path="M 0 1 L -0.06 0.9 L 0.06 0.9 Z",
                line=dict(color=PALETTE["border"], width=1),
                fillcolor="rgba(0,0,0,0)")
    # Orejas
    ear_l = dict(type="circle", xref="x", yref="y",
                 x0=-1.05, y0=-0.25, x1=-0.85, y1=0.25,
                 line=dict(color=PALETTE["border"], width=1), fillcolor="rgba(0,0,0,0)")
    ear_r = dict(type="circle", xref="x", yref="y",
                 x0=0.85, y0=-0.25, x1=1.05, y1=0.25,
                 line=dict(color=PALETTE["border"], width=1), fillcolor="rgba(0,0,0,0)")

    # Marcas de electrodos
    pos = _scalp_positions()
    em_x = [pos[k][0] for k in pos]
    em_y = [pos[k][1] for k in pos]
    em_t = [f"{k}: {float(scalp_values.get(k, np.nan)):.2f}" for k in pos]

    fig.add_trace(
        go.Scatter(
            x=em_x, y=em_y, mode="markers+text",
            marker=dict(size=10, color="#666", line=dict(color="#FFF", width=1)),
            text=list(pos.keys()), textposition="top center",
            hovertext=em_t, hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1, range=[-1.15, 1.15]),
        yaxis=dict(visible=False, range=[-1.15, 1.15]),
        shapes=[circle, nose, ear_l, ear_r],
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return apply_light_theme(fig)


# ------------------------------
# Proceso del dashboard (Dash)
# ------------------------------

def run_dash(q, port=8051, window_sec=120, offset=0.8, title="Neuro Live", xmin=30.0):
    from collections import defaultdict, deque
    import numpy as np
    import dash
    from dash import dcc, html
    import plotly.graph_objects as go

    # Estado
    scalp_state = None  # dict de últimos valores
    radars_state = {}

    def _to_list(v):
        if isinstance(v, (list, tuple)):
            return list(v)
        try:
            return [float(v)]
        except Exception:
            return list(np.atleast_1d(v).astype(float))

    def drain_queue():
        nonlocal scalp_state, radars_state
        while True:
            try:
                msg = q.get_nowait()
            except Exception:
                break
            if isinstance(msg, dict) and msg.get("_shutdown"):
                import os
                os._exit(0)
            if not isinstance(msg, dict):
                continue
            # guarda último scalp
            s = msg.get("scalp")
            if isinstance(s, dict) and any(k in s for k in ("TP9","AF7","AF8","TP10")):
                scalp_state = s
            # radar
            rads = msg.get("radars")
            if isinstance(rads, dict) and len(rads) >= 1:
                radars_state = rads

    # ---------------- UI ----------------
    app = dash.Dash(__name__)
    app.title = title
    app.layout = html.Div(
        style={
            "backgroundColor": PALETTE["bg"],
            "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            "color": PALETTE["text"],
            "padding": "16px",
            "height": "100vh",
            "boxSizing": "border-box",
        },
        children=[
            html.H2(title, style={"margin": "0 0 12px 0", "opacity": 0.95}),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "2fr 1fr",
                    "gridTemplateRows": "1fr",
                    "gridGap": "16px",
                    "height": "calc(100vh - 70px)",
                    "minHeight": 0,
                    "overflow": "hidden",
                },
                children=[
                    # Columna izquierda: SCALP (topo-like)
                    html.Div(
                        style={
                            "background": PALETTE["panel"],
                            "borderRadius": "16px",
                            "padding": "8px",
                            "boxShadow": "0 2px 14px rgba(14, 24, 33, 0.06)",
                            "border": f"1px solid {PALETTE['border']}",
                            "height": "100%",
                            "minHeight": 0,
                            "overflow": "hidden",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                        children=[
                            html.Div(
                                "Mapa de sincronía (scalp)",
                                style={
                                    "opacity": 0.85,
                                    "padding": "4px 8px",
                                    "background": PALETTE["panel"],
                                    "color": PALETTE["subtext"],
                                },
                            ),
                            dcc.Graph(
                                id="scalp-graph",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "100%"},
                            ),
                        ],
                    ),
                    # Columna derecha: Radar con series sobrepuestas
                    html.Div(
                        style={
                            "background": PALETTE["panel"],
                            "borderRadius": "16px",
                            "padding": "8px",
                            "boxShadow": "0 2px 14px rgba(14, 24, 33, 0.06)",
                            "border": f"1px solid {PALETTE['border']}",
                            "minHeight": 0,
                            "display": "flex",
                            "flexDirection": "column",
                        },
                        children=[
                            html.Div(
                                "Métricas cognitivas",
                                style={
                                    "opacity": 0.85,
                                    "padding": "4px 8px",
                                    "background": PALETTE["panel"],
                                    "color": PALETTE["subtext"],
                                },
                            ),
                            dcc.Graph(
                                id="radar-graph",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "100%"},
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Interval(id="tick", interval=500, n_intervals=0),
            # --- Compat container for legacy callbacks ---
            html.Div(id="lines-container", style={"display":"none"}),
        ],
    )

    @app.callback(
            dash.Output("scalp-graph", "figure"),
            [dash.Input("tick", "n_intervals")],
            prevent_initial_call=False,
    )
    def _on_tick(_):
        drain_queue()        # ----- Scalp -----
        if isinstance(scalp_state, dict) and len(scalp_state) > 0:
            scalp_fig = make_scalp_figure(scalp_state)
        else:
            scalp_fig = go.Figure()
            scalp_fig.update_layout(showlegend=False)
            scalp_fig.add_annotation(text="Esperando datos…", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, opacity=0.5)
            apply_light_theme(scalp_fig)
        return scalp_fig

    def _make_overlay_radar(series_by_name):
        fig = go.Figure()
        if not isinstance(series_by_name, dict) or len(series_by_name) == 0:
            fig.update_layout(
                template=None,
                paper_bgcolor=PALETTE["panel"],
                font=dict(color=PALETTE["text"]),
                margin=dict(l=30, r=10, t=10, b=20),
                polar=dict(
                    radialaxis=dict(visible=True, gridcolor=PALETTE["grid"], linecolor=PALETTE["border"]),
                    angularaxis=dict(direction="clockwise", gridcolor=PALETTE["grid"]),
                    domain=dict(x=[0.15, 0.85], y=[0.15, 0.85]),
                ),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(255,255,255,0.6)",
                ),
            )
            fig.add_annotation(text="Sin datos…", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, opacity=0.5)
            return fig

        first_key = next(iter(series_by_name))
        cats = list(series_by_name[first_key].keys())

        colorway = ["#CDB4DB", "#A2D2FF", "#FFC8DD", "#BDE0FE", "#80ED99", "#FFF3B0"]
        for idx, (serie_name, values) in enumerate(series_by_name.items()):
            r = [float(values[c]) for c in cats]
            theta = cats + [cats[0]]
            rr = r + [r[0]]
            color = colorway[idx % len(colorway)]
            fig.add_trace(
                go.Scatterpolar(
                    r=rr, theta=theta, fill="toself", name=serie_name, opacity=0.35,
                    line=dict(width=2, color=color),
                )
            )

        fig.update_layout(
            template=None,
            paper_bgcolor=PALETTE["panel"],
            font=dict(color=PALETTE["text"]),
            margin=dict(l=30, r=10, t=10, b=20),
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0,1], tickmode="array", tickvals=[0.0,0.5,1.0],
                    ticktext=["Bajo","Medio","Alto"], gridcolor=PALETTE["grid"], linecolor=PALETTE["border"],
                ),
                angularaxis=dict(direction="clockwise", gridcolor=PALETTE["grid"]),
                domain=dict(x=[0.15, 0.85], y=[0.15, 0.85]),
            ),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                bgcolor="rgba(255,255,255,0.6)",
            ),
        )
        return fig

        # Legacy-compatible callback: returns lines-container.children and radar-graph.figure
    @app.callback(
        [dash.Output("lines-container", "children"), dash.Output("radar-graph", "figure")],
        [dash.Input("tick", "n_intervals")],
        prevent_initial_call=False,
    )
    def _compat_lines_and_radar(_):
        drain_queue()
        # lines-container vacío (ya no usamos lineplots)
        children = []
        radar_fig = _make_overlay_radar(radars_state)
        return children, radar_fig

    app.run(host="127.0.0.1", port=port, debug=False)

# neuro_dashboard.py (v2)
# Mantiene API: run_dash, push, shutdown

from collections import defaultdict, deque, OrderedDict
import os
import queue as _queue
import time
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

PALETTE = {
    "colorway": [
        "#4C78A8",
        "#F58518",
        "#E45756",
        "#72B7B2",
        "#54A24B",
        "#EECA3B",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ],
    "panel": "#FFFFFF",
    "border": "#E9EEF2",
    "primary": "#0D6EFD",
    "subtext": "#6B7280",
    "bg": "#F5F7FA",
    "grid": "rgba(0, 0, 0, 0.3)",
    "text": "#111827",
}


def _color_for_index(idx: int) -> str:
    return PALETTE["colorway"][idx % len(PALETTE["colorway"])]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{a})"


def apply_light_theme(fig: go.Figure):
    fig.update_layout(
        colorway=PALETTE["colorway"],
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"]),
        hoverlabel=dict(bgcolor="#FFFFFF"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        linecolor=PALETTE["border"],
        mirror=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        zeroline=False,
        linecolor=PALETTE["border"],
        mirror=False,
    )


def push(q, t, series_dict, metric1=None, metric2=None, radars=None):
    try:
        q.put_nowait(
            {
                "type": "data",
                "t": float(t),
                "series": dict(series_dict or {}),
                "radars": dict(radars or {}),
            }
        )
    except Exception:
        pass


def shutdown(q):
    try:
        q.put_nowait({"type": "shutdown"})
    except Exception:
        pass


def run_dash(q, port=8051, window_sec=120, offset=0.8, title="Neuro Live", xmin=30.0):
    MAX_POINTS = 10000
    xs = defaultdict(lambda: deque(maxlen=MAX_POINTS))
    ys = defaultdict(lambda: deque(maxlen=MAX_POINTS))
    order = []
    radars_state: "OrderedDict[str, dict]" = OrderedDict()
    shutdown_flag = False

    def ensure_series(name: str):
        if name not in ys:
            _ = xs[name]
            _ = ys[name]
        if name not in order:
            order.append(name)

    def drain_queue():
        nonlocal shutdown_flag, radars_state
        while True:
            try:
                msg = q.get_nowait()
            except _queue.Empty:
                break
            if not isinstance(msg, dict):
                continue
            mtype = msg.get("type")
            if mtype == "shutdown":
                shutdown_flag = True
                break
            if mtype == "data":
                t = msg.get("t", time.time())
                series = msg.get("series", {}) or {}
                radars = msg.get("radars", {}) or {}
                for nm, v in series.items():
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    ensure_series(nm)
                    xs[nm].append(float(t))
                    ys[nm].append(val)
                if isinstance(radars, dict) and radars:
                    for k in radars.keys():
                        if k not in radars_state:
                            radars_state[k] = {}
                    for layer, vals in radars.items():
                        radars_state[layer] = dict(vals)

    DEFAULT_CATS = [
        "Atención",
        "Relajación",
        "Activación",
        "Involucramiento",
        "Emoción<br>Positiva",
    ]

    def make_overlay_radar(state_dict: "OrderedDict[str, dict]") -> go.Figure:
        cats = list(DEFAULT_CATS)
        fig = go.Figure()
        if not state_dict:
            fig.add_trace(
                go.Scatterpolar(
                    r=[0] * (len(cats) + 1),
                    theta=cats + [cats[0]],
                    mode="lines",
                    line=dict(width=1.2, color=PALETTE["border"]),
                    name="",
                    hoverinfo="skip",
                )
            )
        else:
            for idx, (serie_name, values) in enumerate(state_dict.items()):
                r = [float(values.get(c, 0.0)) for c in cats]
                theta = cats + [cats[0]]
                rr = r + [r[0]]
                color = _color_for_index(idx)
                fill_col = _hex_to_rgba(color, 0.4)
                fig.add_trace(
                    go.Scatterpolar(
                        r=rr,
                        theta=theta,
                        mode="lines+markers",
                        marker=dict(size=8),
                        line=dict(width=4, color=color),
                        fill="toself",
                        fillcolor=fill_col,
                        name=serie_name,
                        textfont=dict(size=30),
                        opacity=1.0,
                    )
                )

        fig.update_layout(
            margin=dict(l=50, r=50, t=40, b=60),
            polar=dict(
                domain=dict(
                    x=[0.06, 0.94], y=[0.06, 0.94]
                ),  # ocupa el panel completo y se centra
                bgcolor=PALETTE["panel"],
                radialaxis=dict(
                    range=[0, 1.2],
                    tickvals=[0.0, 0.5, 1.0],
                    ticktext=["Bajo", "Medio", "Alto"],
                    tickfont=dict(size=20),
                    visible=True,
                    showline=False,
                    gridcolor=PALETTE["grid"],
                    gridwidth=1.4,
                    # ← ver sección 3 para la escala
                ),
                angularaxis=dict(
                    gridcolor=PALETTE["grid"],
                    tickfont=dict(size=22),
                    # tickpadding=12,  # separación de la circunferencia
                    rotation=90,
                    gridwidth=1.4,
                ),
            ),
            legend=dict(  # leyenda horizontal y centrada
                orientation="h",
                x=0.5,
                xanchor="center",
                y=1.08,
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.0)",
                borderwidth=0,
                font=dict(size=30),
            ),
            showlegend=True,
        )

        apply_light_theme(fig)
        return fig

    app = Dash(__name__)
    server = app.server

    app.layout = html.Div(
        style={
            "background": PALETTE["bg"],
            "minHeight": "100vh",
            "padding": "16px",
            "boxSizing": "border-box",
            "display": "flex",
            "flexDirection": "column",
            "gap": "12px",
            "fontFamily": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        },
        children=[
            html.Div(
                title,
                style={"fontWeight": 700, "fontSize": "4vh", "color": PALETTE["text"]},
            ),
            # El grid ahora ocupa toda la altura restante del viewport
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1.6fr 1fr",
                    "gap": "14px",
                    "alignItems": "stretch",
                    "height": "calc(100vh - 96px)",  # llena viewport y elimina espacio libre debajo
                    "minHeight": "520px",
                },
                children=[
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
                            "gap": "8px",
                        },
                        children=[
                            html.Div(
                                "Sincronía cerebral",
                                style={
                                    "opacity": 0.85,
                                    "padding": "4px 8px",
                                    "background": PALETTE["panel"],
                                    "color": PALETTE["subtext"],
                                    "fontSize": "3vh",
                                    "fontWeight": "600",
                                },
                            ),
                            html.Div(
                                id="lines-avg",
                                style={
                                    # Make this block exactly the upper half of the left panel
                                    "flex": "0 0 50%",
                                    # Center content both ways
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    # Big responsive font: clamp between 120px and 420px, scale with viewport width
                                    "fontSize": "clamp(120px, 22vw, 420px)",
                                    "fontWeight": 800,
                                    "lineHeight": "1",
                                    "color": PALETTE["primary"],
                                    "letterSpacing": "-0.5px",
                                },
                                children="—",
                            ),
                            html.Div(
                                id="lines-container",
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "6px",
                                    "minHeight": 0,
                                    # occupy the lower half
                                    "flex": "1 1 50%",
                                    "height": "100%",
                                },
                                children=[],
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "background": PALETTE["panel"],
                            "borderRadius": "16px",
                            "padding": "8px",
                            "boxShadow": "0 1px 14px rgba(14, 24, 33, 0.06)",
                            "border": f"1px solid {PALETTE['border']}",
                            "height": "100%",
                            "minHeight": 0,
                            # "overflow": "hidden",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "8px",
                        },
                        children=[
                            html.Div(
                                "Estado mental y emocional",
                                style={
                                    "opacity": 0.85,
                                    "padding": "4px 8px",
                                    "background": PALETTE["panel"],
                                    "color": PALETTE["subtext"],
                                    "fontSize": "3vh",
                                    "fontWeight": "600",
                                },
                            ),
                            dcc.Graph(
                                id="radar-graph",
                                config={"displayModeBar": False, "responsive": True},
                                style={
                                    "flex": "1 1 auto",
                                    "height": "100%",
                                    "width": "100%",
                                },
                                figure=make_overlay_radar(radars_state),
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Interval(id="tick", interval=300, n_intervals=0),
        ],
    )

    @app.callback(
        [
            Output("lines-container", "children"),
            Output("radar-graph", "figure"),
            Output("lines-avg", "children"),
        ],
        [Input("tick", "n_intervals")],
        prevent_initial_call=False,
    )
    def _on_tick(_n):
        nonlocal shutdown_flag
        drain_queue()
        if shutdown_flag:
            os._exit(0)

        names = order if order else list(ys.keys())
        last_ts = [xs[nm][-1] for nm in names if len(xs[nm])]
        tmax = max(last_ts) if last_ts else None

        n = max(len(names), 1)
        per_height_css = f"calc((100% - {6 * (n - 1)}px) / {n})"

        children = []
        for i, name in enumerate(names):
            if not len(xs[name]) or not len(ys[name]):
                continue

            x = list(xs[name])
            y = list(ys[name])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=name,
                    line=dict(width=2.2, color=_color_for_index(i)),
                    connectgaps=False,
                    hovertemplate="%{x:.2f}s — %{y:.4f}<extra>" + name + "</extra>",
                )
            )

            if tmax is not None:
                left = max(0.0, tmax - float(window_sec))
                if left < xmin:
                    left = max(0.0, tmax - max(xmin, float(window_sec)))
                fig.update_xaxes(range=[left, tmax])

            is_last = i == len(names) - 1
            fig.update_xaxes(
                showticklabels=is_last,
                title_text=("Tiempo (s)" if is_last else None),
                ticks="outside",
            )
            fig.update_yaxes(title=name, automargin=True)

            if is_last:
                fig.update_layout(margin=dict(l=40, r=16, t=2, b=34))
            else:
                fig.update_layout(margin=dict(l=40, r=16, t=2, b=2))

            fig.update_layout(showlegend=False, autosize=True)
            apply_light_theme(fig)

            children.append(
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": per_height_css, "width": "100%"},
                )
            )

        radar_fig = make_overlay_radar(radars_state)
        latest_vals = [ys[nm][-1] for nm in names if len(ys[nm])]
        avg_text = (
            f"{int(np.round(np.mean(latest_vals) * 100))}%" if latest_vals else "—"
        )
        return children, radar_fig, avg_text

    app.run(host="127.0.0.1", port=port, debug=False)

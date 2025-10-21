
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from datetime import datetime, timedelta
from typing import Tuple, Optional

DEFAULT_CSV = "/mnt/data/synthetic_training_data_20251014_143312.csv"

# -----------------------
# Utility / preprocessing
# -----------------------
def to_datetime_col(df: pd.DataFrame, col: str = "Timestamp") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["acc_mag"]  = np.sqrt(df["Accel_x"]**2 + df["Accel_y"]**2 + df["Accel_z"]**2)
    df["gyro_mag"] = np.sqrt(df["Gyro_x"]**2 + df["Gyro_y"]**2 + df["Gyro_z"]**2)
    df["acc_excess"] = (df["acc_mag"] - 9.8).abs()
    return df

def infer_active_state(
    df: pd.DataFrame,
    acc_excess_thr: float = 0.8,
    gyro_thr: float = 0.5,
    min_active_secs: float = 2.0,
    sampling_seconds: Optional[float] = None
) -> pd.DataFrame:
    df = df.copy()
    cond = (df["acc_excess"] > acc_excess_thr) | (df["gyro_mag"] > gyro_thr)
    active = cond.astype(int)

    if sampling_seconds is None and len(df) > 1:
        sampling_seconds = df["Timestamp"].diff().dt.total_seconds().median()
        if pd.isna(sampling_seconds) or sampling_seconds <= 0:
            sampling_seconds = 0.5

    min_len = max(1, int(round(min_active_secs / sampling_seconds)))

    def _enforce_min_runs(arr, min_len):
        arr = arr.copy()
        if len(arr) == 0:
            return arr
        start = 0
        while start < len(arr):
            val = arr[start]
            end = start + 1
            while end < len(arr) and arr[end] == val:
                end += 1
            run_len = end - start
            if run_len < min_len:
                prev = arr[start - 1] if start - 1 >= 0 else (arr[end] if end < len(arr) else val)
                arr[start:end] = prev
            start = end
        return arr

    df["ActiveState"] = _enforce_min_runs(active.values, min_len)
    return df

def compute_kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"active_pct": 0, "idle_pct": 0, "total_active_time": "0s", "transitions": 0}
    active = int(df["ActiveState"].sum())
    idle = n - active
    active_pct = 100 * active / n
    idle_pct = 100 - active_pct
    if "Timestamp" in df.columns and df["Timestamp"].notna().sum() > 1:
        dt = df["Timestamp"].diff().dt.total_seconds().median()
        if pd.isna(dt) or dt <= 0: dt = 0.5
        total_active_secs = active * dt
    else:
        total_active_secs = active * 0.5
    transitions = int(np.sum(np.diff(df["ActiveState"].values) != 0))

    def fmt_secs(s):
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        parts = []
        if h: parts.append(f"{h}h")
        if m: parts.append(f"{m}m")
        parts.append(f"{s}s")
        return " ".join(parts)

    return {
        "active_pct": round(active_pct, 2),
        "idle_pct": round(idle_pct, 2),
        "total_active_time": fmt_secs(total_active_secs),
        "transitions": transitions
    }

# -----------------------
# Chart builders
# -----------------------
def pie_active_idle(df):
    counts = df["ActiveState"].value_counts().rename(index={0: "Idle", 1: "Active"})
    fig = px.pie(values=counts.values, names=counts.index, hole=0.5, title="Active vs Idle")
    return fig

def line_activity_by_time(df, freq="1min"):
    if "Timestamp" not in df.columns:
        return go.Figure()
    g = df.set_index("Timestamp")["ActiveState"].resample(freq).mean().fillna(0) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g.values, mode="lines", name="% Active"))
    fig.update_layout(title=f"% Active over time ({freq})", xaxis_title="Time", yaxis_title="% Active", yaxis=dict(range=[0,100]))
    return fig

def bar_activity_by_hour(df):
    if "Timestamp" not in df.columns:
        return go.Figure()
    d = df.copy()
    d["hour"] = d["Timestamp"].dt.hour
    group = d.groupby("hour")["ActiveState"].mean().mul(100)
    fig = px.bar(x=group.index, y=group.values, labels={"x":"Hour", "y":"% Active"}, title="% Active by Hour")
    return fig

def heatmap_employee_hour(df):
    if "Employee_ID" not in df.columns:
        return go.Figure()
    d = df.copy()
    d["hour"] = d["Timestamp"].dt.hour
    p = d.pivot_table(index="Employee_ID", columns="hour", values="ActiveState", aggfunc="mean").fillna(0).mul(100)
    fig = px.imshow(p, aspect="auto", color_continuous_scale="Blues", title="% Active Heatmap (Employee × Hour)")
    return fig

def realtime_line(df, minutes=5):
    if "Timestamp" not in df.columns or len(df)==0:
        return go.Figure()
    end_time = df["Timestamp"].max()
    start_time = end_time - pd.Timedelta(minutes=minutes)
    d = df[(df["Timestamp"]>=start_time) & (df["Timestamp"]<=end_time)].copy()
    if len(d)==0:
        d = df.tail(500).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Timestamp"], y=d["acc_mag"], mode="lines", name="Accel Magnitude"))
    fig.add_trace(go.Scatter(x=d["Timestamp"], y=d["gyro_mag"], mode="lines", name="Gyro Magnitude", yaxis="y2"))
    fig.update_layout(
        title=f"Real-time Sensor (last {minutes} min)",
        xaxis_title="Time",
        yaxis=dict(title="Accel mag"),
        yaxis2=dict(title="Gyro mag", overlaying="y", side="right")
    )
    return fig

def hist_box_timeline(df):
    h1 = px.histogram(df, x="acc_excess", nbins=50, title="Distribution: Accel Excess (|a|-g)")
    d2 = df.copy()
    d2["State"] = d2["ActiveState"].map({0:"Idle",1:"Active"})
    b1 = px.box(d2, x="State", y="acc_excess", title="Accel Excess by State")
    d3 = df[["Timestamp","ActiveState"]].copy()
    d3["State"] = d3["ActiveState"].map({0:"Idle",1:"Active"})
    t1 = px.scatter(d3, x="Timestamp", y="ActiveState", color="State", title="State Timeline (0=Idle,1=Active)", height=300)
    t1.update_yaxes(range=[-0.2,1.2])
    return h1, b1, t1

def alerts_table(df, idle_threshold_min=10):
    if "Timestamp" not in df.columns or len(df)==0:
        return pd.DataFrame(columns=["Start","End","Duration(min)"])
    d = df[["Timestamp","ActiveState"]].copy()
    runs = []
    start = 0
    arr = d["ActiveState"].values
    for i in range(1, len(arr)+1):
        if i==len(arr) or arr[i]!=arr[i-1]:
            runs.append((start, i-1, arr[i-1]))
            start = i
    out = []
    for s,e,val in runs:
        if val==0:
            start_t = d.iloc[s]["Timestamp"]
            end_t = d.iloc[e]["Timestamp"]
            dur_min = (end_t - start_t).total_seconds()/60.0 if pd.notna(end_t) else 0.0
            if dur_min >= idle_threshold_min:
                out.append([start_t, end_t, round(dur_min,2)])
    res = pd.DataFrame(out, columns=["Start","End","Duration(min)"])
    return res

# -----------------------
# End-to-end pipeline
# -----------------------
def load_and_process(
    csv_file,
    acc_excess_thr: float,
    gyro_thr: float,
    min_active_secs: float,
    limit_time: tuple = (None, None)
):
    if csv_file is None:
        try:
            df = pd.read_csv(DEFAULT_CSV)
        except Exception as e:
            return None, f"Cannot read default CSV: {e}"
    else:
        try:
            df = pd.read_csv(csv_file.name if hasattr(csv_file, 'name') else csv_file)
        except Exception as e:
            return None, f"Cannot read uploaded CSV: {e}"

    required_cols = {"Timestamp","Accel_x","Accel_y","Accel_z","Gyro_x","Gyro_y","Gyro_z"}
    if not required_cols.issubset(df.columns):
        return None, f"Missing required columns. Found: {list(df.columns)}"

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    df = compute_features(df)
    df = infer_active_state(df, acc_excess_thr, gyro_thr, min_active_secs)

    start_str, end_str = limit_time
    if start_str:
        try:
            start = pd.to_datetime(start_str)
            df = df[df["Timestamp"]>=start]
        except: pass
    if end_str:
        try:
            end = pd.to_datetime(end_str)
            df = df[df["Timestamp"]<=end]
        except: pass

    return df, None

with gr.Blocks(title="Sensor EDA Dashboard — Active vs Idle") as demo:
    gr.Markdown("## Sensor EDA Dashboard — Active vs Idle (Gradio)")
    gr.Markdown("Upload CSV (or leave empty to use the default dataset). Tune thresholds to classify Active/Idle.")

    with gr.Row():
        csv_input = gr.File(label="Upload CSV", file_types=[".csv"], value=None)
        start_time = gr.Textbox(label="Start time (optional, e.g., 2023-01-01 10:00:00)")
        end_time = gr.Textbox(label="End time (optional, e.g., 2023-01-01 12:00:00)")

    with gr.Row():
        acc_thr = gr.Slider(0.2, 3.0, value=0.8, step=0.1, label="Accel Excess Threshold (|a|-g)")
        gyro_thr = gr.Slider(0.1, 3.0, value=0.5, step=0.1, label="Gyro Magnitude Threshold")
        min_active = gr.Slider(0.0, 10.0, value=2.0, step=0.5, label="Min Active Duration (seconds)")

    run_btn = gr.Button("Run / Refresh")

    with gr.Tabs():
        with gr.Tab("Overview"):
            kpi_active = gr.HTML()
            kpi_idle = gr.HTML()
            kpi_total = gr.HTML()
            kpi_trans = gr.HTML()
            with gr.Row():
                pie_plot = gr.Plot()
                line_plot = gr.Plot()
            with gr.Row():
                bar_plot = gr.Plot()
                heatmap_plot = gr.Plot()

        with gr.Tab("Live Monitoring"):
            live_plot = gr.Plot()
            alert_table = gr.Dataframe(headers=["Start","End","Duration(min)"], interactive=False)

        with gr.Tab("Activity Analysis"):
            line_ag = gr.Plot()
            hist_plot = gr.Plot()
            box_plot = gr.Plot()
            strip_plot = gr.Plot()

        with gr.Tab("Evaluation & Alerts"):
            drift_plot = gr.Plot()
            outlier_plot = gr.Plot()
            quality_html = gr.HTML()

    state_df = gr.State()

    def controller(csv_file, a_thr, g_thr, min_act, start_str, end_str):
        df, err = load_and_process(csv_file, a_thr, g_thr, min_act, (start_str, end_str))
        if err:
            empty_fig = go.Figure()
            return (
                f"<b>Error:</b> {err}", "", "", "",
                empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, pd.DataFrame(columns=['Start','End','Duration(min)']),
                empty_fig, empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, "<i>No data</i>"
            )

        k = compute_kpis(df)
        k_active = f"<div style='font-size:20px'><b>Active %:</b> {k['active_pct']}%</div>"
        k_idle   = f"<div style='font-size:20px'><b>Idle %:</b> {k['idle_pct']}%</div>"
        k_total  = f"<div style='font-size:20px'><b>Total Active Time:</b> {k['total_active_time']}</div>"
        k_trans  = f"<div style='font-size:20px'><b>Transitions:</b> {k['transitions']}</div>"

        fig_pie = pie_active_idle(df)
        fig_line = line_activity_by_time(df, "1min")
        fig_bar = bar_activity_by_hour(df)
        fig_heat = heatmap_employee_hour(df)

        fig_live = realtime_line(df, minutes=5)
        tbl_alerts = alerts_table(df, idle_threshold_min=10)

        fig_ag = go.Figure()
        fig_ag.add_trace(go.Scatter(x=df["Timestamp"], y=df["acc_mag"], mode="lines", name="Accel mag"))
        fig_ag.add_trace(go.Scatter(x=df["Timestamp"], y=df["gyro_mag"], mode="lines", name="Gyro mag", yaxis="y2"))
        fig_ag.update_layout(title="Accel & Gyro Magnitude", xaxis_title="Time",
                             yaxis=dict(title="Accel mag"),
                             yaxis2=dict(title="Gyro mag", overlaying="y", side="right"))
        h1, b1, t1 = hist_box_timeline(df)

        d = df.set_index("Timestamp")["acc_mag"].rolling("5min").mean().reset_index()
        fig_drift = px.line(d, x="Timestamp", y="acc_mag", title="Rolling Mean Accel Magnitude (5 min)")
        z = (df["acc_mag"] - df["acc_mag"].mean()) / (df["acc_mag"].std() + 1e-9)
        fig_out = px.scatter(x=df["Timestamp"], y=df["acc_mag"], color=(z.abs()>3),
                             labels={"x":"Time","y":"Accel mag","color":"Outlier"},
                             title="Outlier Detection (|z|>3)")

        missing_ratio = df.isna().mean().mean()*100
        quality = f"<div><b>Missing ratio (all cols):</b> {missing_ratio:.2f}%</div>"

        return (
            k_active, k_idle, k_total, k_trans,
            fig_pie, fig_line, fig_bar, fig_heat,
            fig_live, tbl_alerts,
            fig_ag, h1, b1, t1,
            fig_drift, fig_out, quality
        )

    run_btn.click(
        controller,
        inputs=[csv_input, acc_thr, gyro_thr, min_active, start_time, end_time],
        outputs=[
            kpi_active, kpi_idle, kpi_total, kpi_trans,
            pie_plot, line_plot, bar_plot, heatmap_plot,
            live_plot, alert_table,
            line_ag, hist_plot, box_plot, strip_plot,
            drift_plot, outlier_plot, quality_html
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

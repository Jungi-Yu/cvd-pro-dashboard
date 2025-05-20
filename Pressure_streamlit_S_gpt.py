import streamlit as st
import pandas as pd
import numpy as np
import base64, io, zipfile, chardet, logging
import plotly.graph_objs as go
import plotly.express as px
from scipy import stats
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="CVD Pro Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# 스타일
st.markdown("""
<style>
body, .stApp { background-color: #1e1e2e !important; color: #e0e0e0 !important; }
.stSidebar { background-color: #27293d !important; }
.stSidebar, .stSidebar * { color: #f5f5f5 !important; }
[data-testid="stFileUploader"] * { color: #f5f5f5 !important; }
[data-testid="stFileUploader"] button { background-color: #44475a !important; color: #f8f8f2 !important; }
.css-1avcm0n, .css-1n76uvr, .st-bb, label[for*="select"], .stRadio { color: #f5f5f5 !important; }
input, select, .stSlider, .stRangeSlider { background-color: #34354a !important; color: #f5f5f5 !important; }
div[role="tab"] { color: #e0e0e0 !important; }
.plotly .xtick text, .plotly .ytick text { fill: #e0e0e0 !important; }
.plotly .gtitle, .plotly .subplottitle { fill: #ffffff !important; }
.plotly .legend text { fill: #e0e0e0 !important; }
[data-testid='stMetricValue'] { font-size:2.5rem !important; color: #ffffff !important; }
h2, h3 { color: #ffffff !important; }
[data-testid="stDataFrame"] { background-color: #1e1e2e !important; }
[data-testid="stDataFrame"] * { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

st.sidebar.title("📂 파일 업로드 & 모드")
uploads = st.sidebar.file_uploader("XLSX/CSV/ZIP 업로드", type=["xlsx", "csv", "zip"], accept_multiple_files=True)
mode = st.sidebar.radio("모드 선택", ["대시보드", "이상치 탐지", "파일 비교", "다운로드"])

PARAM_GROUPS = [
    (['ZONE1_Furnace','ZONE2_Furnace','ZONE3_Furnace','ZONE4_Furnace','ZONE5_Furnace'], "가열로 온도 (°C)"),
    (['ZONE1_Internal','ZONE2_Internal','ZONE3_Internal','ZONE4_Internal','ZONE5_Internal'], "내부 온도 (°C)"),
    ([f'MFC-{i}' for i in range(1,15)], "가스 유량 (sccm)"),
    (['ChamberPressure'], "로내 압력 (hPa)"),
    (['EvapoPressure'], "에바포 압력 (hPa)"),
    (['ALGENPressure'], "AL-GENE 압력 (hPa)"),
    (['CH3CN'], "CH3CN (ml/min)"),
    (['Evaporator'], "에바포레이터 (unit)")
]
COLORS = px.colors.qualitative.Plotly
HEADERS = ['Process', 'Timestamp'] + [col for grp, _ in PARAM_GROUPS for col in grp]

def parse_file(b64, name, raw_bytes):
    raw = base64.b64decode(b64)
    skip = 4 if name.lower().endswith('trou.csv') else 3
    if name.lower().endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(raw), skiprows=skip, header=None, usecols="A:AE", names=HEADERS)
    else:
        enc = chardet.detect(raw)['encoding'] or 'cp949'
        df = pd.read_csv(io.StringIO(raw.decode(enc, 'replace')), skiprows=skip,
                         header=None, names=HEADERS, usecols=range(0, len(HEADERS)), low_memory=False)
    for c in HEADERS[2:]:  # Process와 Timestamp 제외
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    df.attrs['fn'] = name
    return df

@st.cache_data
def load_data(files):
    dfs = []
    for up in files:
        raw = up.read()
        if up.name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for fn in z.namelist():
                    if fn.lower().endswith(('.xlsx', '.csv')):
                        b = z.read(fn)
                        dfs.append(parse_file(base64.b64encode(b).decode(), fn, b))
        else:
            dfs.append(parse_file(base64.b64encode(raw).decode(), up.name, raw))
    return dfs

# 공정 지속 시간 계산 함수
def calculate_process_durations(df):
    durations = []
    process_changes = df[df['Process'] != df['Process'].shift()][['Timestamp', 'Process']].reset_index()
    process_changes = pd.concat([process_changes, pd.DataFrame({
        'index': [df.index[-1]],
        'Timestamp': [df['Timestamp'].iloc[-1]],
        'Process': [df['Process'].iloc[-1]]
    })], ignore_index=True)
    
    for i in range(len(process_changes) - 1):
        start_time = process_changes['Timestamp'].iloc[i]
        end_time = process_changes['Timestamp'].iloc[i + 1]
        process_name = process_changes['Process'].iloc[i]
        duration = end_time - start_time
        duration_minutes = duration.total_seconds() / 60
        duration_str = f"{int(duration_minutes // 60)}시간 {int(duration_minutes % 60)}분" if duration_minutes >= 60 else f"{int(duration_minutes)}분"
        durations.append({
            'Process': process_name,
            'Start Time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'End Time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Duration': duration_str
        })
    
    return pd.DataFrame(durations)

# 공정 경과 시간 계산 함수
def calculate_elapsed_time(df):
    elapsed_times = []
    process_changes = df[df['Process'] != df['Process'].shift()]['Timestamp'].reset_index()
    process_changes = pd.concat([process_changes, pd.DataFrame({
        'index': [df.index[-1]],
        'Timestamp': [df['Timestamp'].iloc[-1]]
    })], ignore_index=True)
    
    for i in range(len(process_changes) - 1):
        start_idx = process_changes['index'].iloc[i]
        end_idx = process_changes['index'].iloc[i + 1]
        start_time = process_changes['Timestamp'].iloc[i]
        for idx in range(start_idx, end_idx):
            elapsed = (df['Timestamp'].iloc[idx] - start_time).total_seconds() / 60
            elapsed_times.append(f"{int(elapsed)}분")
    
    # 마지막 인덱스 처리
    if len(elapsed_times) < len(df):
        elapsed_times.extend([elapsed_times[-1]] * (len(df) - len(elapsed_times)))
    
    return elapsed_times

if not uploads:
    st.info("파일을 업로드해주세요.")
    st.stop()

dfs = load_data(uploads)
if len(dfs) < 1:
    st.warning("1개 이상의 파일을 업로드해야 합니다.")
    st.stop()

# 범위 계산: 전체 파일 중 min/max
range_values = {}
for params, label in PARAM_GROUPS:
    min_v, max_v = np.inf, -np.inf
    for df in dfs:
        for param in params:
            if df[param].dropna().empty:
                continue
            min_val = df[param].min()
            max_val = df[param].max()
            if pd.notna(min_val): min_v = min(min_v, min_val)
            if pd.notna(max_val): max_v = max(max_v, max_val)
    if not np.isinf(min_v) and not np.isinf(max_v):
        margin = (max_v - min_v) * 0.1
        range_values[label] = (min_v - margin, max_v + margin)
    else:
        range_values[label] = (0, 100)

# 대시보드
if mode == "대시보드":
    st.title("🚀 CVD Pro Analytics Dashboard")
    st.subheader("업로드된 파일")
    cols = st.columns(len(dfs))
    for col, df in zip(cols, dfs): col.metric(df.attrs['fn'], f"{len(df)} 행")
    
    # 공정 지속 시간 표시
    st.subheader("공정별 지속 시간")
    for df in dfs:
        st.markdown(f"#### {df.attrs['fn']}")
        durations_df = calculate_process_durations(df)
        st.dataframe(durations_df, use_container_width=True)
    
    st.subheader("파일별 트렌드")
    grid = st.columns(min(4, len(dfs)))
    height = 350
    
    # 토글 버튼
    show_process = st.sidebar.checkbox("공정명 및 선 표시", value=False)
    show_hover_process = st.sidebar.checkbox("호버에 공정명 및 경과 시간 포함", value=True)

    for idx, df in enumerate(dfs):
        with grid[idx % len(grid)]:
            st.markdown(f"#### {df.attrs['fn']}")
            # 데이터 디버깅
            st.write(f"데이터프레임 정보 ({df.attrs['fn']}):")
            st.write(f"Timestamp 열: {df['Timestamp'].head(5)}")
            st.write(f"Process 열: {df['Process'].head(5)}")
            
            # 공정 경과 시간 계산
            elapsed_times = calculate_elapsed_time(df)
            
            for params, label in PARAM_GROUPS:
                fig = go.Figure()
                for i, p in enumerate(params):
                    hover_text = (
                        'Value: %{y}<br>' +
                        'Time: %{x}<br>'
                    )
                    if show_hover_process:
                        hover_text += 'Process: %{customdata[0]}<br>'
                        hover_text += 'Elapsed Time: %{customdata[1]}<br>'
                    hover_text += '<extra></extra>'
                    
                    fig.add_trace(go.Scatter(
                        x=df['Timestamp'], y=df[p], mode='lines', name=p,
                        line=dict(color=COLORS[i % len(COLORS)], width=2),
                        customdata=list(zip(df['Process'], elapsed_times)) if show_hover_process else None,
                        hovertemplate=hover_text,
                        hoverlabel=dict(
                            bgcolor="#1e1e2e",
                            font=dict(color="#e0e0e0"),
                            align="left",
                            namelength=-1
                        )
                    ))
                process_changes = df[df['Process'] != df['Process'].shift()]['Timestamp']
                last_timestamp = None
                total_points = len(process_changes)
                if total_points > 0:
                    # 처음, 중간(1/3, 2/3), 끝 지점만 표시
                    indices_to_show = [0, max(0, total_points // 3 - 1), max(0, 2 * total_points // 3 - 1), total_points - 1]
                    for idx in indices_to_show:
                        ts = process_changes.iloc[idx] if idx < total_points else df['Timestamp'].iloc[-1]
                        if pd.notna(ts):
                            fig.add_annotation(
                                x=ts,
                                y=-0.15,
                                yref="paper",
                                text=ts.strftime('%H:%M'),  # 시간:분 형식으로 간소화
                                showarrow=False,
                                font=dict(color="#e0e0e0", size=6),  # 폰트 크기 추가 축소
                                xanchor="center",
                                align="center"
                            )
                        # 토글 활성화 시 선과 공정명 표시
                        if show_process and pd.notna(ts):
                            process_name = df[df['Timestamp'] == ts]['Process'].iloc[0] if not df[df['Timestamp'] == ts]['Process'].empty else "N/A"
                            if pd.notna(process_name):
                                fig.add_shape(
                                    type="line",
                                    x0=ts, x1=ts,
                                    y0=0, y1=1,
                                    yref="paper",
                                    line=dict(
                                        color="#000000",
                                        width=0.5,
                                        dash="solid"
                                    )
                                )
                                fig.add_annotation(
                                    x=ts,
                                    y=1,
                                    yref="paper",
                                    text=process_name,
                                    showarrow=False,
                                    font=dict(color="#ffffff", size=10),
                                    xanchor="left",
                                    yanchor="top"
                                )
                ymin, ymax = range_values[label]
                fig.update_yaxes(range=[ymin, ymax])
                fig.update_layout(
                    title=label,
                    template='plotly_dark',
                    font_color='#e0e0e0',
                    paper_bgcolor='#1e1e2e',
                    plot_bgcolor='#1e1e2e',
                    showlegend=False,  # 범례 제거
                    xaxis=dict(
                        tickfont=dict(color='#e0e0e0'),
                        type="date",
                        side="bottom",
                        range=[df['Timestamp'].min(), df['Timestamp'].max()]
                    ),
                    yaxis=dict(tickfont=dict(color='#e0e0e0')),
                    dragmode='zoom',
                    uirevision='true',
                    height=height,
                    margin=dict(t=50, b=150),
                    hovermode='x unified',
                    hoverlabel=dict(
                        bgcolor="#1e1e2e",
                        font=dict(color="#e0e0e0"),
                        align="left",
                        namelength=-1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

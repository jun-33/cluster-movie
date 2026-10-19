
import streamlit as st
import pandas as pd
import math
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

FEATURES = {
    "스크린 수(로그)": "first_scrn_log",
    "누적 관객(로그)": "total_audi_log",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "longrun_index",
}
CLUSTER_LABELS = ["㉮", "㉯", "㉰"]


# =========================================================
# 데이터 불러오기 및 전처리
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: " + ", ".join(missing_columns)
        )

    for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["movieNm"] = df["movieNm"].fillna("제목 없음").astype(str)

    # 원래 단위의 네 가지 특성
    df["롱런 지수"] = df["total_audi"] / df["first_week_audi"]
    df.loc[df["first_week_audi"] <= 0, "롱런 지수"] = float("nan")
    df["롱런 지수"] = df["롱런 지수"].clip(upper=20)
    df["스크린 수"] = df["first_scrn"]
    df["누적 관객"] = df["total_audi"]
    df["10위권 일수"] = df["days_in_top10"]

    # 군집화에 사용할 변환 특성
    # 로그는 양수에 대해서만 정의되므로 0 이하 값도 제외 대상이 됨
    df["first_scrn_log"] = df["first_scrn"].apply(
        lambda value: math.log10(value) if pd.notna(value) and value > 0 else float("nan")
    )
    df["total_audi_log"] = df["total_audi"].apply(
        lambda value: math.log10(value) if pd.notna(value) and value > 0 else float("nan")
    )
    df["days_in_top10"] = pd.to_numeric(df["days_in_top10"], errors="coerce")
    df["longrun_index"] = df["롱런 지수"]

    # 네 속성 중 값이 없거나 첫 주 관객이 0 이하인 영화 제외
    required_features = list(FEATURES.values())
    valid = (
        df[required_features].notna().all(axis=1)
        & (df["first_week_audi"] > 0)
        & (df["first_scrn"] > 0)
        & (df["total_audi"] > 0)
    )
    cleaned = df.loc[valid].copy()
    return df, cleaned


st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 스크린 수, 누적 관객, 10위권 일수, 롱런 지수를 바탕으로 "
    "비슷한 특성을 가진 영화 세 유형을 찾아봅니다."
)

try:
    raw_df, valid_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오거나 처리하지 못했습니다: {e}")
    st.info("데이터 URL과 CSV 열 이름, 인터넷 연결을 확인해 주세요.")
    st.stop()

st.caption(
# =========================================================
# 3차원 산점도
# =========================================================
st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 군집화에 사용할 속성을 세 개 이상 선택해 주세요.")
else:
    default_indices = [0, 1, 2]
    axis_cols = st.columns(3)
    axis_keys = ["scatter3d_x", "scatter3d_y", "scatter3d_z"]
    axis_labels = ["X축 속성", "Y축 속성", "Z축 속성"]
    axis_values = []
    for i, col in enumerate(axis_cols):
        with col:
            axis_values.append(
                st.selectbox(
                    axis_labels[i],
                    options=selected_features,
                    index=default_indices[i],
                    key=axis_keys[i]
                )
            )

    plot3d = clustered.copy()
    plot3d["_x3"] = plot3d[FEATURES[axis_values[0]]]
    plot3d["_y3"] = plot3d[FEATURES[axis_values[1]]]
    plot3d["_z3"] = plot3d[FEATURES[axis_values[2]]]

    fig3d = px.scatter_3d(
        plot3d,
        x="_x3",
        y="_y3",
        z="_z3",
        color="영화 유형",
        hover_name="movieNm",
        hover_data={
            "_x3": False,
            "_y3": False,
            "_z3": False,
            "영화 유형": True,
            "genre": True,
            "nation": True,
        },
        labels={
            "_x3": axis_values[0],
            "_y3": axis_values[1],
            "_z3": axis_values[2],
            "영화 유형": "영화 유형"
        },
        title="영화 유형별 3차원 분포"
    )
    fig3d.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3d.update_layout(
        legend_title_text="영화 유형",
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title=axis_values[0],
            yaxis_title=axis_values[1],
            zaxis_title=axis_values[2],
        )
    )
    st.plotly_chart(fig3d, use_container_width=True)


# =========================================================
# 유형별 평균 통계
# =========================================================
st.subheader("4. 유형별 요약")
summary_rows = []
for label in CLUSTER_LABELS:
    group = clustered[clustered["영화 유형"] == label]
    if group.empty:
        continue
    summary_rows.append({
        "영화 유형": label,
        "영화 편수": len(group),
        "스크린 수 평균": group["스크린 수"].mean(),
        "누적 관객 평균": group["누적 관객"].mean(),
        "10위권 일수 평균": group["10위권 일수"].mean(),
        "롱런 지수 평균": group["롱런 지수"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.rename(columns={
    "스크린 수 평균": "스크린 수 평균(개)",
    "누적 관객 평균": "누적 관객 평균(명)",
    "10위권 일수 평균": "10위권 일수 평균(일)",
})
for col in ["스크린 수 평균(개)", "누적 관객 평균(명)", "10위권 일수 평균(일)", "롱런 지수 평균"]:
    summary_df[col] = summary_df[col].round(2)

st.dataframe(summary_df, use_container_width=True, hide_index=True)


# =========================================================
# 유형별 누적 관객 상위 5편
# =========================================================
st.subheader("5. 유형별 누적 관객 상위 5편")
for label in CLUSTER_LABELS:
    group = clustered[clustered["영화 유형"] == label].sort_values(
        "total_audi", ascending=False
    ).head(5)

    if group.empty:
        continue

    with st.expander(f"영화 유형 {label} — 누적 관객 상위 {len(group)}편", expanded=True):
        top_table = group[["movieNm", "total_audi", "first_week_audi", "first_scrn", "days_in_top10"]].copy()
        top_table = top_table.rename(columns={
            "movieNm": "영화 제목",
            "total_audi": "누적 관객(명)",
            "first_week_audi": "첫 주 관객(명)",
            "first_scrn": "스크린 수(개)",
            "days_in_top10": "10위권 일수(일)",
        })
        st.dataframe(top_table, use_container_width=True, hide_index=True)

st.caption(
    "참고: K-평균은 선택한 속성을 표준화한 값으로 군집화합니다. "
    "요약 표와 상위 영화 목록은 원래 단위의 데이터로 계산합니다. "
    "군집은 정답이 있는 분류가 아니라 특성이 비슷한 영화를 자동으로 묶은 결과입니다."
)


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import requests
from datetime import datetime 
from st_pages import Page, show_pages

show_pages([
    Page("자치구대시보드.py", "자치구 대시보드", "📊"),
    Page("pages/단위도서관대시보드.py", "개별 도서관 대시보드", "📚"),
])

st.set_page_config(page_title="서울시 도서관 대시보드", layout="wide")
st.title("📊 서울시 자치구별 현황")

# 자치구 코드 ↔ 이름 매핑
gu_dict = {
    1: '강남구', 2: '강동구', 3: '강북구', 4: '강서구', 5: '관악구', 6: '광진구',
    7: '구로구', 8: '금천구', 9: '노원구', 10: '도봉구', 11: '동대문구', 12: '동작구',
    13: '마포구', 14: '서대문구', 15: '서초구', 16: '성동구', 17: '성북구', 18: '송파구',
    19: '양천구', 20: '영등포구', 21: '용산구', 22: '은평구', 23: '종로구', 24: '중구', 25: '중랑구'
}
reverse_gu_dict = {v: k for k, v in gu_dict.items()}

# 파일 불러오기
pop_df = pd.read_csv("자치구별_연령별_성별_인구수_UTF8BOM_sorted.csv")
xlsx_path = "2024년 서울시 공공도서관 서비스 성과조사(이용자)_data(F)_송부.xlsx"
df = pd.read_excel(xlsx_path)

# ------------------------ 공통 처리 ------------------------
question_cols = [col for col in df.columns if col.startswith(("Q1_", "Q2_", "Q3_", "Q4_", "Q5_", "Q6_"))]
df = df[['SQ3'] + question_cols].rename(columns={"SQ3": "자치구"})

def convert_score(x):
    return (x - 1) * 100 / 6 if pd.notna(x) and x != 9 else None

def classify_section(q):
    if q.startswith("Q1_"): return "공간이용"
    if q.startswith("Q2_"): return "정보활용"
    if q.startswith("Q3_"): return "소통정책"
    if q.startswith("Q4_"): return "문화교육"
    if q.startswith("Q5_"): return "사회관계"
    if q.startswith("Q6_"): return "장기효과"





# ------------------------ 인구 시각화 ------------------------
gu_list = sorted(pop_df['자치구'].unique())
selected_gu = st.selectbox("자치구 선택", gu_list)
selected_gu_code = reverse_gu_dict[selected_gu]

gu_df = pop_df[pop_df['자치구'] == selected_gu]
age_order = gu_df['연령'].unique().tolist()
age_df = gu_df.groupby('연령')['인구수'].sum().reindex(age_order)
bar_x = age_df.index.tolist()
bar_y = age_df.values.tolist()

color1 = "#1959a8"
color2 = "#fa6e30"
bar_colors = []
found_old = False
for label in bar_x:
    if any(x in label for x in ["65", "70", "75", "80", "85", "90", "95"]):
        found_old = True
    bar_colors.append(color2 if found_old else color1)

male_sum = gu_df[gu_df['성별'] == '남자']['인구수'].sum()
female_sum = gu_df[gu_df['성별'] == '여자']['인구수'].sum()
total_sum = male_sum + female_sum
percent_man = male_sum / total_sum if total_sum else 0
percent_woman = female_sum / total_sum if total_sum else 0

option_man = {
    "series": [{
        "type": "liquidFill",
        "shape": "circle",
        "data": [percent_man],
        "backgroundStyle": {"color": "#fff"},
        "outline": {"show": True, "borderDistance": 0, "itemStyle": {"borderWidth": 2, "borderColor": "#5AC1F2"}},
        "color": ["#5AC1F2"],
        "label": {"normal": {"formatter": "남성\n{:.1f}%".format(percent_man * 100), "fontSize": 16, "fontWeight": "bold", "color": "#5AC1F2"}}
    }]
}
option_woman = {
    "series": [{
        "type": "liquidFill",
        "shape": "circle",
        "data": [percent_woman],
        "backgroundStyle": {"color": "#fff"},
        "outline": {"show": True, "borderDistance": 0, "itemStyle": {"borderWidth": 2, "borderColor": "#F576AB"}},
        "color": ["#F576AB"],
        "label": {"normal": {"formatter": "여성\n{:.1f}%".format(percent_woman * 100), "fontSize": 16, "fontWeight": "bold", "color": "#F576AB"}}
    }]
}

st.markdown("### 👥 인구 구조")
col_pop1, col_pop2 = st.columns([1, 4])
with col_pop1:
    st_echarts(option_man, height="120px")
    st_echarts(option_woman, height="120px")
with col_pop2:
    fig = go.Figure(data=[go.Bar(x=bar_x, y=bar_y, marker_color=bar_colors)])
    fig.update_layout(title="연령별 인구수 (남+여 합계)", xaxis_title="연령대", yaxis_title="인구수",
                      yaxis=dict(tickformat=",", separatethousands=True), margin=dict(l=30, r=20, t=40, b=40), height=350)
    st.plotly_chart(fig, use_container_width=True)

# 좌우 컬럼 레이아웃
left_col, _, right_col = st.columns([1.08, 0.04, 0.92])
with left_col:
    # ------------------------ 수급률 강조 ------------------------
    # 자치구 통합 데이터 로드 (중복 방지를 위해 if 필요시 상단으로 이동)
    df_gu = pd.read_csv("자치구 데이터 통합본 (연령, 성별제외).csv")
    df_gu.rename(columns={df_gu.columns[0]: "자치구"}, inplace=True)

    def make_welfare_stat_block(gu_name):
        value = df_gu[df_gu["자치구"] == gu_name]["수급률"].values[0]
        value_fmt = f"{value:.1f}%"

        seoul_avg = 5.044  # 서울 전체 자치구 평균 수급률
        return f"""
        <div style="padding: 10px; background-color: #f8f9fa; border: 1px solid #ddd;
                    border-radius: 8px; text-align: center; font-size: 20px;">
            <strong>수급률</strong><br>
            <span style="font-size: 36px; color: #0d6efd;"><strong>{value_fmt}</strong></span><br>
            <span style="font-size: 14px; color: #dc3545;">서울 평균: {seoul_avg:.1f}%</span>
        </div>
        """

    st.markdown("### 💠 자치구 수급률")
    st.markdown(make_welfare_stat_block(selected_gu), unsafe_allow_html=True)



    # ------------------------ 다문화 국적 구성 (상위 5개 + 기타) ------------------------
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("### 🌐 다문화 국적 구성 비율")

    # NA 처리 및 숫자형 변환
    df_gu = pd.read_csv("자치구 데이터 통합본 (연령, 성별제외).csv", na_values="NA")
    df_gu.rename(columns={df_gu.columns[0]: "자치구"}, inplace=True)
    df_gu.iloc[:, 2:34] = df_gu.iloc[:, 2:34].apply(pd.to_numeric, errors="coerce")

    multicultural_cols = df_gu.columns[2:34]
    row = df_gu[df_gu["자치구"] == selected_gu][multicultural_cols].iloc[0]

    if row.sum() == 0 or row.dropna().empty:
        st.warning("선택한 자치구에는 다문화 국적 데이터가 없습니다.")
    else:
        # 상위 5개 + 기타 계산
        top5 = row.sort_values(ascending=False).head(5)
        others_sum = row.sum() - top5.sum()
        top5["기타"] = others_sum

        percentages = top5 / top5.sum() * 100

        fig = go.Figure()
        for col, val in zip(percentages.index, percentages.values):
            fig.add_trace(go.Bar(
                x=[val],
                y=["다문화 국적 구성"],
                orientation='h',
                name=col,
                text=f"{val:.1f}%",
                textposition='inside'
            ))

        fig.update_layout(
            barmode='stack',
            height=140,
            margin=dict(t=30, b=30),
            xaxis=dict(range=[0, 100], showticklabels=False),
            yaxis=dict(showticklabels=False),
            title="다문화 국적 구성 비율 (상위 5개 + 기타)",
            legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center")
        )

        st.plotly_chart(fig, use_container_width=True)

    # ------------------------ 장애 유형 100% 누적 막대그래프 ------------------------
    import plotly.graph_objects as go

    st.markdown("### ♿ 장애 유형별 인원 구성 비율")

    # NA 처리 및 숫자형 변환
    df_gu = pd.read_csv("자치구 데이터 통합본 (연령, 성별제외).csv", na_values="NA")
    df_gu.rename(columns={df_gu.columns[0]: "자치구"}, inplace=True)
    df_gu.iloc[:, 35:41] = df_gu.iloc[:, 35:41].apply(pd.to_numeric, errors="coerce")

    disability_cols = df_gu.columns[35:41]
    row = df_gu[df_gu["자치구"] == selected_gu][disability_cols].iloc[0]

    if row.sum() == 0 or row.dropna().empty:
        st.warning("선택한 자치구에는 장애 유형 데이터가 없습니다.")
    else:
        total = row.sum()
        percentages = row / total * 100

        fig = go.Figure()
        for col, val in zip(percentages.index, percentages.values):
            fig.add_trace(go.Bar(
                x=[val],
                y=["장애 유형 구성"],
                orientation='h',
                name=col,
                text=f"{val:.1f}%",
                textposition='inside'
            ))

        fig.update_layout(
            barmode='stack',
            xaxis=dict(range=[0, 100], showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=140,
            margin=dict(t=30, b=30),
            legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
            title="장애 유형 구성 비율 (100% 누적)"
        )

        st.plotly_chart(fig, use_container_width=True)




    # ------------------------ 가구 유형 시각화 (열 이름 strip 포함) ------------------------
    import plotly.graph_objects as go

    st.markdown("### 🏠 가구 유형별 구성 비율")

    # 데이터 로딩 및 정리
    df_gu = pd.read_csv("자치구 데이터 통합본 (연령, 성별제외).csv", na_values="NA")
    df_gu.rename(columns={df_gu.columns[0]: "자치구"}, inplace=True)
    df_gu.columns = df_gu.columns.str.strip()  # 열 이름 공백 제거

    # 가구유형 열 (AP~AS, 총 4개)
    df_gu.iloc[:, 41:45] = df_gu.iloc[:, 41:45].apply(pd.to_numeric, errors="coerce")
    house_cols = df_gu.columns[41:45]
    row = df_gu[df_gu["자치구"] == selected_gu][house_cols].iloc[0]

    if row.sum() == 0 or row.dropna().empty:
        st.warning("선택한 자치구의 가구 유형 데이터가 없습니다.")
    else:
        total = row.sum()
        percentages = row / total * 100

        fig_house = go.Figure()
        for col, val in zip(percentages.index, percentages.values):
            fig_house.add_trace(go.Bar(
                x=[val],
                y=["가구 유형"],
                orientation='h',
                name=col,
                text=f"{val:.1f}%",
                textposition='inside'
            ))

        fig_house.update_layout(
            barmode='stack',
            height=140,
            margin=dict(t=30, b=30),
            xaxis=dict(range=[0, 100], showticklabels=False),
            yaxis=dict(showticklabels=False),
            title="가구 구성원 수 기준 비율 (100% 누적)",
            legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center")
        )
        fig_house.add_annotation(
            text="서울시 1인가구 비율 평균: 40.9%",
            xref="paper", yref="paper",
            x=0.5, y=-0.35,  # 중앙 아래 위치
            showarrow=False,
            font=dict(size=12, color="red"),
            align="center"
        )


        st.plotly_chart(fig_house, use_container_width=True)


    # ------------------------ 1인 가구 수 vs 서울 평균 ------------------------
    st.markdown("### 🏘️ 1인 가구 수 비교")

    # 1인 가구 수 데이터 로딩
    household_data = pd.read_csv("자치구 데이터 통합본 (연령, 성별제외).csv")  # 수정된 통합 데이터 사용
    household_data = household_data.rename(columns={household_data.columns[0]: "자치구"})  # 첫 열 이름 통일

    #선택된 자치구의 1인 가구 수
    selected_gu_oneperson = household_data[household_data["자치구"] == selected_gu]["1인가구"].values[0]

    # 서울 전체 평균 1인 가구 수
    avg_oneperson = round(household_data["1인가구"].mean())

    # 시각화
    fig_household = go.Figure()

    # 자치구 막대
    fig_household.add_trace(go.Bar(
        y=[selected_gu],
        x=[selected_gu_oneperson],
        orientation='h',
        name=selected_gu,
        marker_color='steelblue',
        text=[f"{selected_gu_oneperson:,}가구"],
        textposition='outside',
        textfont=dict(color='black') 
    ))

    # 평균선
    fig_household.add_shape(
        type="line",
        x0=avg_oneperson,
        x1=avg_oneperson,
        y0=-0.5,
        y1=0.5,
        line=dict(color="red", width=2)
    )

    # 평균 텍스트 주석
    fig_household.add_annotation(
        x=avg_oneperson,
        y=0,
        text=f"서울 평균: {avg_oneperson:,}가구",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-30,
        font=dict(color="black")
    )

    fig_household.update_layout(
        title=f"📏 1인 가구 수 비교 ({selected_gu} vs 서울 평균)",
        xaxis=dict(range=[0, 180000], showgrid=False),
        yaxis=dict(showticklabels=False),
        margin=dict(l=0, r=0, t=60, b=60),
        height=180,
        showlegend=False,
        plot_bgcolor='#f9f9f9'
    )

    st.plotly_chart(fig_household, use_container_width=True)

with right_col:
    # ------------------------ 문화지표 1/2 강조 숫자 시각화 ------------------------
    import streamlit as st
    import pandas as pd

    st.markdown("### 📚 문화지표 요약 (강좌 비율, 이용 관심도 등)")

    # 데이터 로딩 및 정제
    df_culture = pd.read_csv("자치구 데이터 통합본 (연령, 성별제외).csv", na_values="NA")
    df_culture.rename(columns={df_culture.columns[0]: "자치구"}, inplace=True)
    df_culture.columns = df_culture.columns.str.strip()

    # 선택된 자치구 데이터 추출
    if selected_gu not in df_culture["자치구"].values:
        st.warning("해당 자치구의 문화지표 데이터가 없습니다.")
    else:
        row = df_culture[df_culture["자치구"] == selected_gu].iloc[0]

        # 문화지표1: AU (강좌_비율), AW (운영_관심도_점수)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📘 강좌 비율 (1만명 당 강좌횟수)", f"{row['강좌_비율']:.1f}" if pd.notna(row['강좌_비율']) else "정보 없음")
            st.metric("⭐ 운영 관심도 점수", f"{row['운영_관심도_점수']:.1f}" if pd.notna(row['운영_관심도_점수']) else "정보 없음")

        # 문화지표2: AV (참가자_비율), AX (이용_관심도_점수)
        with col2:
            st.metric("👥 참가자 비율 (1만명 당 참가자 수)", f"{row['참가자_비율']:.1f}" if pd.notna(row['참가자_비율']) else "정보 없음")
            st.metric("🌟 이용 관심도 점수", f"{row['이용_관심도_점수']:.1f}" if pd.notna(row['이용_관심도_점수']) else "정보 없음")


    # ------------------------ 문화·복지 시설 유형별 개수 ------------------------
    import plotly.graph_objects as go

    st.markdown("### 🏛️ 문화·복지 시설 유형별 개수")

    # 파일 불러오기 및 전처리
    df_fac = pd.read_csv("자치구 데이터 통합본 (연령, 성별제외).csv", na_values="NA")
    df_fac.rename(columns={df_fac.columns[0]: "자치구"}, inplace=True)
    df_fac.columns = df_fac.columns.str.strip()

    # 해당 열들 추출 (AY~BB, 총 4열)
    facility_cols = df_fac.columns[50:54]
    df_fac[facility_cols] = df_fac[facility_cols].apply(pd.to_numeric, errors="coerce")
    row = df_fac[df_fac["자치구"] == selected_gu][facility_cols].iloc[0]

    if row.dropna().sum() == 0:
        st.warning("해당 자치구의 문화·복지 시설 데이터가 없습니다.")
    else:
        fig_fac = go.Figure()
        for col in facility_cols:
            color = '#d62728' if "공공도서관" in col else '#1f77b4'
            fig_fac.add_trace(go.Bar(
                x=[col],
                y=[row[col]],
                name=col,
                marker_color=color,
                text=[f"{int(row[col])}개"],
                textposition='outside'
            ))

        fig_fac.update_layout(
            title=f"{selected_gu} 내 문화·복지 시설 수 (공공도서관 강조)",
            yaxis_title="시설 수",
            yaxis=dict(range=[0, 20]),  # ✅ y축 최대값 고정
            height=350,
            margin=dict(t=60, b=60),
            showlegend=False
        )
        st.plotly_chart(fig_fac, use_container_width=True)


    # ------------------------ 초/중/고 학교 수 시각화 ------------------------
    st.markdown("### 🏫 초·중·고 학교 수")

    # 데이터 로딩
    df_school = pd.read_csv("자치구 데이터 통합본 (연령, 성별제외).csv", na_values="NA")
    df_school.rename(columns={df_school.columns[0]: "자치구"}, inplace=True)
    df_school.columns = df_school.columns.str.strip()

    # 정확한 열 이름 반영 (초등학교 / 중학교 / 고등학교)
    if selected_gu not in df_school["자치구"].values:
        st.warning("해당 자치구의 학교 수 데이터가 없습니다.")
    else:
        row = df_school[df_school["자치구"] == selected_gu].iloc[0]

        elementary = int(row['초등학교']) if pd.notna(row['초등학교']) else 0
        middle = int(row['중학교']) if pd.notna(row['중학교']) else 0
        high = int(row['고등학교']) if pd.notna(row['고등학교']) else 0

        # 카드 스타일
        card_style = """
            <div style="
                padding: 35px;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 8px;
                text-align: center;
                font-size: 18px;
                line-height: 1.5;">
                <strong>{title}</strong><br>
                <span style='font-size: 32px; color: #2ca02c;'><strong>{value}</strong>개</span>
            </div>
        """

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(card_style.format(title="초등학교", value=elementary), unsafe_allow_html=True)
        with col2:
            st.markdown(card_style.format(title="중학교", value=middle), unsafe_allow_html=True)
        with col3:
            st.markdown(card_style.format(title="고등학교", value=high), unsafe_allow_html=True)

    #---------------행사수--------------------------
    # 🎭 선택한 자치구의 이번 달 행사 수 카드
   

    st.markdown(f"### 🎭 {selected_gu} – {datetime.now().strftime('%Y년 %m월')} 문화행사")

    # API 요청
    api_key = '5075546443646f6833344f5553734b'
    current_month = datetime.now().strftime('%Y-%m')
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/culturalEventInfo/1/1000//%20/{current_month}"

    r = requests.get(url)
    r.encoding = 'utf-8'
    data = r.json()

    # 전체 행사 목록
    events = data.get('culturalEventInfo', {}).get('row', [])

    # 선택된 자치구의 이번 달 행사만 필터링
    filtered_events = []
    for e in events:
        date_str = e.get("DATE")
        if date_str and date_str.startswith(current_month) and e.get("GUNAME") == selected_gu:
            filtered_events.append(e)

    # 카드 출력
    count = len(filtered_events)
    card_html = f"""
    <div style="
        padding: 15px;
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 8px;
        text-align: center;
        font-size: 18px;
        line-height: 1.5;
        height: 120px;">
        <strong>{selected_gu} – {datetime.now().strftime('%Y년 %m월')} 문화행사 수</strong><br>
        <span style='font-size: 32px; color: #2ca02c;'><strong>{count}</strong>건</span>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)



# ------------------------ 만족도 시각화 ------------------------
melted = df.melt(id_vars='자치구', var_name='문항', value_name='원점수')
melted['환산점수'] = melted['원점수'].apply(convert_score)
melted['항목'] = melted['문항'].apply(classify_section)
melted = melted.dropna(subset=['환산점수', '항목'])

grouped = melted.groupby(['자치구', '항목'])['환산점수'].mean().reset_index()
seoul_avg = grouped.groupby('항목')['환산점수'].mean().reset_index(name='서울시 평균')
detail_grouped = melted.groupby(['자치구', '항목', '문항'])['환산점수'].mean().reset_index()

sections = ['공간이용', '정보활용', '소통정책', '문화교육', '사회관계', '장기효과']
selected_section = st.selectbox("항목 선택", sections)

# 레이더 차트
st.markdown("---")
col1, col2 = st.columns([1.2, 1.3])
with col1:
    data_gu = grouped[grouped['자치구'] == selected_gu_code]
    data_gu = data_gu.merge(seoul_avg, on='항목')
    fig1 = go.Figure()
    fig1.add_trace(go.Scatterpolar(
        r=data_gu['환산점수'],
        theta=data_gu['항목'],
        fill='toself',
        name=selected_gu
    ))
    fig1.add_trace(go.Scatterpolar(
        r=data_gu['서울시 평균'],
        theta=data_gu['항목'],
        name='서울 평균',
        line=dict(color='black', dash='dash')  # ✅ 검정색 dashed 선으로 수정
    ))
    fig1.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[60, 100], tickfont=dict(color="black"))
        ),
        showlegend=True,
        title=f"{selected_gu} vs 서울 평균 (항목별 평균 점수)",
        margin=dict(l=80, r=80, t=80, b=80),
        height=550
    )
    st.plotly_chart(fig1, use_container_width=True)


# ------------------------ 막대 차트 (세부 문항) ------------------------
with col2:
    detail_data = detail_grouped[
        (detail_grouped['자치구'] == selected_gu_code) &
        (detail_grouped['항목'] == selected_section)
    ]

    seoul_section = detail_grouped[
        detail_grouped['항목'] == selected_section
    ].groupby('문항', sort=False)['환산점수'].mean().reset_index(name='서울시 평균')

    merged = detail_data.merge(seoul_section, on='문항', how='left')

    # 자치구 항목 평균선
    section_avg = grouped[
        (grouped['자치구'] == selected_gu_code) &
        (grouped['항목'] == selected_section)
    ]['환산점수'].values[0]

    # ✅ 정렬 키 기반 문항 순서 고정 (Q1_1 → 101, Q1_10 → 110 등으로 변환 후 정렬)
    import re
    from pandas.api.types import CategoricalDtype

    def extract_question_order(q):
        match = re.match(r"Q(\d)_(\d+)", str(q))
        return int(match.group(1)) * 100 + int(match.group(2)) if match else float('inf')

    merged = merged.sort_values(by="문항", key=lambda col: col.map(extract_question_order))
    question_order_sorted = merged["문항"].tolist()
    cat_type = CategoricalDtype(categories=question_order_sorted, ordered=True)
    merged["문항"] = merged["문항"].astype(cat_type)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=merged['문항'], y=merged['환산점수'], name=selected_gu, marker_color='blue'))
    fig2.add_trace(go.Bar(x=merged['문항'], y=merged['서울시 평균'], name="서울 평균", marker_color='orange'))
    fig2.add_trace(go.Scatter(x=merged['문항'], y=[section_avg]*len(merged), name="자치구 평균선", mode='lines',
                              line=dict(color='red', dash='solid')))
    fig2.update_layout(
        barmode='group',
        title=f"[{selected_section}] 세부 문항별 점수 + 자치구 평균선",
        yaxis=dict(range=[0, 100]),
        height=550
    )
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------ 선택 항목 문항 테이블 ------------------------
st.markdown("### 📄 해당 항목 문항 목록")

meta_table = pd.read_csv("이용자조사_변수가이드.csv")
meta_table = meta_table[['문항번호', '내용']].copy()
meta_table = meta_table.dropna()
meta_table = meta_table[meta_table['문항번호'].astype(str).str.startswith("Q")]

selected_questions = merged['문항'].tolist()
filtered_table = meta_table[meta_table['문항번호'].isin(selected_questions)]

# ✅ 문항 테이블도 동일한 순서로 정렬 고정
filtered_table['문항번호'] = filtered_table['문항번호'].astype(
    CategoricalDtype(categories=question_order_sorted, ordered=True)
)
filtered_table = filtered_table.sort_values('문항번호')

st.dataframe(filtered_table, use_container_width=True, height=300)


# ------------------------ A/B 서비스 효과 vs 영향력 ------------------------
st.markdown("---")
st.markdown("### 🎯 차원별 서비스 효과(A) vs 영향력(B) 비교")

# ⬇️ 변수 가이드 로드
meta_raw = pd.read_csv("이용자조사_변수가이드.csv")
meta_ab = meta_raw[['문항번호', '차원', '분류']].copy()
meta_ab = meta_ab[meta_ab['문항번호'].notna()]
meta_ab = meta_ab[meta_ab['문항번호'].astype(str).str.startswith("Q")]
meta_ab = meta_ab[meta_ab['분류'].isin(['A', 'B'])]
meta_ab = meta_ab[~meta_ab['차원'].str.contains("Q6")]  # ✅ Q6 제외

# ⬇️ 차원명 한글 매핑
dimension_map = {
    "Q1": "공간 및 이용 편의성",
    "Q2": "정보 획득 및 활용",
    "Q3": "소통 및 정책반영",
    "Q4": "문화ㆍ교육향유",
    "Q5": "사회적 관계형성"
}
meta_ab["차원명"] = meta_ab["차원"].map(dimension_map)

# ⬇️ 긴 형 데이터 변환
df_long = df[["자치구"] + question_cols].melt(id_vars="자치구", var_name="문항번호", value_name="원점수")
df_long = df_long[df_long["원점수"] != 9]
df_long["환산점수"] = df_long["원점수"].apply(convert_score)

# ⬇️ 메타 병합 및 필터링
df_long = df_long.merge(meta_ab, on="문항번호", how="left")
df_long = df_long.dropna(subset=["차원명", "분류"])
df_ab = df_long[df_long["분류"].isin(["A", "B"])]
df_ab_gu = df_ab[df_ab["자치구"] == selected_gu_code]

# ⬇️ 자치구 기준 A/B 평균 계산
grouped_gu = df_ab_gu.groupby(["차원명", "분류"])["환산점수"].mean().reset_index()

# ✅ 차원 순서 고정
from pandas.api.types import CategoricalDtype
ordered_dims = [
    "공간 및 이용 편의성",
    "정보 획득 및 활용",
    "소통 및 정책반영",
    "문화ㆍ교육향유",
    "사회적 관계형성"
]
cat_type = CategoricalDtype(categories=ordered_dims, ordered=True)
grouped_gu["차원명"] = grouped_gu["차원명"].astype(cat_type)

# ⬇️ 묶음 막대그래프
pivot = grouped_gu.pivot(index="차원명", columns="분류", values="환산점수").reset_index()

fig_ab = go.Figure()
fig_ab.add_trace(go.Bar(
    x=pivot["차원명"],
    y=pivot["A"],
    name="A (서비스 효과)",
    offsetgroup="A",
    marker_color="#1f77b4",
    text=pivot["A"].round(1),
    textposition="outside"
))
fig_ab.add_trace(go.Bar(
    x=pivot["차원명"],
    y=pivot["B"],
    name="B (서비스 영향력)",
    offsetgroup="B",
    marker_color="#d62728",
    text=pivot["B"].round(1),
    textposition="outside"
))

fig_ab.update_layout(
    barmode='group',
    title=f"📌 {selected_gu} – 서비스 효과(A) / 영향력(B) 비교",
    yaxis_title="평균 환산점수",
    yaxis=dict(range=[0, 100]),
    xaxis_title="차원 (Q1~Q5)",
    height=550
)

st.plotly_chart(fig_ab, use_container_width=True)

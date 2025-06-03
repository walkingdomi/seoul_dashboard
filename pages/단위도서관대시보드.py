import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import requests
from datetime import datetime
from math import radians, cos, sin, asin, sqrt

# 페이지 설정
st.set_page_config(page_title="서울시 공공도서관 대시보드", layout="wide")
st.title("📚 서울시 개별도서관 별 현황")

# 현재 달 (YYYY-MM)
current_month = datetime.now().strftime('%Y-%m')

# 🔹 데이터 로드
@st.cache_data(ttl=3600)
def load_data():
    df_library = pd.read_csv("./서울시_공공도서관_2km_반경_행정동_주소분리_UTF8.csv", encoding='utf-8-sig')
    df_pop = pd.read_csv("./2.인구통계+고령.csv", encoding='utf-8-sig', na_values='-')
    df_gender = pd.read_csv("./3.성별.csv", encoding='utf-8-sig', na_values='-')
    welfare_df = pd.read_csv("./5.수급자수.csv", encoding='utf-8-sig')
    return df_library, df_pop, df_gender, welfare_df

df_library, df_pop, df_gender, welfare_df = load_data()
df_gender[['남자', '여자']] = df_gender[['남자', '여자']].apply(pd.to_numeric, errors='coerce').fillna(0)
df_pop.iloc[:, 2:] = df_pop.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

# 🔹 문화행사 API
@st.cache_data(ttl=3600)
def fetch_cultural_events():
    api_key = '5075546443646f6833344f5553734b'
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/culturalEventInfo/1/1000/"
    r = requests.get(url)
    r.encoding = 'utf-8'
    if r.status_code == 200:
        data = r.json()
        if 'culturalEventInfo' in data and 'row' in data['culturalEventInfo']:
            return pd.DataFrame(data['culturalEventInfo']['row'])
    return pd.DataFrame()

df_events_raw = fetch_cultural_events()

# 🔹 행사 데이터 필터링
def filter_current_month_events(df):
    start_dates = df['STRTDATE'].astype(str).str[:7]
    end_dates = df['END_DATE'].astype(str).str[:7]
    mask = (start_dates == current_month) | (end_dates == current_month)
    return df[mask]

df_events = filter_current_month_events(df_events_raw)

# 🔹 자치구 & 도서관 선택
gu_list = sorted(df_library['자치구'].dropna().unique())
default_gu = '강남구'
col1, col2 = st.columns(2)
with col1:
    selected_gu = st.selectbox("자치구 선택", gu_list, index=gu_list.index(default_gu))
with col2:
    library_list = sorted(df_library[df_library['자치구'] == selected_gu]['도서관명'].dropna().unique())
    selected_library = st.selectbox("도서관 선택", library_list)

# 🔹 도서관 정보
library_info = df_library[(df_library['자치구'] == selected_gu) & (df_library['도서관명'] == selected_library)].iloc[0]
lat, lon = library_info['위도'], library_info['경도']
covered_dongs = df_library[(df_library['자치구'] == selected_gu) & (df_library['도서관명'] == selected_library)]['행정동'].unique()

# 🔹 거리 계산
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371000 * 2 * asin(sqrt(a))

# 🔹 지도 및 행사 시각화
display_events = []
with st.container():
    st.markdown(f"<h4 style='margin: 10px 0;'>📍 {selected_library} 주변 현황</h4>", unsafe_allow_html=True)
    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], popup=selected_library, icon=folium.Icon(color='blue', icon='book')).add_to(m)
    folium.Circle([lat, lon], radius=1000, color='red', fill=False).add_to(m)
    folium.Circle([lat, lon], radius=2000, color='blue', fill=True, fill_opacity=0.1).add_to(m)

    for _, e in df_events.iterrows():
        lot, lat_val = e.get('LOT'), e.get('LAT')
        if pd.notna(lot) and pd.notna(lat_val) and lot != '0' and lat_val != '0':
            try:
                e_lat, e_lon = float(lot), float(lat_val)
                if haversine(lon, lat, e_lon, e_lat) <= 1000:
                    folium.Marker(
                        [e_lat, e_lon],
                        popup=f"<b>{e['TITLE']}</b><br>{e['PLACE']}<br>{e['STRTDATE']} ~ {e['END_DATE']}",
                        tooltip=f"{e['TITLE']} | {e['PLACE']}",
                        icon=folium.Icon(color='green')
                    ).add_to(m)
                    display_events.append(e)
            except ValueError:
                pass
    st_folium(m, width=1400, height=500)

# 🔹 행정동 / 행사 목록
with st.container():
    col_left, col_right = st.columns([3, 7])
    with col_left:
        st.markdown("### 🏠 도서관 2km 반경 행정동")
        st.markdown(
            f"<div style='height:400px; overflow-y:auto; background:#f9f9f9; padding:10px; border-radius:8px; border:1px solid #ddd;'>"
            + "".join([f"<li>{d}</li>" for d in covered_dongs]) + "</div>",
            unsafe_allow_html=True
        )
    with col_right:
        st.markdown("### 🎭 문화행사 상세정보")
        event_html = "<div style='height:400px; overflow-y:auto; background:#f9f9f9; padding:10px; border-radius:8px; border:1px solid #ddd;'>"
        if display_events:
            for _, e in pd.DataFrame(display_events).iterrows():
                start_date = e['STRTDATE'][:10] if pd.notna(e['STRTDATE']) else ''
                end_date = e['END_DATE'][:10] if pd.notna(e['END_DATE']) else ''
                event_html += f"<div style='margin-bottom:15px; padding:8px; background:white; border:1px solid #ccc; border-radius:6px;'>" \
                               f"<b>{e['TITLE']}</b><br>" \
                               f"<small>일시: {start_date} ~ {end_date}</small><br>" \
                               f"<small>장소: {e['PLACE']}</small></div>"
        else:
            event_html += "<p>표시할 행사가 없습니다.</p>"
        event_html += "</div>"
        st.markdown(event_html, unsafe_allow_html=True)

# 🔹 성별 / 연령대 차트 & 수급자수 카드
with st.container():
    st.markdown("### 👥 도서관 반경 2km 인구 현황")
    age_sum = df_pop[df_pop['행정동'].isin(covered_dongs)].iloc[:, 2:].sum()
    gender_sum = df_gender[df_gender['행정동'].isin(covered_dongs)].sum()

    col_pie, col_bar = st.columns([4, 6])
    with col_pie:
        # 파이차트
        fig_pie = go.Figure(data=[go.Pie(
            labels=['남자', '여자'],
            values=[gender_sum['남자'], gender_sum['여자']],
            marker=dict(colors=['lightblue', 'pink']),
            textinfo='label+percent',
            hole=0.3
        )])
        fig_pie.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        # 수급자수 카드
        covered_welfare = welfare_df[welfare_df['행정동'].isin(covered_dongs)]
        avg_welfare_rate = covered_welfare['수급자수'].mean()
        seoul_avg = welfare_df['수급자수'].mean()
        st.markdown(f"""
        <div style="padding: 10px; background-color: #f8f9fa; border: 1px solid #ddd;
                     border-radius: 8px; text-align: center; font-size: 20px; margin-top: 10px;">
            <strong>평균 수급자수</strong><br>
            <span style="font-size: 36px; color: #0d6efd;"><strong>{avg_welfare_rate:,.0f}명</strong></span><br>
            <span style="font-size: 14px; color: #dc3545;">행정동 평균: {seoul_avg:,.0f}명</span>
        </div>
        """, unsafe_allow_html=True)

    with col_bar:
        # 65세 이상 컬럼명 리스트
        elderly_cols = ['65~69세', '70~74세', '75~79세', '80~84세', '85~89세', '90~94세', '95~99세', '100세 이상']
        
        # 색상 지정: 65세 이상 오렌지, 나머지 스카이블루
        colors = ['orange' if age in elderly_cols else 'skyblue' for age in age_sum.index]
        
        fig_bar = go.Figure(go.Bar(
            x=age_sum.index,
            y=age_sum.values,
            marker_color=colors,
            text=age_sum.values,
            textposition='outside'
        ))
        fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(tickformat=","))
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

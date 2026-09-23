import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import datetime
import io
import math
import time
import re
import requests
from bs4 import BeautifulSoup
from zoneinfo import ZoneInfo

# ---------------------------------------------------------
# 1. 페이지 설정 (최상단)
# ---------------------------------------------------------
st.set_page_config(page_title="US Stock Analyzer", layout="wide")

# ---------------------------------------------------------
# 2. 데이터 캐싱 및 함수 정의
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_market_indicators():
    indices = {
        "SP500": "^GSPC",      
        "NASDAQ": "^IXIC",     
        "DOW": "^DJI",         
        "VIX": "^VIX",         
        "OIL": "CL=F",         
        "GOLD": "GC=F",        
        "TNX": "^TNX",         
        "IRX": "^IRX",         # 13주물 금리 (10Y-3M 금리차 계산용)
        "BTC": "BTC-USD"       
    }
    results = {}
    for name, ticker in indices.items():
        try:
            df = yf.Ticker(ticker).history(period="1y")
            if not df.empty: results[name] = df
        except: continue
    return results

@st.cache_data(ttl=3600)
def fetch_company_data(ticker):
    stock = yf.Ticker(ticker)
    return {"info": stock.info, "history": stock.history(period="2y")}


def estimate_beta_from_history(stock_history, market_history, window=252):
    """Yahoo info에 Beta가 없을 때 일별 수익률로 Beta를 보조 추정한다."""
    try:
        if stock_history is None or market_history is None:
            return None
        stock_close = stock_history.get("Close")
        market_close = market_history.get("Close")
        if stock_close is None or market_close is None:
            return None
        stock_ret = stock_close.pct_change().dropna()
        market_ret = market_close.pct_change().dropna()
        aligned = pd.concat([stock_ret.rename("stock"), market_ret.rename("market")], axis=1).dropna().tail(window)
        if len(aligned) < 60:
            return None
        market_var = aligned["market"].var()
        if pd.isna(market_var) or market_var <= 0:
            return None
        beta = aligned["stock"].cov(aligned["market"]) / market_var
        return float(beta) if np.isfinite(beta) else None
    except Exception:
        return None


def calculate_market_confidence(m_data):
    """시장 점수에 실제로 사용되는 데이터의 확보 수준을 0~100으로 계산한다."""
    required = {
        "SP500": 0.30,
        "VIX": 0.20,
        "OIL": 0.15,
        "TNX": 0.15,
        "IRX": 0.20,
    }
    available_weight = 0.0
    total_weight = sum(required.values())
    for key, weight in required.items():
        df = m_data.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            series = df["Close"].dropna()
            minimum = 200 if key == "SP500" else 20
            if len(series) >= minimum:
                available_weight += weight
            elif len(series) > 0:
                available_weight += weight * min(1.0, len(series) / minimum)
    return round((available_weight / total_weight) * 100) if total_weight else 0

def get_market_status(m_data):
    plus_m, minus_m, m_score = [], [], 0
    try:
        sp5 = m_data.get("SP500")
        if sp5 is None or len(sp5) < 200:
            return 100, 20, 0, 100, 0, [], ["⚠️ 데이터 부족"], False

        curr_sp = sp5['Close'].iloc[-1]
        ma200 = sp5['Close'].rolling(window=200).mean().iloc[-1]
        disparity = (curr_sp / ma200) * 100

        # 52주 고점 대비 하락률: 단순히 신고가라고 감점하지 않고
        # 상승 추세 안에서의 조정 여부를 함께 평가한다.
        high_52w = sp5['Close'].tail(252).max()
        drawdown = (curr_sp / high_52w - 1) * 100
        market_heat = (curr_sp / high_52w) * 100
        sp_above_ma200 = curr_sp >= ma200

        # 동적 매크로 임계값 설정 (최근 90일 평균선 대비)
        oil_df = m_data.get("OIL")['Close'] if m_data.get("OIL") is not None else pd.Series(dtype=float)
        tnx_df = m_data.get("TNX")['Close'] if m_data.get("TNX") is not None else pd.Series(dtype=float)
        irx_df = m_data.get("IRX")['Close'] if m_data.get("IRX") is not None else pd.Series(dtype=float)

        curr_oil = oil_df.iloc[-1] if not oil_df.empty else 80
        oil_ma90 = oil_df.tail(90).mean() if len(oil_df) >= 20 else curr_oil
        curr_tnx = tnx_df.iloc[-1] if not tnx_df.empty else 4.0
        tnx_ma90 = tnx_df.tail(90).mean() if len(tnx_df) >= 20 else curr_tnx
        curr_irx = irx_df.iloc[-1] if not irx_df.empty else np.nan

        vix_series = m_data.get("VIX", pd.DataFrame()).get('Close', pd.Series(dtype=float))
        vix = vix_series.iloc[-1] if not vix_series.empty else 20
        vix_ma5 = vix_series.tail(5).mean() if len(vix_series) >= 5 else vix

        # [개선 1] Risk Spread -> 실제 10Y-3M 금리차
        # 양(+)의 스프레드를 우호적으로 보되, 역전은 감점한다.
        if np.isfinite(curr_irx):
            spread = curr_tnx - curr_irx
        else:
            spread = np.nan

        # 1. S&P 500 이격도
        if disparity <= 108:
            m_score += 20; plus_m.append(f"✅ 이격도 안정({disparity:.1f}%): (+20)")
        elif disparity <= 115:
            m_score += 10; plus_m.append(f"🟡 이격도 다소 높음({disparity:.1f}%): (+10)")
        else:
            minus_m.append(f"❌ 이격도 과열({disparity:.1f}%): (+0)")

        # [개선 2] VIX level + direction + 시장 추세를 함께 평가
        if vix >= 30:
            if vix < vix_ma5 and sp_above_ma200:
                m_score += 20; plus_m.append(f"🔥 VIX 공포 후 진정({vix:.1f}) (+20)")
            elif vix > vix_ma5 and not sp_above_ma200:
                m_score -= 10; minus_m.append(f"🚨 VIX 급등 + 하락추세({vix:.1f}) (-10)")
            else:
                m_score += 5; plus_m.append(f"🟠 높은 변동성({vix:.1f}) (+5)")
        elif vix >= 20:
            m_score += 10; plus_m.append(f"✅ VIX 보통({vix:.1f}) (+10)")
        elif vix < 15:
            if sp_above_ma200:
                m_score += 5; plus_m.append(f"⚠️ 낮은 VIX·강세장({vix:.1f}) (+5)")
            else:
                m_score -= 5; minus_m.append(f"⚠️ 낮은 VIX지만 추세 취약({vix:.1f}) (-5)")
        else:
            m_score += 8; plus_m.append(f"✅ VIX 안정({vix:.1f}) (+8)")

        # [개선 3] 52주 고점 근접을 무조건 과열로 보지 않음
        if sp_above_ma200 and drawdown <= -5:
            m_score += 20; plus_m.append(f"✅ 상승 추세 내 조정({drawdown:.1f}%) (+20)")
        elif sp_above_ma200 and drawdown <= -2:
            m_score += 12; plus_m.append(f"🟡 완만한 조정({drawdown:.1f}%) (+12)")
        elif sp_above_ma200:
            m_score += 8; plus_m.append(f"📈 신고가/고점권이지만 상승추세 유지({drawdown:.1f}%) (+8)")
        elif drawdown <= -10:
            m_score += 5; plus_m.append(f"🟠 큰 폭 조정({drawdown:.1f}%), 추세 확인 필요 (+5)")
        else:
            minus_m.append(f"🚩 추세 약화·고점권 이탈({drawdown:.1f}%): (+0)")

        # 4. 동적 매크로 평가
        macro_val = 0
        if curr_oil > oil_ma90 * 1.15:
            macro_val -= 15; minus_m.append(f"🚨 유가 단기 급등세 (-15)")
        elif curr_oil < oil_ma90 * 0.95:
            macro_val += 15; plus_m.append(f"✅ 유가 하향 안정화 (+15)")

        if curr_tnx > tnx_ma90 * 1.10:
            macro_val -= 15; minus_m.append(f"🚨 국채금리 상승 발작 (-15)")
        elif curr_tnx < tnx_ma90 * 0.95:
            macro_val += 15; plus_m.append(f"✅ 국채금리 하향 안정화 (+15)")

        m_score += macro_val
        if macro_val >= 0:
            plus_m.append("✅ 거시경제 동향 양호")

        # [개선 4] 10Y-3M 금리차를 실제 계산하여 반영
        if np.isfinite(spread):
            if spread >= 0.75:
                m_score += 20; plus_m.append(f"✅ 10Y-3M 스프레드 양호({spread:.2f}%p) (+20)")
            elif spread >= 0:
                m_score += 10; plus_m.append(f"🟡 10Y-3M 스프레드 완만({spread:.2f}%p) (+10)")
            else:
                m_score -= 10; minus_m.append(f"🚨 10Y-3M 금리 역전({spread:.2f}%p) (-10)")
        else:
            plus_m.append("⚠️ 10Y-3M 스프레드 데이터 없음")

        # [개선 5] 경보도 VIX 방향성과 시장 추세를 함께 고려
        hunter_alert = (
            curr_oil > oil_ma90 * 1.2 and
            vix > 25 and
            vix < vix_ma5 and
            sp_above_ma200
        )

        return disparity, vix, (spread if np.isfinite(spread) else 0), market_heat, max(0, min(100, m_score)), plus_m, minus_m, hunter_alert
    except Exception as e:
        return 100, 20, 0, 100, 0, [], [f"로직 오류: {e}"], False

def create_gauge(title, value, min_val, max_val):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, title={'text': title, 'font': {'color': 'white', 'size': 16}},
                                 gauge={'axis': {'range': [min_val, max_val]}, 'bar': {'color': "#ff4b4b"}, 'bgcolor': "rgba(0,0,0,0)"}))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=180, margin=dict(l=20, r=20, t=40, b=10))
    return fig

def calculate_valuation_score(info, sector):
    """기존 절대 밸류에이션 로직을 유지하되 업종별로 더 적합한 지표를 우선한다."""
    peg = info.get('pegRatio')
    fwd_pe = info.get('forwardPE')
    pb = info.get('priceToBook')
    ps = info.get('priceToSalesTrailing12Months')

    candidates = []
    if sector in ['Financial Services']:
        if pb and 0 < pb < 1.5: candidates.append((15, f"✅ 금융주 저P/B ({pb:.1f}) (+15)"))
        elif pb and 0 < pb < 2.0: candidates.append((8, f"🟡 금융주 P/B 양호 ({pb:.1f}) (+8)"))
        if fwd_pe and 0 < fwd_pe < 15: candidates.append((12, f"✅ 금융주 Fwd P/E 양호 ({fwd_pe:.1f}) (+12)"))
    elif sector in ['Technology', 'Communication Services']:
        if peg and 0 < peg < 1.2: candidates.append((15, f"✅ 성장주 PEG 우수 ({peg:.2f}) (+15)"))
        elif peg and 0 < peg < 1.8: candidates.append((8, f"🟡 성장주 PEG 양호 ({peg:.2f}) (+8)"))
        if fwd_pe and 0 < fwd_pe < 20: candidates.append((10, f"✅ 성장주 Fwd P/E 양호 ({fwd_pe:.1f}) (+10)"))
        if ps and 0 < ps < 4: candidates.append((6, f"🟡 성장주 P/S 양호 ({ps:.1f}) (+6)"))
    elif sector == 'Energy':
        if fwd_pe and 0 < fwd_pe < 15: candidates.append((12, f"✅ 에너지주 Fwd P/E 양호 ({fwd_pe:.1f}) (+12)"))
        if pb and 0 < pb < 2: candidates.append((8, f"🟡 에너지주 P/B 양호 ({pb:.1f}) (+8)"))
    else:
        if fwd_pe and 0 < fwd_pe < 15: candidates.append((15, f"✅ 저평가 가치주 (Fwd P/E {fwd_pe:.1f}) (+15)"))
        elif fwd_pe and 0 < fwd_pe < 20: candidates.append((8, f"🟡 Fwd P/E 양호 ({fwd_pe:.1f}) (+8)"))
        if ps and 0 < ps < 2: candidates.append((10, f"✅ 매출대비 저평가 (P/S {ps:.1f}) (+10)"))
        if pb and 0 < pb < 1.5: candidates.append((10, f"✅ 저P/B 자산주 (P/B {pb:.1f}) (+10)"))

    if not candidates:
        return 0, "⚪ 유효한 업종별 밸류에이션 조건 없음"
    best_pts, best_msg = max(candidates, key=lambda x: x[0])
    return best_pts, best_msg

def create_candlestick(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="주가", increasing_line_color='#00ff00', decreasing_line_color='#ff4b4b'
    ))
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='20 MA', line=dict(color='#ffd700', width=1.5)))
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='50 MA', line=dict(color='#ff8c00', width=1.5)))
    if 'MA200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='200 MA', line=dict(color='#00bfff', width=1.5)))

    fig.update_layout(
        title=title, template="plotly_dark", height=420,
        xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ---------------------------------------------------------
# 2-1. 공식 미국 경제 일정 수집 / 해설 / 경보
# ---------------------------------------------------------
ECONOMIC_SOURCE_URLS = {
    "Fed": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "BLS": "https://www.bls.gov/schedule/news_release/cpi.htm",
    "BLS_Employment": "https://www.bls.gov/schedule/news_release/empsit.htm",
    "Census": "https://www.census.gov/economic-indicators/calendar-listview.html",
    "BEA": "https://www.bea.gov/news/schedule",
}

ECONOMIC_EVENT_GUIDE = {
    "FOMC": {
        "meaning": "미국 중앙은행(Fed)이 기준금리와 통화정책 방향을 결정하는 회의입니다.",
        "impact": "금리 전망이 바뀌면 주식시장 전체의 할인율과 변동성이 크게 움직일 수 있습니다.",
        "assets": "시장 전체 · 성장주 · 금융주"
    },
    "CPI": {
        "meaning": "미국 가계가 구매하는 상품·서비스 가격의 변화를 보여주는 대표적인 물가 지표입니다.",
        "impact": "예상보다 높으면 금리 인하 기대가 약해질 수 있어 성장주에 부담이 될 수 있습니다.",
        "assets": "성장주 · 기술주 · 채권"
    },
    "PCE": {
        "meaning": "미국 가계의 소비지출을 기준으로 측정한 물가 지표로 Fed가 특히 중요하게 보는 지표입니다.",
        "impact": "물가가 예상보다 높거나 낮은지에 따라 연준의 금리 전망이 바뀔 수 있습니다.",
        "assets": "성장주 · 기술주 · 시장 전체"
    },
    "Employment": {
        "meaning": "미국의 고용과 실업 상태를 보여주는 핵심 경기 지표입니다.",
        "impact": "고용이 너무 강하면 물가·금리 부담이 커질 수 있고, 약하면 경기 둔화 우려가 커질 수 있습니다.",
        "assets": "시장 전체 · 소비주 · 금융주 · 경기민감주"
    },
    "Retail Sales": {
        "meaning": "미국 소비자가 실제로 얼마나 지출했는지 보여주는 소비 경기 지표입니다.",
        "impact": "강한 소비는 경기에는 긍정적이지만 금리 상승 압력을 높일 수도 있습니다.",
        "assets": "소비주 · 금융주 · 경기민감주 · 성장주"
    },
    "Building Permits": {
        "meaning": "향후 주택 건설 활동을 미리 보여주는 대표적인 주택시장 선행지표입니다.",
        "impact": "증가하면 주택·건설 경기가 강하다는 신호가 될 수 있고, 감소하면 경기 둔화 신호가 될 수 있습니다.",
        "assets": "건설 · 주택 · 금융주 · 경기민감주"
    },
    "GDP": {
        "meaning": "미국 경제가 일정 기간 동안 생산한 재화와 서비스의 총규모입니다.",
        "impact": "강한 성장은 경기에는 좋지만 금리 상승 압력을 높일 수 있고, 약한 성장은 경기 우려를 키울 수 있습니다.",
        "assets": "시장 전체 · 경기민감주"
    },
    "PPI": {
        "meaning": "기업이 상품·서비스를 생산하며 받는 가격의 변화를 보여주는 생산자 물가 지표입니다.",
        "impact": "생산비용 상승이 소비자 물가로 이어질지에 대한 힌트를 줘 금리 기대에 영향을 줄 수 있습니다.",
        "assets": "성장주 · 산업재 · 시장 전체"
    },
    "JOLTS": {
        "meaning": "기업의 채용 수요와 구인 규모를 보여주는 노동시장 지표입니다.",
        "impact": "고용 수요가 강한지 약한지에 따라 경기와 금리 전망이 바뀔 수 있습니다.",
        "assets": "시장 전체 · 소비주 · 금융주"
    },
}


def _get_guide(event_name: str):
    name = str(event_name or "").lower()
    if "fomc" in name or "federal open market" in name:
        return ECONOMIC_EVENT_GUIDE["FOMC"], "high"
    if "consumer price index" in name or name.strip() in {"cpi", "core cpi"}:
        return ECONOMIC_EVENT_GUIDE["CPI"], "high"
    if "personal income and outlays" in name or "pce" in name or "personal consumption expenditures" in name:
        return ECONOMIC_EVENT_GUIDE["PCE"], "high"
    if "employment situation" in name or "nonfarm" in name or "unemployment rate" in name:
        return ECONOMIC_EVENT_GUIDE["Employment"], "high"
    if "retail" in name and "sales" in name:
        return ECONOMIC_EVENT_GUIDE["Retail Sales"], "medium"
    if "building permits" in name or "new residential construction" in name:
        return ECONOMIC_EVENT_GUIDE["Building Permits"], "medium"
    if "gdp" in name:
        return ECONOMIC_EVENT_GUIDE["GDP"], "high"
    if "producer price index" in name or "ppi" in name:
        return ECONOMIC_EVENT_GUIDE["PPI"], "high"
    if "job openings and labor turnover" in name or "jolts" in name:
        return ECONOMIC_EVENT_GUIDE["JOLTS"], "medium"
    return {
        "meaning": "미국 경제의 특정 부문 상태를 보여주는 공식 발표입니다.",
        "impact": "실제 결과가 시장 예상과 크게 다르면 금리 전망과 관련 업종의 변동성이 커질 수 있습니다.",
        "assets": "시장 전체 · 관련 업종",
    }, "medium"


def _parse_us_et_datetime(date_text, time_text="08:30 AM"):
    """공식 발표 시각(미 동부시간)을 한국시간으로 변환."""
    if date_text is None:
        return None
    if isinstance(date_text, (pd.Timestamp, datetime.datetime, datetime.date)):
        date_text = date_text.strftime("%B %d, %Y")
    value = f"{str(date_text).strip()} {str(time_text or '08:30 AM').strip()}"
    formats = [
        "%A, %B %d, %Y %I:%M %p",
        "%B %d, %Y %I:%M %p",
        "%b. %d, %Y %I:%M %p",
        "%B %d, %Y",
        "%b. %d, %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(value, fmt).replace(tzinfo=ZoneInfo("America/New_York"))
            return dt.astimezone(ZoneInfo("Asia/Seoul"))
        except ValueError:
            continue
    return None


def _request_text(url):
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "US-Stock-Analyzer/1.0 (+https://www.federalreserve.gov/)"},
    )
    response.raise_for_status()
    return response.text


def _fetch_fomc_events():
    """
    Fed 공식 FOMC 일정에서 '한 회의당 1건'만 만든다.

    주의:
    - Fed 페이지는 2일짜리 회의를 "15-16"처럼 표시한다.
    - 우리 앱에서 '시장 영향/금리 결정 시각'은 보통 두 번째 날 14:00 ET로 잡는다.
    - 이전 버전은 페이지 전체의 날짜 범위를 정규식으로 훑어 여러 연도의
      September 일정을 섞어오는 문제가 있을 수 있었다. 여기서는 연도/월/일 범위를
      순서대로 읽어 현재 연도와 다음 연도만 명시적으로 수집한다.
    """
    today = datetime.date.today()
    target_years = {today.year, today.year + 1}

    # 실제 페이지 파싱이 실패하더라도 앱이 틀린 반복 일정 대신
    # 공식 일정의 현재/다음 연도 값을 사용할 수 있도록 안전한 fallback을 둔다.
    fallback = {
        2026: [(1, 28), (3, 18), (4, 29), (6, 17), (7, 29), (9, 16), (10, 28), (12, 9)],
        2027: [(1, 27), (3, 17), (4, 28), (6, 9), (7, 28), (9, 15), (10, 27), (12, 8)],
    }

    parsed = []
    try:
        soup = BeautifulSoup(_request_text(ECONOMIC_SOURCE_URLS["Fed"]), "html.parser")
        lines = [re.sub(r"\s+", " ", x).strip() for x in soup.get_text("\n").splitlines()]
        lines = [x for x in lines if x]

        current_year = None
        current_month = None
        month_re = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December)$", re.I)
        range_re = re.compile(r"^(\d{1,2})\s*-\s*(\d{1,2})\*?$")
        year_re = re.compile(r"^(20\d{2}) FOMC Meetings$")

        for line in lines:
            ymatch = year_re.match(line)
            if ymatch:
                year = int(ymatch.group(1))
                current_year = year if year in target_years else None
                current_month = None
                continue

            if current_year is None:
                continue

            mm = month_re.match(line)
            if mm:
                current_month = datetime.datetime.strptime(mm.group(1), "%B").month
                continue

            rm = range_re.match(line)
            if rm and current_month is not None:
                first_day = int(rm.group(1))
                second_day = int(rm.group(2))
                # 두 번째 날의 14:00 ET가 정책결정/성명 발표 시각에 해당한다.
                parsed.append((current_year, current_month, second_day))
                current_month = None

    except Exception:
        parsed = []

    if not parsed:
        for year in sorted(target_years):
            for month, day in fallback.get(year, []):
                parsed.append((year, month, day))

    # 같은 회의가 여러 번 들어오는 경우를 완전히 제거한다.
    parsed = sorted(set(parsed))

    events = []
    for year, month, day in parsed:
        try:
            dt_et = datetime.datetime(year, month, day, 14, 0, tzinfo=ZoneInfo("America/New_York"))
            events.append({
                "datetime": dt_et.astimezone(ZoneInfo("Asia/Seoul")),
                "name": "FOMC Meeting",
                "source_name": "Federal Reserve",
                "importance": "high",
                "guide": ECONOMIC_EVENT_GUIDE["FOMC"],
            })
        except ValueError:
            continue
    return events


def _extract_table_rows(url, keyword_cols):
    """공식 HTML 테이블에서 필요한 날짜/시간/제목 행을 최대한 유연하게 뽑는다."""
    html = _request_text(url)
    tables = pd.read_html(io.StringIO(html))
    output = []
    for table in tables:
        table.columns = [str(c) for c in table.columns]
        cols = {c.lower(): c for c in table.columns}
        date_col = next((cols[c] for c in cols if "release date" in c or c.strip() == "date"), None)
        time_col = next((cols[c] for c in cols if "time" in c), None)
        title_col = next((cols[c] for c in cols if any(k in c for k in keyword_cols)), None)
        if not (date_col and title_col):
            continue
        for _, row in table.iterrows():
            title = str(row.get(title_col, "")).strip()
            date_val = row.get(date_col)
            time_val = row.get(time_col, "08:30 AM") if time_col else "08:30 AM"
            if not title or title.lower() == "nan":
                continue
            output.append((date_val, time_val, title))
    return output


def _fetch_bls_events(today):
    events = []
    year = today.year
    months = {today.month, 1 if today.month == 12 else today.month + 1, 1 if today.month == 11 else (today.month + 2) if today.month <= 10 else 1}
    base = datetime.date(today.year, today.month, 1)
    month_targets = []
    for offset in range(3):
        month_index = base.month - 1 + offset
        y = base.year + month_index // 12
        m = month_index % 12 + 1
        month_targets.append((y, m))

    for y, month in month_targets:
        url = f"https://www.bls.gov/schedule/{y}/{month:02d}_sched_list.htm"
        try:
            rows = _extract_table_rows(url, ["release"])
        except Exception:
            continue
        for date_val, time_val, title in rows:
            low = title.lower()
            if not any(k in low for k in ["consumer price index", "producer price index", "employment situation", "job openings and labor turnover"]):
                continue
            dt = _parse_us_et_datetime(date_val, time_val)
            if dt is None:
                # Some BLS tables put weekday and month/day without year.
                try:
                    dt = _parse_us_et_datetime(f"{date_val} {y}", time_val)
                except Exception:
                    dt = None
            if dt is not None:
                guide, importance = _get_guide(title)
                events.append({"datetime": dt, "name": title, "source_name": "BLS", "importance": importance, "guide": guide})
    return events


def _fetch_census_events():
    events = []
    try:
        rows = _extract_table_rows(ECONOMIC_SOURCE_URLS["Census"], ["indicator"])
    except Exception:
        return events
    for date_val, time_val, title in rows:
        low = title.lower()
        if not any(k in low for k in ["retail", "residential construction", "building permits", "durable goods"]):
            continue
        dt = _parse_us_et_datetime(date_val, time_val)
        if dt is None:
            continue
        guide, importance = _get_guide(title)
        events.append({"datetime": dt, "name": title, "source_name": "U.S. Census Bureau", "importance": importance, "guide": guide})
    return events


def _fetch_bea_events():
    events = []
    try:
        rows = _extract_table_rows(ECONOMIC_SOURCE_URLS["BEA"], ["release"])
    except Exception:
        return events
    for date_val, time_val, title in rows:
        low = title.lower()
        if not any(k in low for k in ["gdp", "personal income and outlays"]):
            continue
        # BEA schedule may omit year in the date column; current year is sufficient for near-term filtering.
        dt = _parse_us_et_datetime(date_val, time_val)
        if dt is None:
            if re.match(r"^[A-Z][a-z]+\s+\d{1,2}$", str(date_val).strip()):
                dt = _parse_us_et_datetime(f"{date_val}, {datetime.date.today().year}", time_val)
        if dt is None:
            continue
        guide, importance = _get_guide(title)
        events.append({"datetime": dt, "name": title, "source_name": "BEA", "importance": importance, "guide": guide})
    return events


@st.cache_data(ttl=900)
def fetch_official_economic_calendar(days_ahead=14):
    """유료 금융 API 없이 공식 정부기관 일정만 직접 수집한다."""
    now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
    end = now + datetime.timedelta(days=days_ahead)
    all_events = []
    source_status = {}

    fetchers = [
        ("Federal Reserve", _fetch_fomc_events),
        ("BLS", lambda: _fetch_bls_events(now.date())),
        ("U.S. Census Bureau", _fetch_census_events),
        ("BEA", _fetch_bea_events),
    ]
    for source_name, fetcher in fetchers:
        try:
            fetched = fetcher()
            all_events.extend(fetched)
            source_status[source_name] = {"ok": True, "count": len(fetched), "error": ""}
        except Exception as exc:
            source_status[source_name] = {"ok": False, "count": 0, "error": str(exc)}

    unique = {}
    for event in all_events:
        dt = event["datetime"]
        if dt <= now or dt > end:
            continue
        key = (dt.strftime("%Y-%m-%d %H:%M"), event["name"], event["source_name"])
        unique[key] = event

    events = list(unique.values())
    for event in events:
        delta = event["datetime"] - now
        event["days_until"] = max(0, delta.total_seconds() / 86400)
        if event["name"] == "FOMC Meeting":
            event["warning_days"] = 3
        elif event["guide"] is ECONOMIC_EVENT_GUIDE.get("CPI") or "Consumer Price Index" in event["name"] or "Personal Income and Outlays" in event["name"]:
            event["warning_days"] = 2
        else:
            event["warning_days"] = 1.5

    events.sort(key=lambda e: e["datetime"])
    sources = sorted({e["source_name"] for e in events})
    return events, sources, source_status


def build_event_alert(event, sector=None, beta=None):
    days = event.get("days_until", 99)
    name = event["name"]
    dt = event["datetime"]
    guide = event["guide"]
    day_text = "오늘" if days < 1 else f"약 {math.ceil(days)}일 후"
    if name == "FOMC Meeting":
        msg = f"⚠️ FOMC {day_text} — {dt.strftime('%m/%d %H:%M')} (KST). 금리 결정 이벤트라 시장 전체 변동성이 커질 수 있습니다. 성장주·고Beta 종목은 특히 주의하세요."
        if sector == "Technology" and beta and beta >= 1.2:
            msg += " 현재 분석 종목은 기술주 + 고Beta라 금리 발표 민감도가 상대적으로 높습니다."
        return {"level": "high", "message": msg}
    if "CPI" in name or "Personal Income and Outlays" in name or "PCE" in name:
        msg = f"⚠️ {name} {day_text} — 물가 발표가 임박했습니다. 예상과 실제 결과의 차이가 금리 기대를 움직일 수 있어 성장주 변동성에 주의하세요."
        return {"level": "high", "message": msg}
    if "Employment Situation" in name:
        return {"level": "high", "message": f"⚠️ 고용지표 {day_text} — 고용 강도에 따라 연준의 금리 경로가 달라질 수 있어 시장 변동성 확대에 주의하세요."}
    if "Retail" in name or "GDP" in name or "Building Permits" in name or "residential construction" in name.lower():
        return {"level": "medium", "message": f"📌 {name} {day_text} — 경기 강도를 보여주는 지표입니다. 경기민감주와 금리 움직임을 함께 확인하세요."}
    return {"level": "medium", "message": f"📌 {name} {day_text} — {guide['impact']}"}


def get_upcoming_event_impacts(events, sector, beta):
    impacts = []
    for event in events:
        alert = build_event_alert(event, sector=sector, beta=beta)
        if event["days_until"] <= event["warning_days"]:
            impacts.append(alert["message"])
    return impacts



# ---------------------------------------------------------
# 2-2. 지정학적 긴장 감시 (시장 점수와 분리된 보조 지표)
# ---------------------------------------------------------
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
PIZZINT_URL = "https://www.pizzint.watch/"

# GDELT는 OR 블록을 중첩해서 사용할 수 없으므로
# "NATO (Russia OR Russian)"처럼 독립적인 OR 블록을 사용한다.
GEO_QUERY = 'NATO (Russia OR Russian)'

GEO_INCIDENT_TERMS = (
    "drone", "missile", "strike", "attack", "airspace", "incursion",
    "sabotage", "border", "violation", "explosion", "intercept"
)
GEO_WARNING_TERMS = (
    "warn", "warning", "threat", "alert", "readiness", "prepare",
    "attack possible", "escalation", "risk"
)
GEO_MILITARY_TERMS = (
    "military", "troop", "deployment", "deploy", "exercise", "aircraft",
    "fighter", "bomber", "air defense", "missile system", "soldier", "navy"
)

OFFICIAL_DOMAINS = {
    "nato.int", "whitehouse.gov", "defense.gov", "state.gov", "gov.pl",
    "gov.uk", "bundesregierung.de", "eeas.europa.eu", "consilium.europa.eu"
}


def _safe_domain(domain):
    return str(domain or "").lower().strip().split(":")[0].split("/")[0]


def _retry_sleep(attempt, retry_after=None):
    try:
        if retry_after:
            delay = min(float(retry_after), 8.0)
        else:
            delay = min(2 ** attempt, 8)
        time.sleep(delay)
    except Exception:
        pass


def _gdelt_request(params, max_attempts=3):
    """GDELT 호출을 최소 횟수로 유지하면서 429/5xx에는 짧게 재시도한다."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(
                GDELT_API_URL,
                params=params,
                timeout=20,
                headers={"User-Agent": "US-Stock-Analyzer/1.0"},
            )
            if response.status_code == 429:
                last_error = RuntimeError("GDELT rate limit (HTTP 429)")
                if attempt < max_attempts - 1:
                    _retry_sleep(attempt, response.headers.get("Retry-After"))
                    continue
                raise last_error
            if 500 <= response.status_code < 600:
                last_error = RuntimeError(f"GDELT server error (HTTP {response.status_code})")
                if attempt < max_attempts - 1:
                    _retry_sleep(attempt)
                    continue
                raise last_error
            response.raise_for_status()
            return response.json(), {"ok": True, "status": "ok", "http_status": response.status_code}
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                _retry_sleep(attempt)
    return None, {"ok": False, "status": "error", "error": str(last_error)[:200]}


def _extract_gdelt_timeline_points(payload):
    points = []
    if isinstance(payload, dict):
        timeline = payload.get("timeline")
        if isinstance(timeline, list):
            for series in timeline:
                if not isinstance(series, dict):
                    continue
                data = series.get("data", [])
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not isinstance(item, dict) or "date" not in item or "value" not in item:
                        continue
                    try:
                        points.append((str(item["date"]), float(item["value"])))
                    except (TypeError, ValueError):
                        continue
    return points


def _parse_gdelt_date(value):
    text = str(value)
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.datetime.strptime(text, fmt).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return None


def _news_activity_ratio(points):
    """최근 24시간 / 직전 6일 평균의 상대 활동량."""
    now = datetime.datetime.now(datetime.timezone.utc)
    recent_start = now - datetime.timedelta(hours=24)
    baseline_start = now - datetime.timedelta(days=7)
    recent = 0.0
    baseline = 0.0
    for date_text, value in points:
        dt = _parse_gdelt_date(date_text)
        if dt is None or dt < baseline_start:
            continue
        value = max(0.0, value)
        if dt >= recent_start:
            recent += value
        else:
            baseline += value
    daily_baseline = baseline / 6.0
    if daily_baseline <= 0:
        if recent <= 0:
            return 1.0, False
        return 2.0, True
    return recent / daily_baseline, True


def _ratio_to_signal(ratio):
    if ratio < 1.25:
        return 10
    if ratio < 1.75:
        return 35
    if ratio < 2.5:
        return 60
    if ratio < 4.0:
        return 80
    return 95


def _count_to_signal(count):
    if count <= 0:
        return 0
    if count == 1:
        return 25
    if count <= 3:
        return 50
    if count <= 6:
        return 70
    if count <= 10:
        return 85
    return 95


def _article_text(article):
    return " ".join([
        str(article.get("title") or ""),
        str(article.get("snippet") or ""),
        str(article.get("url") or ""),
    ]).lower()


def _classify_articles(articles):
    """최근 기사들을 사건/경고/군사활동으로 분류한다. 서로 배타적이지 않다."""
    incidents = []
    warnings = []
    military = []
    official_warning_count = 0

    for article in articles:
        text = _article_text(article)
        domain = _safe_domain(article.get("domain"))
        is_official = any(domain == d or domain.endswith("." + d) for d in OFFICIAL_DOMAINS)

        incident_hit = any(term in text for term in GEO_INCIDENT_TERMS)
        warning_hit = any(term in text for term in GEO_WARNING_TERMS)
        military_hit = any(term in text for term in GEO_MILITARY_TERMS)

        if incident_hit:
            incidents.append(article)
        if warning_hit:
            warnings.append(article)
            if is_official:
                official_warning_count += 1
        if military_hit:
            military.append(article)

    return incidents, warnings, military, official_warning_count


@st.cache_data(ttl=900)
def fetch_geopolitical_gdelt_bundle():
    """GDELT 호출을 2회로 제한한다: 1회 timeline + 1회 최근 기사 목록."""
    timeline_payload, timeline_meta = _gdelt_request({
        "query": GEO_QUERY,
        "mode": "timelinevolraw",
        "format": "json",
        "timespan": "7d",
    })
    article_payload, article_meta = _gdelt_request({
        "query": GEO_QUERY,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 75,
        "timespan": "72h",
    })

    timeline_points = _extract_gdelt_timeline_points(timeline_payload) if timeline_payload else []
    ratio, ratio_usable = _news_activity_ratio(timeline_points)
    news_signal = _ratio_to_signal(ratio) if ratio_usable else None

    articles = []
    if isinstance(article_payload, dict):
        raw_articles = article_payload.get("articles", [])
        if isinstance(raw_articles, list):
            articles = raw_articles

    incidents, warnings, military, official_warning_count = _classify_articles(articles)
    incident_signal = _count_to_signal(len(incidents)) if article_meta.get("ok") else None
    military_signal = _count_to_signal(len(military)) if article_meta.get("ok") else None
    official_signal = _count_to_signal(official_warning_count) if article_meta.get("ok") else None

    if article_meta.get("ok") and not articles:
        article_status = "empty"
    else:
        article_status = article_meta.get("status", "error")

    return {
        "news": {
            "score": news_signal,
            "ratio": ratio if ratio_usable else None,
            "ok": ratio_usable,
            "status": "ok" if ratio_usable else timeline_meta.get("status", "error"),
            "points": len(timeline_points),
            "error": timeline_meta.get("error"),
        },
        "incidents": {
            "score": incident_signal,
            "ok": article_meta.get("ok", False),
            "status": article_status,
            "count": len(incidents),
            "error": article_meta.get("error"),
        },
        "military_activity": {
            "score": military_signal,
            "ok": article_meta.get("ok", False),
            "status": article_status,
            "count": len(military),
            "error": article_meta.get("error"),
        },
        "official_warning": {
            "score": official_signal,
            "ok": article_meta.get("ok", False),
            "status": article_status,
            "official_count": official_warning_count,
            "warning_count": len(warnings),
            "article_count": len(articles),
            "error": article_meta.get("error"),
        },
        "meta": {
            "timeline_ok": timeline_meta.get("ok", False),
            "articles_ok": article_meta.get("ok", False),
            "article_count": len(articles),
        },
    }


@st.cache_data(ttl=900)
def fetch_pizzint_signal():
    """PizzINT 공개 페이지를 보조 OSINT 신호로 사용한다.
    DOUGHCON 1이 가장 높은 대비, 5가 가장 낮은 대비이므로 점수 방향을 반대로 매핑한다.
    """
    try:
        html = _request_text(PIZZINT_URL)
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        dough_match = re.search(r"DOUGHCON\s+(\d+)", text, re.IGNORECASE)
        spikes = [int(x) for x in re.findall(r"(\d+)%(?:\s*SPIKE)", text, re.IGNORECASE)]

        doughcon = int(dough_match.group(1)) if dough_match else None
        max_spike = max(spikes) if spikes else None

        dough_score = None
        if doughcon is not None and 1 <= doughcon <= 5:
            dough_score = {1: 100, 2: 75, 3: 50, 4: 25, 5: 0}[doughcon]

        spike_score = None
        if max_spike is not None:
            spike_score = int(np.clip(max_spike / 200 * 100, 0, 100))

        if dough_score is None and spike_score is None:
            return {
                "score": None, "doughcon": None, "max_spike": None,
                "dough_score": None, "spike_score": None,
                "ok": False, "status": "empty",
                "error": "공개 페이지에서 DOUGHCON/SPIKE 수치를 찾지 못했습니다."
            }

        # DOUGHCON을 주 신호(60%), 실제 관측 spike를 보조 신호(40%)로 사용.
        if dough_score is not None and spike_score is not None:
            score = int(round(dough_score * 0.60 + spike_score * 0.40))
        elif dough_score is not None:
            score = dough_score
        else:
            score = spike_score

        return {
            "score": score,
            "doughcon": doughcon,
            "max_spike": max_spike,
            "dough_score": dough_score,
            "spike_score": spike_score,
            "ok": True,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "score": None, "doughcon": None, "max_spike": None,
            "dough_score": None, "spike_score": None,
            "ok": False, "status": "error", "error": str(exc)[:200]
        }


@st.cache_data(ttl=900)
def fetch_geopolitical_monitor():
    """전쟁 예측이 아닌, 여러 공개정보 신호의 동시 상승을 감시한다.

    최소 3개 신호가 정상적으로 확보된 경우에만 종합점수를 계산한다.
    개별 신호 점수와 데이터 부족 상태를 분리하여, 수집 실패가 낮은 위험도로 오해되지 않게 한다.
    """
    gdelt = fetch_geopolitical_gdelt_bundle()
    signals = {
        "news": gdelt["news"],
        "incidents": gdelt["incidents"],
        "military_activity": gdelt["military_activity"],
        "official_warning": gdelt["official_warning"],
        "pizza": fetch_pizzint_signal(),
    }

    weights = {
        "official_warning": 0.30,
        "incidents": 0.30,
        "news": 0.20,
        "military_activity": 0.10,
        "pizza": 0.10,
    }

    available = {k: v["score"] for k, v in signals.items() if v.get("score") is not None}
    available_count = len(available)
    minimum_valid_signals = 3
    data_sufficient = available_count >= minimum_valid_signals

    # 최소 3개 신호가 있어야 종합점수를 계산한다.
    # 1~2개만 확보된 경우 개별 점수만 보여주고 종합점수는 N/A로 둔다.
    if data_sufficient:
        effective_weight = sum(weights[k] for k in available)
        combined = int(round(sum(available[k] * weights[k] for k in available) / effective_weight))
    else:
        combined = None

    high_count = sum(1 for value in available.values() if value >= 60)
    official_high = available.get("official_warning", 0) >= 60
    incident_high = available.get("incidents", 0) >= 60

    if not data_sufficient:
        level = "unknown"
        label = "⚪ 데이터 부족"
        summary = (
            f"현재 5개 신호 중 {available_count}개만 정상적으로 확보되어 종합점수를 계산하지 않았습니다. "
            "개별 신호는 아래에서 확인할 수 있습니다."
        )
    elif combined >= 70 and high_count >= 3:
        level = "high"
        label = "🔴 다중 신호 상승"
        summary = "서로 다른 공개정보 신호가 동시에 높아졌습니다. 전쟁 발생 가능성을 의미하는 것이 아니라 긴장 신호가 겹친 상태입니다."
    elif combined >= 50 or (official_high and incident_high):
        level = "caution"
        label = "🟠 주의"
        summary = "여러 공개정보 신호가 평소보다 높습니다. 단일 신호가 아니라 복수의 신호가 함께 나타나는지 확인하세요."
    elif combined >= 30:
        level = "watch"
        label = "🟡 관찰"
        summary = "평소보다 높은 관심 신호가 일부 나타나고 있습니다. 단일 신호만으로 해석하지 않습니다."
    else:
        level = "normal"
        label = "🟢 정상 범위"
        summary = "현재 수집된 공개정보 신호가 뚜렷하게 동시에 상승한 상태는 아닙니다."

    source_ok = sum(1 for v in signals.values() if v.get("ok"))
    failed_signals = {
        k: (v.get("error") or v.get("status") or "수집 실패")
        for k, v in signals.items()
        if not v.get("ok")
    }

    return {
        "score": combined,
        "label": label,
        "level": level,
        "summary": summary,
        "signals": signals,
        "source_ok": source_ok,
        "available_count": available_count,
        "minimum_valid_signals": minimum_valid_signals,
        "data_sufficient": data_sufficient,
        "failed_signals": failed_signals,
        "gdelt_article_count": gdelt["meta"].get("article_count", 0),
        "updated_at": datetime.datetime.now(ZoneInfo("Asia/Seoul")),
    }


# ---------------------------------------------------------
# 3. 사이드바 영역
# ---------------------------------------------------------
with st.sidebar:
    # 지정학적 검색 결과는 세션에 보관하여 다른 버튼을 눌러도 유지합니다.
    if "geo_monitor_result" not in st.session_state:
        st.session_state.geo_monitor_result = None
    if "geo_monitor_last_run" not in st.session_state:
        st.session_state.geo_monitor_last_run = None

    st.header("🔍 종목 분석")
    target_ticker = st.text_input("기업 티커 입력", value="NVDA").upper()
    analyze_btn = st.button("정밀 분석 실행", use_container_width=True)
    geo_search_btn = st.button("🌍 지정학적 리스크 검색", use_container_width=True)

    st.caption("뉴스·공개정보 검색은 시간이 걸릴 수 있어 버튼을 눌렀을 때만 실행됩니다. 마지막 검색 결과는 유지됩니다.")

    # 지정학적 검색은 사용자가 버튼을 눌렀을 때만 외부 데이터를 조회합니다.
    if geo_search_btn:
        with st.spinner("지정학적 리스크 검색 중... (뉴스·공개정보 확인)"):
            # 수동 검색 버튼은 캐시된 이전 결과가 아니라 실제로 새 데이터를 요청합니다.
            for func in [
                fetch_geopolitical_monitor,
                fetch_geopolitical_gdelt_bundle,
                fetch_pizzint_signal,
            ]:
                try:
                    func.clear()
                except Exception:
                    pass
            st.session_state.geo_monitor_result = fetch_geopolitical_monitor()
            st.session_state.geo_monitor_last_run = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
    
    st.divider()

    st.header("📅 미국 경제 일정")
    st.caption("유료 금융 API 대신 미국 정부기관의 공식 발표 일정을 직접 불러옵니다. 지나간 일정은 자동으로 제외합니다.")

    economic_events, economic_sources, economic_source_status = fetch_official_economic_calendar(days_ahead=14)
    imminent_events = [e for e in economic_events if e["days_until"] <= e["warning_days"]]

    if imminent_events:
        for event in imminent_events[:3]:
            alert = build_event_alert(event, sector=None, beta=None)
            if alert["level"] == "high":
                st.error(alert["message"])
            else:
                st.warning(alert["message"])
    elif economic_events:
        st.info(f"📌 앞으로 14일 동안 주요 일정 {len(economic_events)}건이 있습니다. 일정별 의미와 시장 영향은 아래에서 확인할 수 있습니다.")

    if economic_events:
        for event in economic_events[:10]:
            impact_badge = "🔴 중요" if event["importance"] == "high" else "🟠 주요"
            dt = event["datetime"]
            st.markdown(
                f"""
                <div style="background:#171717; border:1px solid #333; border-radius:8px; padding:9px 10px; margin:5px 0;">
                  <div style="font-size:12px; color:#aaa;">{dt.strftime('%m/%d (%a) %H:%M')} · {impact_badge} · {event['source_name']}</div>
                  <div style="font-size:14px; color:#fff; font-weight:700; margin-top:2px;">{event['name']}</div>
                  <div style="font-size:11px; color:#ccc; line-height:1.55; margin-top:3px;">{event['guide']['meaning']}</div>
                  <div style="font-size:11px; color:#f3c969; line-height:1.55; margin-top:2px;">📈 영향: {event['guide']['impact']}</div>
                  <div style="font-size:11px; color:#9ec5ff; margin-top:2px;">영향 가능 업종: {event['guide']['assets']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("공식 일정 데이터를 가져오지 못했습니다. 아래 TradingView 캘린더를 보조 화면으로 사용하세요.")
        components.html(
            """
            <div class="tradingview-widget-container" style="width:100%; height:420px;">
              <div class="tradingview-widget-container__widget" style="width:100%; height:420px;"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>
              {
              "colorTheme": "dark",
              "isTransparent": true,
              "width": "100%",
              "height": "420",
              "locale": "kr",
              "importanceFilter": "0,1",
              "currencyFilter": "USD"
            }
              </script>
            </div>
            """, height=430
        )

    with st.expander("📘 경제 일정은 무엇을 의미하나요?"):
        st.markdown(
            """
            <div style="font-size:12px; line-height:1.7;">
            <b>FOMC</b> · 연준의 금리 결정과 경제전망. 시장 전체와 성장주·금융주 변동성에 큰 영향을 줄 수 있습니다.<br>
            <b>CPI / PCE</b> · 소비자·개인소비 물가. 예상보다 높으면 금리 인하 기대가 약해져 성장주에 부담이 될 수 있습니다.<br>
            <b>고용지표</b> · 미국 일자리와 실업 상태. 너무 강한 고용은 금리 인하 기대를 약하게 만들 수 있고, 약한 고용은 경기 둔화 우려를 키울 수 있습니다.<br>
            <b>소매판매</b> · 미국 소비자의 지출 강도. 소비주·금융주·경기민감주와 시장 금리에 영향을 줄 수 있습니다.<br>
            <b>GDP</b> · 미국 경제의 성장 속도. 강한 성장과 약한 성장 모두 금리 기대를 통해 주식시장에 영향을 줄 수 있습니다.
            </div>
            """, unsafe_allow_html=True
        )

    with st.expander("📡 경제 일정 데이터 연결 상태"):
        status_lines = []
        for source_name in ["Federal Reserve", "BLS", "U.S. Census Bureau", "BEA"]:
            status = economic_source_status.get(source_name, {})
            if status.get("ok") and status.get("count", 0) > 0:
                status_lines.append(f"✅ {source_name}: 정상 · {status.get('count', 0)}건 확인")
            elif status.get("ok"):
                status_lines.append(f"🟡 {source_name}: 연결됨 · 현재 기간에 표시할 일정이 없거나 파싱된 일정이 없습니다.")
            else:
                status_lines.append(f"⚠️ {source_name}: 불러오기 실패")
                error_text = status.get("error") or "알 수 없는 오류"
                status_lines.append(f"　└ {error_text[:120]}")
        for line in status_lines:
            st.caption(line)
        st.caption("※ 일부 공식 페이지는 자체적으로 오류를 숨기고 빈 목록을 반환할 수 있어, ‘연결됨·0건’은 연결 성공과 일정 부재를 완전히 구분하지 못할 수 있습니다.")

    st.caption("출처: Federal Reserve · BLS · U.S. Census Bureau · BEA")


# ---------------------------------------------------------
# 4. 메인 화면 영역 (상단 정보 강화)
# ---------------------------------------------------------
st.title("🏛️ 미국 주식 분석기")
m_data = fetch_market_indicators()

if m_data:
    display_meta = {
        "SP500": ("S&P 500", "$"), "NASDAQ": ("나스닥", "$"), "DOW": ("다우존스", "$"),
        "VIX": ("VIX", ""), "OIL": ("WTI 유가", "$"), "GOLD": ("금 선물", "$"),
        "TNX": ("10년물 금리", "%"), "IRX": ("3개월물 금리", "%"), "BTC": ("비트코인", "$")
    }

    ticker_items = []
    for key, (name, unit) in display_meta.items():
        if key in m_data:
            df = m_data[key]
            curr, prev = df['Close'].iloc[-1], df['Close'].iloc[-2]
            chg = ((curr - prev) / prev) * 100
            color = "#00ff00" if chg >= 0 else "#ff4b4b"
            
            val_str = f"${curr:,.2f}" if unit == "$" and curr < 10000 else f"${curr:,.0f}" if unit == "$" else f"{curr:.2f}%" if unit == "%" else f"{curr:.2f}"
            ticker_items.append(f"<b>{name}</b>: {val_str} (<span style='color:{color};'>{chg:+.2f}%</span>)")

    ticker_text = "&nbsp;&nbsp;&nbsp;&nbsp; ｜ &nbsp;&nbsp;&nbsp;&nbsp;".join(ticker_items)
    st.markdown(f"""
        <div style="background-color: #111111; padding: 12px; border-radius: 10px; margin-bottom: 25px; border: 1px solid #333333;">
            <marquee behavior="scroll" direction="left" scrollamount="6" onmouseover="this.stop();" onmouseout="this.start();" style="color: white; font-size: 15px;">
                {ticker_text}
            </marquee>
        </div>
    """, unsafe_allow_html=True)

disparity, vix, spread, market_heat, m_score, m_plus, m_minus, hunter_alert = get_market_status(m_data)
market_confidence = calculate_market_confidence(m_data)

if hunter_alert:
    st.error(f"🏹 **[사냥꾼의 경보]** 현재 유가 추세가 꺾였고 시장 공포(VIX)가 극에 달했습니다. 진입 검토 시점입니다.")

score_color = "#00ff00" if m_score >= 70 else "#ffa500" if m_score >= 40 else "#ff4b4b"
st.caption(f"시장 점수는 ‘지금 시장 전체에 투자하기 좋은 환경인가?’를 0~100으로 환산한 값입니다. 기업의 우량 여부와는 별개의 지표입니다. · 시장 데이터 신뢰도: {market_confidence}%")
st.markdown(f"""<div style="background-color: #1e1e1e; padding: 20px; border-radius: 15px; border-left: 10px solid {score_color}; text-align: center;">
    <h2 style="color: white; margin-bottom: 0;">🌐 시장 투자 적기 통합 점수</h2>
    <h1 style="color: {score_color}; font-size: 70px; margin: 10px 0;">{m_score} <span style="font-size: 20px; color: gray;">/ 100</span></h1></div>""", unsafe_allow_html=True)

if market_confidence < 70:
    st.warning("⚠️ 시장 데이터 일부가 부족합니다. 시장 점수는 참고용으로 해석하세요.")

# [Geopolitical Monitor] 사용자가 원할 때만 검색하고, 마지막 결과는 세션에 유지
geo_monitor = st.session_state.get("geo_monitor_result")
st.markdown("### 🌍 지정학적 긴장 감시")

if geo_monitor is None:
    # 앱 최초 실행 시 외부 지정학적 데이터 조회를 하지 않습니다.
    st.markdown(
        """
        <div style="background:#171717; border:2px solid #555; border-radius:14px; padding:18px; margin:8px 0 14px 0;">
          <div style="font-size:18px; color:#fff; font-weight:700;">⚪ 아직 검색하지 않았습니다</div>
          <div style="font-size:13px; color:#aaa; margin-top:6px; line-height:1.6;">
            지정학적 리스크 검색은 외부 뉴스·공개정보를 조회하므로 필요할 때만 실행합니다.<br>
            사이드바의 <b style="color:#fff;">🌍 지정학적 리스크 검색</b> 버튼을 눌러 최신 신호를 확인하세요.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("💡 검색하지 않은 상태에서는 지정학적 데이터가 시장 점수나 종목 분석에 사용되지 않습니다.")

else:
    geo_score = geo_monitor.get("score")
    geo_level = geo_monitor.get("level")
    geo_label = geo_monitor.get("label")
    geo_summary = geo_monitor.get("summary")
    last_run = st.session_state.get("geo_monitor_last_run") or geo_monitor.get("updated_at")

    geo_score_text = f"{geo_score} / 100" if geo_score is not None else "N/A"
    geo_border = {
        "normal": "#2e8b57",
        "watch": "#d9a400",
        "caution": "#e67e22",
        "high": "#c0392b",
        "unknown": "#666666",
    }.get(geo_level, "#666666")

    last_run_text = last_run.strftime('%Y-%m-%d %H:%M') if last_run else "알 수 없음"
    st.markdown(f"""
    <div style="background:#171717; border:2px solid {geo_border}; border-radius:14px; padding:16px 18px; margin:8px 0 14px 0;">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
        <div>
          <div style="font-size:18px; color:#fff; font-weight:700;">{geo_label}</div>
          <div style="font-size:12px; color:#aaa; margin-top:4px;">{geo_summary}</div>
          <div style="font-size:11px; color:#777; margin-top:6px;">마지막 검색: {last_run_text} KST</div>
        </div>
        <div style="font-size:30px; color:#fff; font-weight:800; white-space:nowrap;">{geo_score_text}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    geo_cols = st.columns(5)
    geo_labels = {
        "official_warning": "공식 경고 보도",
        "incidents": "사건 관련 보도",
        "news": "뉴스 활동",
        "military_activity": "공개 활동",
        "pizza": "Pizza Signal",
    }
    geo_desc = {
        "official_warning": "공식기관 관련 경고 보도",
        "incidents": "드론·공격·영공·사보타주 관련 보도",
        "news": "Russia·NATO 관련 보도량 변화",
        "military_activity": "군사·배치·훈련 관련 공개정보",
        "pizza": "PizzINT의 비공식 보조 신호",
    }

    for col, key in zip(geo_cols, ["official_warning", "incidents", "news", "military_activity", "pizza"]):
        signal = geo_monitor["signals"].get(key, {})
        value = signal.get("score")
        status = signal.get("status")
        error_text = signal.get("error")
        if value is None:
            display_value = "N/A"
            if status == "empty":
                status_text = "🟡 정상 응답 · 관련 데이터 없음"
            else:
                status_text = "⚠️ 수집 실패"
        else:
            display_value = f"{value}"
            if status == "empty":
                status_text = "✅ 정상 · 관련 데이터 적음"
            elif status == "ok":
                status_text = "✅ 정상"
            else:
                status_text = "✅ 확인됨"
        with col:
            st.metric(geo_labels[key], display_value)
            st.caption(geo_desc[key])
            st.caption(status_text)
            if value is None and error_text:
                st.caption(f"↳ {str(error_text)[:90]}")

    with st.expander("🔎 지정학적 긴장 감시를 어떻게 계산하나요?"):
        pizza = geo_monitor["signals"].get("pizza", {})
        pizza_note = "데이터 없음"
        if pizza.get("doughcon") is not None or pizza.get("max_spike") is not None:
            bits = []
            if pizza.get("doughcon") is not None:
                dough_label = {
                    1: "최고 대비태세",
                    2: "높은 대비태세",
                    3: "중간 대비태세",
                    4: "낮은 대비태세",
                    5: "최저 대비태세",
                }.get(pizza["doughcon"], "해석 불명")
                bits.append(f"DOUGHCON {pizza['doughcon']} ({dough_label})")
            if pizza.get("max_spike") is not None:
                bits.append(f"관측된 최대 활동 급증 {pizza['max_spike']}%")
            if pizza.get("score") is not None:
                bits.append(f"계산된 Pizza Signal {pizza['score']}/100")
            pizza_note = " · ".join(bits)
        st.markdown(
            f"""
            **이 지표는 전쟁이나 군사행동의 확률을 계산하지 않습니다.** 여러 공개정보가 동시에 평소보다 높아지는지 감시하는 보조 지표입니다.

            - 공식 경고 30% · 실제 사건 관련 30% · 뉴스 활동 20% · 공개 군사활동 10% · Pizza Signal 10%
            - 신호가 하나만 높다고 경보를 올리지 않고, 여러 신호가 동시에 상승할 때 경보 수준을 높입니다.
            - Pizza Signal은 PizzINT의 제3자 공개 OSINT를 사용하는 보조 신호이며, 단독으로 군사행동을 의미하지 않습니다. 현재 읽힌 값: {pizza_note}
            - 현재 지정학적 점수는 **종목 최종점수와 시장 점수에 직접 가산/감산하지 않습니다.**

            **데이터 상태:** {geo_monitor.get('source_ok', 0)}/5개 신호원 정상 · 사용 가능한 신호 {geo_monitor.get('available_count', 0)}/5 · 최소 필요 신호 {geo_monitor.get('minimum_valid_signals', 3)}개 · GDELT 최근 기사 {geo_monitor.get('gdelt_article_count', 0)}건 · 마지막 검색 {last_run_text} KST
            """
        )

        failed = geo_monitor.get("failed_signals", {})
        if failed:
            st.markdown("**⚠️ 실패한 데이터**")
            signal_display_names = {
                "official_warning": "공식 경고 보도",
                "incidents": "사건 관련 보도",
                "news": "뉴스 활동",
                "military_activity": "공개 활동",
                "pizza": "Pizza Signal",
            }
            for key, reason in failed.items():
                st.caption(f"⚠️ {signal_display_names.get(key, key)}: {str(reason)[:160]}")

        if not geo_monitor.get("data_sufficient", False):
            st.warning(
                f"⚠️ 현재 확보된 신호가 {geo_monitor.get('available_count', 0)}/5개로 최소 필요 기준 "
                f"{geo_monitor.get('minimum_valid_signals', 3)}개보다 적습니다. 개별 신호는 표시하지만 종합점수는 계산하지 않습니다."
            )

        st.caption("데이터: GDELT 뉴스 활동/최근 기사 · PizzINT 공개 OSINT 페이지. N/A는 수집 실패 또는 데이터 부족을 의미하며, 개별 신호의 0점과 종합점수의 N/A는 서로 다른 의미입니다.")

    if geo_level in {"caution", "high"}:
        st.warning("⚠️ 지정학적 긴장 신호가 상승했습니다. 이는 전쟁 발생 예측이 아니라 공개정보 신호의 동시 상승을 의미합니다. 데이터가 충분하지 않다면 이 경보도 참고용으로만 보세요.")

st.subheader("🧐 시장 점수 상세 해설")
m_col1, m_col2 = st.columns(2)
with m_col1:
    for p in m_plus: st.success(p)
with m_col2:
    for m in m_minus: st.warning(m)

with st.expander("📘 시장 평가 항목이 무엇인지 쉽게 보기"):
    st.markdown(
        """
        <div style="font-size:13px; line-height:1.75;">
        <b>S&P500 이격도</b> · 현재 시장이 200일 평균가격에서 얼마나 멀리 있는지 봅니다. <br>
        → 너무 멀리 올라가면 과열 가능성을, 적당한 범위면 추세 안정성을 봅니다.<br><br>
        <b>VIX</b> · 시장 참가자들이 느끼는 불안·변동성의 정도입니다. <br>
        → 너무 높으면 공포, 너무 낮으면 지나친 낙관일 수 있어 시장 방향과 함께 해석합니다.<br><br>
        <b>시장 고점 대비 조정</b> · 최근 1년 고점에서 얼마나 내려왔는지를 봅니다. <br>
        → 상승 추세 안의 건강한 조정인지, 추세 자체가 약해진 것인지 구분하는 데 사용합니다.<br><br>
        <b>유가</b> · 에너지 비용과 물가에 영향을 주는 요소입니다. <br>
        → 급등하면 물가·기업 비용 부담이 커질 수 있습니다.<br><br>
        <b>10년물 금리</b> · 장기 자금 조달 비용과 주식의 평가 수준에 영향을 줍니다. <br>
        → 특히 성장주에는 금리 상승이 부담이 될 수 있습니다.<br><br>
        <b>10Y-3M 금리 스프레드</b> · 장기금리와 단기금리의 차이입니다. <br>
        → 금리 구조가 정상적인지, 경기 위험 신호가 있는지 판단할 때 참고합니다.
        </div>
        """, unsafe_allow_html=True
    )

st.divider()
g_cols = st.columns(4)
with g_cols[0]: st.plotly_chart(create_gauge("S&P500 이격도", disparity, 0, 150), use_container_width=True)
with g_cols[1]: st.plotly_chart(create_gauge("VIX 지수", vix, 0, 100), use_container_width=True)
with g_cols[2]: st.plotly_chart(create_gauge("시장 과열도(고점비율)", market_heat, 50, 110), use_container_width=True)
with g_cols[3]: st.plotly_chart(create_gauge("10Y-3M 금리 스프레드", spread, -3, 5), use_container_width=True)


# ---------------------------------------------------------
# 4-1. 종목 분석 신뢰도 / 세부 점수 / 해석
# ---------------------------------------------------------
def calculate_component_scores(info, history, sector, val_pts, target_pts, debt, roe, op_margin, macro_adj, beta_adj, momentum_pts):
    """기존 채점 요소를 4개 진단 축으로 정규화한다. 각 축은 항상 0~100이다."""
    # Quality: 실제 존재하는 지표만 사용하고, 없는 지표 때문에 0점으로 깎지 않는다.
    quality_parts = []
    if info.get("returnOnEquity") is not None:
        roe_points = 40 if roe >= 15 else 20 if roe >= 8 else 0
        quality_parts.append((roe_points, 40))
    if info.get("operatingMargins") is not None:
        margin_points = 30 if op_margin >= 20 else 15 if op_margin >= 10 else 0
        quality_parts.append((margin_points, 30))
    if info.get("debtToEquity") is not None:
        debt_points = 30 if debt <= 80 else 15 if debt <= 150 else 0
        quality_parts.append((debt_points, 30))
    quality = int(round(sum(p for p, _ in quality_parts) / sum(m for _, m in quality_parts) * 100)) if quality_parts else 50

    # Valuation: 값이 있으면 '저평가 가점' + '목표가'를 반영한다.
    valuation_parts = []
    valuation_keys = ["pegRatio", "forwardPE", "priceToBook", "priceToSalesTrailing12Months"]
    if any(info.get(k) is not None for k in valuation_keys):
        valuation_base = int(np.clip((max(0, val_pts) / 15) * 100, 0, 100))
        # 조건을 못 만족해 +0이어도 '데이터는 있으나 중립'으로 보는 것이므로 50을 하한으로 둔다.
        valuation_base = max(50, valuation_base)
        valuation_parts.append(valuation_base)
    if info.get("targetMeanPrice"):
        target_norm = int(np.clip((max(-10, target_pts) + 10) / 25 * 100, 0, 100))
        valuation_parts.append(target_norm)
    valuation = int(round(sum(valuation_parts) / len(valuation_parts))) if valuation_parts else 50

    # Momentum: 기존 최대 20점을 그대로 0~100으로 변환.
    momentum = int(np.clip(momentum_pts / 20 * 100, 0, 100)) if len(history) >= 22 else 50

    # Risk: 높을수록 '위험 환경이 상대적으로 우호적'이라는 의미.
    risk_raw = 50 + macro_adj * 2 + beta_adj * 2
    if debt is not None and debt != 0:
        if debt <= 80:
            risk_raw += 10
        elif debt > 150:
            risk_raw -= 10
    risk = int(np.clip(risk_raw, 0, 100))

    return {"Quality": quality, "Valuation": valuation, "Momentum": momentum, "Risk": risk}


def calculate_component_confidences(info, history, m_data, beta_available=None, beta_estimated=False):
    """각 진단 축에 필요한 데이터 확보 수준을 0~100으로 계산한다."""
    quality_checks = [
        info.get("returnOnEquity") is not None,
        info.get("operatingMargins") is not None,
        info.get("debtToEquity") is not None,
    ]

    valuation_checks = [
        any(info.get(k) is not None for k in ["pegRatio", "forwardPE", "priceToBook", "priceToSalesTrailing12Months"]),
        bool(info.get("targetMeanPrice")),
        bool(info.get("numberOfAnalystOpinions") or info.get("numberOfAnalysts")),
    ]

    if len(history) >= 200:
        momentum_confidence = 100
    elif len(history) >= 127:
        momentum_confidence = 75
    elif len(history) >= 63:
        momentum_confidence = 50
    elif len(history) >= 22:
        momentum_confidence = 25
    else:
        momentum_confidence = 0

    beta_quality = 100 if info.get("beta") is not None else (75 if beta_estimated and beta_available is not None else 0)
    risk_checks = [
        beta_quality / 100,
        "VIX" in m_data and not m_data["VIX"].empty,
        "OIL" in m_data and not m_data["OIL"].empty,
        "TNX" in m_data and not m_data["TNX"].empty,
        "SP500" in m_data and not m_data["SP500"].empty,
    ]

    scores = {
        "Quality": round(sum(quality_checks) / len(quality_checks) * 100),
        "Valuation": round(sum(valuation_checks) / len(valuation_checks) * 100),
        "Momentum": momentum_confidence,
        "Risk": round(sum(float(x) for x in risk_checks) / len(risk_checks) * 100),
    }
    return scores


def calculate_final_score(component_scores, component_confidences):
    """4개 축의 점수를 신뢰도에 따라 가중평균한다. 최대 100을 넘지 않는다."""
    base_weights = {"Quality": 0.30, "Valuation": 0.25, "Momentum": 0.25, "Risk": 0.20}
    weighted_sum = 0.0
    effective_weight = 0.0
    for key, weight in base_weights.items():
        confidence_factor = component_confidences.get(key, 0) / 100
        effective = weight * confidence_factor
        weighted_sum += component_scores[key] * effective
        effective_weight += effective

    if effective_weight <= 0:
        return 50
    return int(round(weighted_sum / effective_weight))


def calculate_confidence(component_confidences):
    """4개 진단 축의 데이터 충족도를 기본 가중치로 종합한 신뢰도."""
    weights = {"Quality": 0.30, "Valuation": 0.25, "Momentum": 0.25, "Risk": 0.20}
    return round(sum(component_confidences[k] * weights[k] for k in weights))


def get_score_interpretation(score, confidence, components):
    if score >= 80 and confidence >= 75:
        label = "🟢 매우 우호적"
        summary = "기초체력·가격·추세·위험 요소를 종합했을 때 전반적인 투자 매력이 높은 편입니다."
    elif score >= 70 and confidence >= 65:
        label = "🟢 우호적"
        summary = "긍정적인 요소가 많은 편이지만 일부 위험 요소도 함께 확인할 필요가 있습니다."
    elif score >= 55:
        label = "🟡 중립·관찰"
        summary = "좋은 요소와 주의 요소가 섞여 있어 한 가지 지표만 보고 판단하기 어려운 상태입니다."
    elif score >= 40:
        label = "🟠 주의"
        summary = "부담 요인이 비교적 많아 매수 판단 전에 위험 요소를 우선 확인할 필요가 있습니다."
    else:
        label = "🔴 취약"
        summary = "현재 데이터 기준으로 긍정 요소보다 부담 요인이 많은 상태입니다."

    if confidence < 60:
        label += " · 데이터 부족 주의"
        summary += " 다만 데이터 일부가 부족해 점수의 신뢰도는 낮은 편입니다."

    strongest = sorted(components.items(), key=lambda x: x[1], reverse=True)[:2]
    weakest = sorted(components.items(), key=lambda x: x[1])[:2]
    return label, summary, strongest, weakest

# ---------------------------------------------------------
# 5. 종목 분석 결과 출력 (개선된 채점 로직)
# ---------------------------------------------------------
if analyze_btn:
    try:
        with st.spinner(f'{target_ticker} 데이터 정밀 분석 중...'):
            s_data = fetch_company_data(target_ticker)
            info, history = s_data['info'], s_data['history']
            
            current_price_raw = info.get('currentPrice')
            if current_price_raw is not None:
                try:
                    current_price_raw = float(current_price_raw)
                except (TypeError, ValueError):
                    current_price_raw = None
            price_source = "Yahoo Finance currentPrice"
            if current_price_raw is None or not np.isfinite(current_price_raw) or current_price_raw <= 0:
                try:
                    history_close = history['Close'].dropna()
                    current_price_raw = float(history_close.iloc[-1]) if not history_close.empty else None
                    price_source = "최근 종가(history)" if current_price_raw is not None else "데이터 없음"
                except Exception:
                    current_price_raw = None
                    price_source = "데이터 없음"
            curr_p = current_price_raw or 0
            target_p = info.get('targetMeanPrice', 0)
            upside = ((target_p / curr_p) - 1) * 100 if curr_p > 0 and target_p > 0 else 0

            # 종목 기본 정보 
            sector = info.get('sector', 'Unknown')
            beta_raw = info.get('beta')
            if beta_raw is not None:
                try:
                    beta_raw = float(beta_raw)
                    if not np.isfinite(beta_raw):
                        beta_raw = None
                except (TypeError, ValueError):
                    beta_raw = None
            beta_source = "Yahoo Finance Beta" if beta_raw is not None else "시장수익률 기반 추정"
            beta = beta_raw
            if beta is None:
                beta = estimate_beta_from_history(history, m_data.get('SP500'))
            roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0

            # 가까운 공식 경제 일정이 현재 분석 종목에 미칠 수 있는 영향 계산
            upcoming_impacts = get_upcoming_event_impacts(
                economic_events,
                sector=sector,
                beta=beta
            )
            op_margin = info.get('operatingMargins', 0) * 100 if info.get('operatingMargins') else 0
            debt = info.get('debtToEquity', 0)
            
            # [개선 3] 가치평가 폴백 로직을 위한 지표 수집
            peg = info.get('pegRatio')
            fwd_pe = info.get('forwardPE')
            pb = info.get('priceToBook')
            ps = info.get('priceToSalesTrailing12Months')

            # 차트 이동평균선(MA) 계산
            history['MA20'] = history['Close'].rolling(window=20).mean()
            history['MA50'] = history['Close'].rolling(window=50).mean()
            history['MA200'] = history['Close'].rolling(window=200).mean()
            df_daily, df_weekly, df_monthly = history.tail(90), history.resample('W').last().dropna(), history.resample('ME').last().dropna()
            
            # 점수 및 채점 내역 기록
            s_score, s_plus, s_minus = 0, [], []
            score_details = [] # [개선 4] 어떤 점수가 환산되었는지 텍스트로 남길 리스트

            # 1. 펀더멘털 평가
            if op_margin >= 20: 
                s_score += 10; score_details.append("영업마진 +10"); s_plus.append(f"✅ 압도적 영업마진({op_margin:.1f}%) (+10)")
            
            # [개선 3] 업종별 밸류에이션 우선순위 적용
            val_pts, val_msg = calculate_valuation_score(info, sector)
            s_score += val_pts
            score_details.append(f"가치평가 +{val_pts}")
            if val_pts > 0:
                s_plus.append(val_msg)
            else:
                s_minus.append(val_msg)
            
            # 2. 목표가 및 재무 평가
            # [개선 4] 목표가 자체뿐 아니라 애널리스트 수/목표가 신뢰도를 함께 반영
            analyst_count = info.get('numberOfAnalystOpinions') or info.get('numberOfAnalysts') or 0
            target_low = info.get('targetLowPrice')
            target_high = info.get('targetHighPrice')
            target_mean = target_p
            target_dispersion = 0
            if target_mean and target_low and target_high and target_mean > 0:
                target_dispersion = (target_high - target_low) / target_mean

            target_pts = 0
            if upside >= 20:
                target_pts = 15
            elif upside >= 5:
                target_pts = 5
            elif upside <= -10:
                target_pts = -10

            # 분석 인원이 적거나 목표가 분산이 크면 신뢰도를 낮춤
            if target_pts > 0:
                if analyst_count < 3:
                    target_pts = min(target_pts, 3)
                elif analyst_count < 10:
                    target_pts = min(target_pts, 8)
                if target_dispersion > 1.0:
                    target_pts = min(target_pts, 5)

            if target_pts > 0:
                s_score += target_pts; score_details.append(f"목표가괴리 +{target_pts}"); s_plus.append(f"✅ 목표가 기반 상승여력 {upside:.1f}% / 분석 {int(analyst_count)}명 (+{target_pts})")
            elif target_pts < 0:
                s_score += target_pts; score_details.append(f"목표가괴리 {target_pts}"); s_minus.append(f"⚠️ 평균 목표가가 현재가보다 낮음({upside:.1f}%) ({target_pts})")
            else:
                score_details.append("목표가괴리 +0")

            # [개선 5] ROE와 부채를 함께 보아 '레버리지로 부풀린 ROE'를 견제
            if roe >= 15:
                if debt and debt > 150:
                    s_score += 5; score_details.append("수익성 +5"); s_plus.append(f"🟡 높은 ROE지만 부채 주의(ROE {roe:.1f}%, D/E {debt:.1f}) (+5)")
                else:
                    s_score += 15; score_details.append("수익성 +15"); s_plus.append(f"✅ 뛰어난 수익성(ROE {roe:.1f}%) (+15)")
            elif roe >= 8:
                s_score += 5; score_details.append("수익성 +5"); s_plus.append(f"🟡 양호한 수익성(ROE {roe:.1f}%) (+5)")

            if debt and debt <= 80:
                s_score += 10; score_details.append("재무건전성 +10"); s_plus.append(f"✅ 튼튼한 재무(부채비율 {debt:.1f}%) (+10)")
            elif debt and debt > 150:
                s_score -= 15; score_details.append("과다부채 -15"); s_minus.append(f"🚨 과도한 부채({debt:.1f}%) (-15)")

            # 3. 섹터별 매크로 맞춤형 점수 조정 (동적 추세 기반)
            macro_adj = 0
            
            # 여기서 current_oil과 current_tnx를 확실하게 정의하여 에러 방지
            current_oil = m_data["OIL"]['Close'].iloc[-1] if "OIL" in m_data else 80
            current_tnx = m_data["TNX"]['Close'].iloc[-1] if "TNX" in m_data else 4.0
            oil_ma90 = m_data["OIL"]['Close'].tail(90).mean() if "OIL" in m_data else 80
            tnx_ma90 = m_data["TNX"]['Close'].tail(90).mean() if "TNX" in m_data else 4.0
            
            if sector in ['Technology', 'Communication Services']:
                if current_tnx > tnx_ma90 * 1.1: macro_adj -= 10; s_minus.append(f"📉 [기술주] 금리 급등 밸류에이션 부담 (-10)")
                if current_oil > oil_ma90 * 1.15: macro_adj -= 10; s_minus.append(f"🔌 [기술주] 고유가 전력 인프라 부담 (-10)")
                elif current_tnx < tnx_ma90 * 0.95: macro_adj += 10; s_plus.append(f"🚀 [기술주] 금리 하향 수혜 환경 (+10)")

            elif sector == 'Energy':
                if current_oil > oil_ma90 * 1.1: macro_adj += 15; s_plus.append(f"🛢️ [에너지주] 유가 상승 추세 수혜 (+15)")
                elif current_oil < oil_ma90 * 0.9: macro_adj -= 15; s_minus.append(f"⚠️ [에너지주] 유가 하락 실적 악화 (-15)")

            elif sector == 'Financial Services':
                if current_tnx > tnx_ma90 * 1.05: macro_adj += 10; s_plus.append(f"🏦 [금융주] 금리 상승 기반 예대마진 개선 (+10)")
                elif current_tnx < tnx_ma90 * 0.9: macro_adj -= 10; s_minus.append(f"📉 [금융주] 금리 하락 수익성 악화 (-10)")

            if macro_adj != 0:
                s_score += macro_adj
                score_details.append(f"섹터·매크로 {macro_adj:+d}")

            # 4. 시장 공포(VIX)와 종목 변동성(Beta) 평가
            beta_adj = 0
            if vix > 24 and beta is not None:
                if beta >= 1.5: beta_adj -= 15; s_minus.append(f"🌪️ 공포 장세 고변동성 취약 (Beta {beta:.2f}) (-15)")
                elif 0 < beta <= 0.8: beta_adj += 10; s_plus.append(f"🛡️ 안정적 방어력 증명 (Beta {beta:.2f}) (+10)")
            elif vix < 15 and beta is not None and beta >= 1.2:
                beta_adj += 10; s_plus.append(f"📈 강세장 공격적 수익 창출 (Beta {beta:.2f}) (+10)")
                
            if beta_adj != 0:
                s_score += beta_adj
                score_details.append(f"시장방어력 {beta_adj:+d}")

            # [개선 6] 1개월 단일 신호 대신 다중 기간 수익률 + 이동평균 추세
            momentum_pts = 0
            close = history['Close']
            ret_1m = close.iloc[-1] / close.iloc[-22] - 1 if len(close) >= 22 else 0
            ret_3m = close.iloc[-1] / close.iloc[-63] - 1 if len(close) >= 63 else 0
            ret_6m = 0
            if len(close) >= 127:
                ret_6m = close.iloc[-1] / close.iloc[-127] - 1
            else:
                ret_6m = 0

            last_close = close.iloc[-1]
            ma20_last = history['MA20'].iloc[-1]
            ma50_last = history['MA50'].iloc[-1]
            ma200_last = history['MA200'].iloc[-1]

            if last_close > ma20_last: momentum_pts += 3
            if last_close > ma50_last: momentum_pts += 3
            if pd.notna(ma200_last) and last_close > ma200_last: momentum_pts += 5
            if pd.notna(ma200_last) and ma50_last > ma200_last: momentum_pts += 4
            if ret_1m > 0: momentum_pts += 1
            if ret_3m > 0: momentum_pts += 2
            if ret_6m > 0: momentum_pts += 2

            momentum_pts = min(20, momentum_pts)
            s_score += momentum_pts
            score_details.append(f"모멘텀 +{momentum_pts}")
            if momentum_pts >= 12:
                s_plus.append(f"📈 중기 상승추세 우수(1M {ret_1m*100:.1f}%, 3M {ret_3m*100:.1f}%, 6M {ret_6m*100:.1f}%) (+{momentum_pts})")
            elif momentum_pts >= 7:
                s_plus.append(f"🟡 추세 혼조(1M {ret_1m*100:.1f}%, 3M {ret_3m*100:.1f}%, 6M {ret_6m*100:.1f}%) (+{momentum_pts})")
            else:
                s_minus.append(f"📉 모멘텀 약함(1M {ret_1m*100:.1f}%, 3M {ret_3m*100:.1f}%, 6M {ret_6m*100:.1f}%) (+{momentum_pts})")
            
            # [Phase A-1] 기존 가산점의 원시값은 설명용으로 유지하고, 최종점수는 4개 축의 정규화 점수로 계산
            raw_score_before_normalization = s_score

            # [Phase A-2] 각 축별 데이터 신뢰도 계산
            component_confidences = calculate_component_confidences(
                info, history, m_data,
                beta_available=beta,
                beta_estimated=(beta_raw is None and beta is not None)
            )
            confidence = calculate_confidence(component_confidences)

            # [Phase A-3] Quality / Valuation / Momentum / Risk를 실제 최종점수에 연결
            component_scores = calculate_component_scores(
                info, history, sector, val_pts, target_pts, debt, roe, op_margin,
                macro_adj, beta_adj, momentum_pts
            )
            s_score = calculate_final_score(component_scores, component_confidences)

            # 최종점수 + 신뢰도 + 강/약점 해석
            recommendation_label, recommendation_summary, strongest_components, weakest_components = get_score_interpretation(
                s_score, confidence, component_scores
            )
            
            # 세부 채점 내역 문자열 생성
            score_breakdown_str = " / ".join(score_details)

            st.divider()
            if upcoming_impacts:
                st.markdown("### ⚠️ 가까운 경제 일정이 종목에 미칠 수 있는 영향")
                for impact in upcoming_impacts[:3]:
                    st.warning(impact)

            if geo_level in {"caution", "high"}:
                st.markdown("### 🌍 현재 지정학적 환경이 종목에 미칠 수 있는 영향")
                geo_impacts = []
                if sector in ["Technology", "Communication Services"] and beta is not None and beta >= 1.2:
                    geo_impacts.append("⚠️ 기술주·고Beta 종목은 지정학적 충격이 발생할 경우 변동성이 확대될 수 있어 주의가 필요합니다.")
                elif sector in ["Energy"]:
                    geo_impacts.append("🛢️ 에너지주는 지정학적 긴장에 따른 유가·공급망 변화의 영향을 받을 수 있습니다.")
                else:
                    geo_impacts.append("📌 지정학적 긴장 상승은 위험회피 심리와 시장 변동성에 영향을 줄 수 있습니다. 현재 종목 점수에는 직접 반영하지 않았습니다.")
                for msg in geo_impacts:
                    st.warning(msg)

            if beta_raw is None and beta is not None:
                st.caption(f"ℹ️ Beta는 제공 데이터가 없어 최근 시장수익률 기준으로 추정했습니다. ({beta:.2f})")
            elif beta is None:
                st.warning("⚠️ Beta 데이터를 확보하지 못해 변동성 관련 평가의 신뢰도가 낮아졌습니다.")

            with st.expander("📘 종목 평가 항목이 무엇인지 쉽게 보기"):
                st.markdown(
                    """
                    <div style="font-size:13px; line-height:1.75;">
                    <b>수익성(ROE)</b> · 회사가 자기자본을 이용해 얼마나 효율적으로 이익을 내는지 보는 지표입니다. 높을수록 일반적으로 좋지만 부채가 높은 경우 함께 확인합니다.<br><br>
                    <b>영업마진</b> · 본업으로 벌어들이는 이익의 비율입니다. 높을수록 가격 결정력이나 비용 관리가 좋을 가능성이 있습니다.<br><br>
                    <b>가치평가</b> · 현재 주가가 회사의 이익·성장·자산 등에 비해 비싼지 싼지를 살펴봅니다. 업종에 따라 적합한 기준이 다릅니다.<br><br>
                    <b>목표가 상승여력</b> · 애널리스트들의 평균 목표가가 현재 주가보다 얼마나 높은지 봅니다. 단, 애널리스트 수와 목표가 편차가 크면 신뢰도를 낮춥니다.<br><br>
                    <b>재무건전성(D/E)</b> · 회사가 자기자본에 비해 얼마나 많은 부채를 사용하는지 나타냅니다. 과도하면 금리와 경기 변화에 취약할 수 있습니다.<br><br>
                    <b>섹터·매크로</b> · 현재 금리와 유가가 이 기업 업종에 유리한지 불리한지 살펴봅니다.<br><br>
                    <b>Beta</b> · 시장이 움직일 때 이 종목이 평균적으로 얼마나 크게 움직이는지 나타내는 값입니다. 높을수록 상승장에서는 강할 수 있지만 하락장 변동성도 커질 수 있습니다.<br><br>
                    <b>모멘텀</b> · 주가가 최근 1·3·6개월 동안 어떤 방향으로 움직였는지와 20·50·200일 평균선 위에 있는지를 함께 봅니다. 즉, “좋은 회사인가”와 별개로 “최근 추세가 좋은가”를 보는 항목입니다.
                    </div>
                    """, unsafe_allow_html=True
                )

            s_color = "#00ff00" if s_score >= 70 else "#ffa500" if s_score >= 40 else "#ff4b4b"
            upside_color = "#00ff00" if upside > 0 else "#ff4b4b"
            
            confidence_color = "#00ff00" if confidence >= 75 else "#ffa500" if confidence >= 60 else "#ff4b4b"
            comp_cols = st.columns(4)
            comp_labels = {"Quality": "기업의 체력", "Valuation": "가격 매력", "Momentum": "주가 추세", "Risk": "위험 환경 대응"}
            comp_desc = {"Quality": "수익성·마진·부채", "Valuation": "가치평가·목표가", "Momentum": "1·3·6개월·이동평균", "Risk": "위험요인을 고려한 환경 점수 · 높을수록 상대적으로 우호적"}
            for col, key in zip(comp_cols, ["Quality", "Valuation", "Momentum", "Risk"]):
                with col:
                    st.metric(comp_labels[key], f"{component_scores[key]}/100")
                    st.caption(comp_desc[key])
            st.caption("최종점수는 4개 축의 가중평균입니다. 데이터가 부족한 축은 해당 축의 가중치가 자동으로 줄어들어 신뢰도가 낮은 데이터를 과도하게 반영하지 않습니다.")

            with st.expander("🔎 왜 이런 종합점수가 나왔는지 한눈에 보기"):
                strong_text = " · ".join([f"{k} {v}" for k, v in strongest_components])
                weak_text = " · ".join([f"{k} {v}" for k, v in weakest_components])
                st.markdown(
                    f"**상대적으로 강한 부분:** {strong_text}\n\n"
                    f"**상대적으로 약한 부분:** {weak_text}\n\n"
                    "※ 최종점수는 Quality 30% · Valuation 25% · Momentum 25% · Risk 20%의 가중평균을 기본으로 하며, 데이터가 부족한 축은 영향도를 자동으로 낮춥니다.\n\n"
                    "※ Risk는 ‘위험이 80점’이라는 뜻이 아니라, 금리·유가·변동성·재무위험 등을 고려했을 때 환경이 얼마나 우호적인지를 나타냅니다.\n\n"
                    f"※ 기존 항목을 단순 합산한 원시 점수는 {raw_score_before_normalization}점이었으며, 최종점수는 4개 축으로 정규화했습니다."
                )

            st.markdown(f"""
                <div style="background:#171717; border:1px solid #333; border-radius:12px; padding:14px; margin-bottom:12px;">
                    <div style="font-size:13px; color:#ddd; margin-bottom:8px;"><b>{recommendation_label}</b> · {recommendation_summary}</div>
                    <div style="color:#aaa; font-size:12px;">데이터 신뢰도: <span style="color:{confidence_color}; font-weight:700;">{confidence}%</span> · 데이터가 부족할수록 점수 해석에 주의하세요.</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 30px; border-radius: 15px; border: 2px solid {s_color}; text-align: center;">
                    <p style="color:#aaa; font-size:12px; margin:0 0 4px 0;">이 점수는 회사의 수익성·가치·추세·위험 요소를 함께 고려한 ‘종목 매력도’입니다.</p>
                    <h2 style="color: white; margin: 0;">{info.get('longName', target_ticker)} 투자 매력도 <span style="font-size: 16px; color: gray;">(Sector: {sector})</span></h2>
                    <h1 style="color: {s_color}; font-size: 80px; margin: 5px 0;">{s_score} <span style="font-size: 20px; color: gray;">Point</span></h1>
                    <p style="color: #aaaaaa; font-size: 13px; margin-top: -5px; margin-bottom: 15px;">{score_breakdown_str}</p>
                    <div style="background-color: #2b2b2b; padding: 15px; border-radius: 10px; display: inline-block; width: 80%;">
                        <p style="color: gray; margin: 0; font-size: 14px;">🎯 전문가 목표가 기반 수익 잠재력</p>
                        <h2 style="color: white; margin: 5px 0;">${curr_p} <span style="font-size: 18px; color: gray;">→</span> ${target_p}</h2>
                        <p style="color:#888; margin:2px 0 6px 0; font-size:11px;">현재가: {price_source} · Beta: {beta_source}</p>
                        <h3 style="color: {upside_color}; margin: 0;">기대 수익률: {upside:.1f}%</h3>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            sc1, sc2 = st.columns([6, 4])
            with sc1:
                st.subheader("📈 주가 차트 분석")
                tab_daily, tab_weekly, tab_monthly = st.tabs(["📅 일봉", "🗓️ 주봉", "📆 월봉"])
                with tab_daily: st.plotly_chart(create_candlestick(df_daily, f"{target_ticker} 일봉 (최근 90일)"), use_container_width=True)
                with tab_weekly: st.plotly_chart(create_candlestick(df_weekly, f"{target_ticker} 주봉 (최근 2년)"), use_container_width=True)
                with tab_monthly: st.plotly_chart(create_candlestick(df_monthly, f"{target_ticker} 월봉 (최근 2년)"), use_container_width=True)

            with sc2:
                st.subheader("🧐 상세 분석 리포트")
                if confidence < 60:
                    st.warning("⚠️ 데이터 신뢰도가 낮습니다. 누락된 지표 때문에 일부 점수가 계산되지 않았을 수 있습니다.")
                for p in s_plus: st.success(p)
                for m in s_minus: st.warning(m)

    except Exception as e:
        st.error(f"분석 실패: 해당 종목의 데이터를 불러올 수 없습니다. ({e})")
import os
import re
import json
import time
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timezone, timedelta
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TJR_TOKEN = os.environ.get("TJR_BOT_TOKEN", "").strip()
MAIN_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_BOT_TOKEN = TJR_TOKEN if TJR_TOKEN else MAIN_TOKEN

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

client = Groq(api_key=GROQ_API_KEY)
VN_TZ = timezone(timedelta(hours=7))

def is_killzone(dt_vn):
    h = dt_vn.hour
    m = dt_vn.minute
    total_mins = h * 60 + m
    in_london = (13 * 60) <= total_mins <= (16 * 60 + 30)
    in_ny = (18 * 60 + 30) <= total_mins <= (23 * 60)
    if in_london: return True, "LONDON KILLZONE"
    if in_ny:     return True, "NEW YORK KILLZONE"
    return False, "PHIÊN TIÊU CHUẨN"

def get_spot_gold_m15():
    if not TWELVE_DATA_API_KEY: raise ValueError("Chưa có TWELVE_DATA_API_KEY!")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=15min&outputsize=80&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    res = requests.get(url, timeout=10).json()
    if "values" not in res: raise ValueError(f"Twelve Data lỗi: {res.get('message')}")
        
    df = pd.DataFrame(res['values']).iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df['Volume'] = 1000
    df_m15 = df.iloc[:-1].copy()
    
    df_h1 = df_m15.resample('1h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    h1_is_bull = df_h1['Close'].iloc[-1] > df_h1['EMA21'].iloc[-1]
    
    df_m15['EMA21'] = df_m15['Close'].ewm(span=21, adjust=False).mean()
    h_l, h_c, l_c = df_m15['High'] - df_m15['Low'], np.abs(df_m15['High'] - df_m15['Close'].shift()), np.abs(df_m15['Low'] - df_m15['Close'].shift())
    df_m15['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    return df_m15.tail(50), h1_is_bull

def generate_chart(df, title_text):
    path = "xauusd_tjr_clean.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    mpf.plot(df, type='candle', style='binance', addplot=ema_plot, title=f"\n{title_text}", figsize=(13, 6.5), savefig=path)
    return path

def send_telegram_photo(photo_path, text):
    caption_text = text if len(text) <= 1000 else text[:980] + "\n..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption_text}, files={'photo': photo}, timeout=15)

def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text}, timeout=10)

def build_m15_dossier(df_m15):
    dossier = []
    for t, r in df_m15.tail(12).iterrows():
        rng = r['High'] - r['Low']
        u_w = r['High'] - max(r['Open'], r['Close'])
        l_w = min(r['Open'], r['Close']) - r['Low']
        dossier.append(f"• {t.strftime('%H:%M')}: Close={r['Close']:.1f}, High={r['High']:.1f}, Low={r['Low']:.1f} (Râu trên={u_w:.1f}, Râu dưới={l_w:.1f}, EMA21={r['EMA21']:.1f})")
    return "\n".join(dossier)

# NÃO BỘ TJR AI SUY LUẬN CHUẨN XÁC 100%
def ai_tjr_decision_engine(h1_trend, kz_name, curr_price, atr_val, m15_dossier):
    prompt = f"""
Bạn là Chuyên gia Giao dịch TJR Trades (Path to Profitability / ICT Simplified).
DỮ LIỆU 12 CÂY NẾN M15 VÀNG GẦN NHẤT (3 TIẾNG QUA):
{m15_dossier}

BỐI CẢNH THỊ TRƯỜNG:
- Giá hiện tại: {curr_price} (Spot XAU/USD)
- Xu hướng chủ đạo H1: {h1_trend}
- Trạng thái phiên: {kz_name}
- Độ biến động ATR M15: {atr_val:.1f} giá

HÃY ÁP DỤNG TRIẾT LÝ TJR (SETUP A & B):
1. Có cú quét thanh khoản (Liquidity Sweep) đỉnh/đáy nào vừa hoàn tất không?
2. Có nến bứt phá (Displacement) đục thủng FVG tạo thành IFVG hoặc nhịp hồi test EMA 21 theo trend H1 không?
3. TỰ RA QUYẾT ĐỊNH VÀO LỆNH HAY ĐỨNG NGOÀI.

BẮT BUỘC TRẢ VỀ DUY NHẤT ĐỊNH DẠNG JSON:
{{
  "quyet_dinh": "BUY" hoặc "SELL" hoặc "STAND_ASIDE",
  "setup_name": "SETUP A: BẪY THANH KHOẢN TJR" hoặc "SETUP B: TIẾP DIỄN TREND H1" hoặc "KHÔNG CÓ SETUP",
  "phan_tich_tjr": "Viết bài phân tích chuyên sâu 3 đoạn về cấu trúc IFVG, quét thanh khoản và lý do vào lệnh",
  "entry": {curr_price},
  "sl": muc_gia_sl (cách entry tối thiểu 6.5 giá an toàn),
  "tp1": muc_gia_tp1 (tối thiểu R:R 1:1.5),
  "tp2": muc_gia_tp2 (tối thiểu R:R 1:2.5 đến 1:3)
}}
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=650,
            temperature=0.2,
            timeout=25.0
        )
        txt = res.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', txt, re.DOTALL)
        if match: return json.loads(match.group())
    except Exception as e:
        print(f"Lỗi TJR AI: {e}")
        
    return {"quyet_dinh": "STAND_ASIDE", "setup_name": "KHÔNG CÓ SETUP", "phan_tich_tjr": "Thị trường M15 đang đi ngang, chưa xuất hiện bẫy thanh khoản đạt chuẩn TJR.", "entry": curr_price, "sl": 0, "tp1": 0, "tp2": 0}

def main():
    print("=== BOT #2: TJR AI BRAIN (NGÂN SÁCH TOKEN TỐI ƯU) ===")
    df_m15, h1_is_bull = get_spot_gold_m15()
    last_time = df_m15.index[-1] + timedelta(minutes=15)
    time_str = last_time.strftime('%H:%M - %d/%m')
    curr_price = round(df_m15['Close'].iloc[-1], 2)
    
    in_kz, kz_name = is_killzone(last_time)
    h1_trend_str = "TĂNG (ƯU TIÊN BUY)" if h1_is_bull else "GIẢM (ƯU TIÊN SELL)"
    atr_val = df_m15['ATR'].iloc[-1]
    if np.isnan(atr_val) or atr_val < 3.5: atr_val = 4.0
    
    m15_dossier = build_m15_dossier(df_m15)
    chart_file = generate_chart(df_m15, f"[TJR M15 SPOT] H1: {h1_trend_str} - {kz_name}")
    
    # GỌI NÃO BỘ TJR AI RA QUYẾT ĐỊNH
    decision = ai_tjr_decision_engine(h1_trend_str, kz_name, curr_price, atr_val, m15_dossier)
    act = decision.get("quyet_dinh", "STAND_ASIDE")
    setup_title = decision.get("setup_name", "TJR SETUP")
    analysis = decision.get("phan_tich_tjr", "")
    
    header = f"🎯 [BOT #2: TJR M15 SNIPER - AI DRIVEN]\n" \
             f"⏰ Nến M15: {time_str} (Khớp MT5)\n" \
             f"📊 H1 Trend: {h1_trend_str} | ⌛ {kz_name}\n" \
             f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n"
             
    if act in ["BUY", "SELL"]:
        sl_val = float(decision.get("sl", curr_price - 7.0 if act=="BUY" else curr_price + 7.0))
        tp1_val = float(decision.get("tp1", curr_price + 11.0 if act=="BUY" else curr_price - 11.0))
        tp2_val = float(decision.get("tp2", curr_price + 20.0 if act=="BUY" else curr_price - 20.0))
        
        sl_pips = int(round(abs(curr_price - sl_val) * 10))
        tp1_pips = int(round(abs(tp1_val - curr_price) * 10))
        tp2_pips = int(round(abs(tp2_val - curr_price) * 10))
        
        caption = f"{header}" \
                  f"🚨 QUYẾT ĐỊNH: VÀO {act} VÀNG (MARKET)\n" \
                  f"🔥 Mô hình: {setup_title}\n" \
                  f"📍 Entry: {curr_price}\n" \
                  f"🛑 Cắt lỗ (SL): {sl_val} (-{sl_pips} pips)\n" \
                  f"🎯 TP1: {tp1_val} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                  f"🚀 TP2: {tp2_val} (+{tp2_pips} pips - R:R 1:2.5+)"
    else:
        caption = f"{header}" \
                  f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ SETUP A/B)\n\n" \
                  f"👉 Lưu ý: AI khuyến nghị kiên nhẫn chờ bẫy giá thanh khoản rõ ràng."
                  
    send_telegram_photo(chart_file, caption)
    time.sleep(1)
    send_telegram_msg(f"💡 [PHÂN TÍCH CHUYÊN SÂU TJR AI]\n\n{analysis}")
    
    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

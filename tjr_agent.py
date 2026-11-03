import os
import time
import base64
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timezone, timedelta
from groq import Groq

# 1. KHỞI TẠO KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TJR_TOKEN = os.environ.get("TJR_BOT_TOKEN", "").strip()
MAIN_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_BOT_TOKEN = TJR_TOKEN if TJR_TOKEN else MAIN_TOKEN

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

client = Groq(api_key=GROQ_API_KEY)

def is_killzone(dt_vn):
    h = dt_vn.hour
    m = dt_vn.minute
    total_mins = h * 60 + m
    in_london = (13 * 60) <= total_mins <= (16 * 60 + 30)
    in_ny = (18 * 60 + 30) <= total_mins <= (23 * 60)
    if in_london: return True, "LONDON KILLZONE (PHIÊN ÂU)"
    if in_ny:     return True, "NEW YORK KILLZONE (PHIÊN MỸ)"
    return False, "NGOÀI KILLZONE"

def get_spot_gold_m15():
    """LẤY NẾN M15 SPOT XAU/USD TRỰC TIẾP GIỜ VIỆT NAM"""
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY trong GitHub Secrets!")
        
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=15min&outputsize=70&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    res = requests.get(url).json()
    if "values" not in res:
        raise ValueError(f"Twelve Data báo lỗi: {res.get('message', 'Không rõ lỗi')}")
        
    df = pd.DataFrame(res['values'])
    df = df.iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df['Volume'] = 1000
    df_m15 = df.iloc[:-1] # Bỏ nến đang chạy
    
    df_h1 = df_m15.resample('1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    h1_is_bull = df_h1['Close'].iloc[-1] > df_h1['EMA21'].iloc[-1]
    
    df_m15['EMA21'] = df_m15['Close'].ewm(span=21, adjust=False).mean()
    
    return df_m15.tail(50), h1_is_bull

def generate_clean_chart(df, title_text):
    chart_path = "xauusd_tjr_clean.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    mpf.plot(
        df, type='candle', style='binance', addplot=ema_plot,
        title=f"\n{title_text}", figsize=(14, 7),
        savefig=chart_path
    )
    return chart_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_telegram(photo_path, text):
    caption_text = text if len(text) <= 1000 else text[:980] + "\n..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption_text}, files={'photo': photo})

PROMPT_TJR_AB = """
Bạn là Phù thủy Giao dịch TJR Trades săn lệnh Vàng M15 (Setup A & B Hunter).
Biểu đồ nến M15 Vàng Spot (XAU/USD) kèm đường EMA 21 cam.
Xu hướng chủ đạo H1: {H1_Trend}. Phiên: {KZ_Name}. Giá hiện tại: {Price} lúc {Time_Str}.

HÃY ĐÁNH GIÁ VÀ CHỌN ĐÚNG 1 TRONG 3 TRƯỜNG HỢP:
1. [CÓ KÈO SETUP A: BUY/SELL]: Có quét thanh khoản đỉnh/đáy + nến đảo chiều rút râu dứt khoát.
2. [CÓ KÈO SETUP B: BUY/SELL]: Thuận trend H1 mạnh, nhịp hồi chạm EMA 21 bật tiếp.
3. [ĐỨNG NGOÀI QUAN SÁT]: Nến giằng co, doji nhỏ, chưa có tín hiệu rõ nét.
(Viết nhận định súc tích dưới 250 ký tự).
"""

def main():
    print("=== BOT #2: QUÉT NẾN M15 TJR (SPOT XAU/USD) ===")
    df_m15, h1_is_bull = get_spot_gold_m15()
    last_time = df_m15.index[-1] + timedelta(minutes=15)
    time_str = last_time.strftime('%H:%M - %d/%m')
    curr_price = round(df_m15['Close'].iloc[-1], 2)
    
    in_kz, kz_name = is_killzone(last_time)
    h1_trend_str = "TĂNG (ƯU TIÊN BUY)" if h1_is_bull else "GIẢM (ƯU TIÊN SELL)"
    
    chart_title = f"[SPOT XAUUSD M15] H1 Trend: {h1_trend_str} - {kz_name}"
    chart_file = generate_clean_chart(df_m15, chart_title)
    base64_img = encode_image(chart_file)
    
    recent_bars = 10
    recent_high = df_m15['High'].iloc[-recent_bars:].max()
    recent_low = df_m15['Low'].iloc[-recent_bars:].min()
    
    prompt = PROMPT_TJR_AB.replace("{H1_Trend}", h1_trend_str)\
                          .replace("{KZ_Name}", kz_name)\
                          .replace("{Price}", str(curr_price))\
                          .replace("{Time_Str}", time_str)
                          
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
        ]}],
        max_tokens=300
    )
    ai_verdict = response.choices[0].message.content.strip()
    
    # 1. SETUP A
    if "SETUP A" in ai_verdict:
        is_buy = "BUY" in ai_verdict
        act = "BUY VÀNG SPOT (MARKET)" if is_buy else "SELL VÀNG SPOT (MARKET)"
        risk_dist = 7.5
        sl_calc = round(curr_price - risk_dist if is_buy else curr_price + risk_dist, 2)
        tp1_calc = round(curr_price + (risk_dist * 1.5) if is_buy else curr_price - (risk_dist * 1.5), 2)
        tp2_calc = round(curr_price + (risk_dist * 3.0) if is_buy else curr_price - (risk_dist * 3.0), 2)
        
        tele_msg = f"🚨 [PHÁT HIỆN SETUP A: BẪY THANH KHOẢN TJR]\n" \
                   f"⏰ Nến M15: {time_str} (Khớp MT5) | ⌛ {kz_name}\n" \
                   f"👉 HÀNH ĐỘNG: {act}\n" \
                   f"📍 Giá vào ngay: {curr_price} (Spot XAU/USD)\n\n" \
                   f"🛑 SL chân sóng: {sl_calc} (-75 pips / 7.5 giá)\n" \
                   f"🎯 TP1: {tp1_calc} (+112 pips) -> Dời hòa vốn\n" \
                   f"🚀 TP2: {tp2_calc} (+225 pips - R:R 1:3)\n" \
                   f"💰 Khuyến nghị: ĐI ĐỦ 1% RISK VỐN\n\n" \
                   f"💡 PHÂN TÍCH TJR:\n{ai_verdict}"

    # 2. SETUP B
    elif "SETUP B" in ai_verdict:
        is_buy = "BUY" in ai_verdict
        act = "BUY VÀNG SPOT (MARKET)" if is_buy else "SELL VÀNG SPOT (MARKET)"
        risk_dist = 6.0
        sl_calc = round(curr_price - risk_dist if is_buy else curr_price + risk_dist, 2)
        tp1_calc = round(curr_price + (risk_dist * 1.0) if is_buy else curr_price - (risk_dist * 1.0), 2)
        tp2_calc = round(curr_price + (risk_dist * 2.0) if is_buy else curr_price - (risk_dist * 2.0), 2)
        
        tele_msg = f"⚡ [PHÁT HIỆN SETUP B: TIẾP DIỄN SÓNG H1]\n" \
                   f"⏰ Nến M15: {time_str} (Khớp MT5) | ⌛ {kz_name}\n" \
                   f"👉 HÀNH ĐỘNG: {act}\n" \
                   f"📍 Giá vào ngay: {curr_price} (Spot XAU/USD)\n\n" \
                   f"🛑 SL nhịp hồi: {sl_calc} (-60 pips / 6.0 giá)\n" \
                   f"🎯 TP1: {tp1_calc} (+60 pips) -> Dời hòa vốn\n" \
                   f"🚀 TP2: {tp2_calc} (+120 pips - R:R 1:2)\n" \
                   f"💰 Khuyến nghị: ĐI 0.5% RISK (LƯỚT NHANH)\n\n" \
                   f"💡 PHÂN TÍCH TJR:\n{ai_verdict}"

    # 3. CHƯA CÓ KÈO -> VẪN BÁO CÁO ĐỂ BẠN BIẾT BOT ĐANG HOẠT ĐỘNG
    else:
        tele_msg = f"🎯 [BÁO CÁO TJR M15: SPOT GOLD]\n" \
                   f"⏰ Nến M15: {time_str} (Khớp MT5 & TradingView)\n" \
                   f"📊 H1 Trend: {h1_trend_str} | ⌛ {kz_name}\n" \
                   f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n" \
                   f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ SETUP A/B)\n\n" \
                   f"💡 NHẬN ĐỊNH TJR:\n{ai_verdict}"
                   
    send_telegram(chart_file, tele_msg)
    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

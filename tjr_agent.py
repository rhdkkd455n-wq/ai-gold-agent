import os
import time
import base64
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
from datetime import datetime, timezone, timedelta
from groq import Groq

# 1. KHỞI TẠO KEY BẢO MẬT
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

TICKER = "XAUUSD=X" # Vàng Spot khớp MT5
client = Groq(api_key=GROQ_API_KEY)
VN_TZ = timezone(timedelta(hours=7))

def is_killzone(dt_vn):
    """Kiểm tra Khung Giờ Vàng (Killzone) TJR"""
    h = dt_vn.hour
    m = dt_vn.minute
    total_mins = h * 60 + m
    
    in_london = (13 * 60) <= total_mins <= (16 * 60 + 30)
    in_ny = (18 * 60 + 30) <= total_mins <= (23 * 60)
    
    if in_london: return True, "LONDON KILLZONE (PHIÊN ÂU)"
    if in_ny:     return True, "NEW YORK KILLZONE (PHIÊN MỸ)"
    return False, "NGOÀI KILLZONE"

def get_h1_and_m15_data():
    gold = yf.Ticker(TICKER)
    df_m15 = gold.history(period="7d", interval="15m")
    df_m15 = df_m15[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df_m15.index = df_m15.index.tz_convert(VN_TZ)
    df_m15 = df_m15.iloc[:-1] # Bỏ nến đang chạy
    
    # Gộp nến H1 để làm Trend Chỉ đường
    df_h1 = df_m15.resample('1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    h1_is_bull = df_h1['Close'].iloc[-1] > df_h1['EMA21'].iloc[-1]
    
    # Tính ATR 14 cho M15
    h_l = df_m15['High'] - df_m15['Low']
    h_c = np.abs(df_m15['High'] - df_m15['Close'].shift())
    l_c = np.abs(df_m15['Low'] - df_m15['Close'].shift())
    df_m15['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    return df_m15.tail(60), h1_is_bull

def find_ifvg_levels(df):
    """THUẬT TOÁN ĐO KHỐI INVERSE FVG TRÊN M15"""
    ifvg_list = []
    
    for i in range(2, len(df) - 3):
        c1_h, c1_l = df['High'].iloc[i-2], df['Low'].iloc[i-2]
        c3_h, c3_l = df['High'].iloc[i], df['Low'].iloc[i]
        
        # FVG Giảm bị đục thủng đi lên -> Thành Hỗ trợ (Bullish IFVG)
        if c3_h < c1_l:
            top_box, bottom_box = c1_l, c3_h
            for j in range(i+1, len(df)):
                if df['Close'].iloc[j] > top_box:
                    ifvg_list.append({
                        'type': 'BULLISH',
                        'idx_start': i-2,
                        'idx_end': len(df)-1,
                        'top': top_box,
                        'bottom': bottom_box,
                        'color': 'cyan'
                    })
                    break
                    
        # FVG Tăng bị đục thủng sập xuống -> Thành Kháng cự (Bearish IFVG)
        elif c3_l > c1_h:
            top_box, bottom_box = c3_l, c1_h
            for j in range(i+1, len(df)):
                if df['Close'].iloc[j] < bottom_box:
                    ifvg_list.append({
                        'type': 'BEARISH',
                        'idx_start': i-2,
                        'idx_end': len(df)-1,
                        'top': top_box,
                        'bottom': bottom_box,
                        'color': 'magenta'
                    })
                    break

    return ifvg_list[-3:]

def generate_tjr_chart(df, ifvg_list, title_text):
    chart_path = "xauusd_tjr_chart.png"
    
    fig, axlist = mpf.plot(
        df, type='candle', style='binance',
        title=f"\n{title_text}", figsize=(14, 7), returnfig=True
    )
    ax = axlist[0]
    
    for box_data in ifvg_list:
        w = box_data['idx_end'] - box_data['idx_start'] + 1
        h = box_data['top'] - box_data['bottom']
        rect = patches.Rectangle(
            (box_data['idx_start'], box_data['bottom']), w, h,
            edgecolor=box_data['color'], facecolor=box_data['color'], alpha=0.35
        )
        ax.add_patch(rect)
        
    fig.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return chart_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_telegram_photo(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})

PROMPT_TJR = """
Bạn là Chuyên gia Giao dịch Hệ thống TJR Trades (H1 Trend + M15 IFVG Trigger).
Biểu đồ đính kèm là nến M15 của Vàng (Hộp Cyan = Bullish IFVG Hỗ trợ, Hộp Magenta = Bearish IFVG Kháng cự).
Xu hướng chủ đạo khung H1: {H1_Trend}.
Phiên giao dịch: {KZ_Name}.
Giá hiện tại: {Price} lúc {Time_Str}.

HÃY ĐÁNH GIÁ CHUẨN XÁC THEO TJR:
1. Khung H1 đang chỉ hướng gì? Giá M15 có đang retest khối IFVG đúng hướng H1 không?
2. Có tín hiệu quét thanh khoản (Liquidity Sweep đỉnh/đáy) lấy đà chưa?
3. QUYẾT ĐỊNH: [VÀO LỆNH BUY TJR] / [VÀO LỆNH SELL TJR] HOẶC [ĐỨNG NGOÀI (CHỜ SETUP)].
- Nhận định súc tích (Dưới 300 ký tự).
"""

def main():
    print("=== BẮT ĐẦU QUÉT BOT #2: H1 TREND + M15 IFVG TJR ===")
    df_m15, h1_is_bull = get_h1_and_m15_data()
    last_time = df_m15.index[-1] + timedelta(minutes=15)
    time_str = last_time.strftime('%H:%M - %d/%m')
    curr_price = round(df_m15['Close'].iloc[-1], 2)
    
    in_kz, kz_name = is_killzone(last_time)
    h1_trend_str = "TĂNG (ƯU TIÊN BUY)" if h1_is_bull else "GIẢM (ƯU TIÊN SELL)"
    
    ifvg_levels = find_ifvg_levels(df_m15)
    chart_title = f"[TJR M15] H1 Trend: {h1_trend_str} - {kz_name}"
    chart_file = generate_tjr_chart(df_m15, ifvg_levels, chart_title)
    base64_img = encode_image(chart_file)
    
    # TÍNH SL AN TOÀN THEO CHÂN SÓNG QUÉT (6.5 - 8.5 GIÁ)
    recent_bars = 10
    recent_high = df_m15['High'].iloc[-recent_bars:].max()
    recent_low = df_m15['Low'].iloc[-recent_bars:].min()
    
    if h1_is_bull:
        act = "BUY VÀNG (MARKET)"
        sl_calc = round(recent_low - 1.0, 2)
        risk_dist = round(curr_price - sl_calc, 1)
        if risk_dist < 6.5: 
            sl_calc = round(curr_price - 6.5, 2)
            risk_dist = 6.5
        elif risk_dist > 8.5:
            sl_calc = round(curr_price - 8.5, 2)
            risk_dist = 8.5
            
        tp1_calc = round(curr_price + (risk_dist * 1.5), 2)
        tp2_calc = round(curr_price + (risk_dist * 2.5), 2)
    else:
        act = "SELL VÀNG (MARKET)"
        sl_calc = round(recent_high + 1.0, 2)
        risk_dist = round(sl_calc - curr_price, 1)
        if risk_dist < 6.5: 
            sl_calc = round(curr_price + 6.5, 2)
            risk_dist = 6.5
        elif risk_dist > 8.5:
            sl_calc = round(curr_price - 8.5, 2)
            risk_dist = 8.5
            
        tp1_calc = round(curr_price - (risk_dist * 1.5), 2)
        tp2_calc = round(curr_price - (risk_dist * 2.5), 2)
        
    sl_pips = int(risk_dist * 10)
    tp1_pips = int(risk_dist * 1.5 * 10)
    tp2_pips = int(risk_dist * 2.5 * 10)
    
    prompt = PROMPT_TJR.replace("{H1_Trend}", h1_trend_str)\
                       .replace("{KZ_Name}", kz_name)\
                       .replace("{Price}", str(curr_price))\
                       .replace("{Time_Str}", time_str)
                       
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
        ]}],
        max_tokens=400
    )
    ai_comment = response.choices[0].message.content.strip()
    
    tele_msg = f"🎯 [BOT #2: TJR INTRADAY - H1 BIAS + M15 IFVG]\n" \
               f"⏰ Nến M15 đóng lúc: {time_str}\n" \
               f"📊 H1 Trend chủ đạo: {h1_trend_str}\n" \
               f"⌛ Phiên: {kz_name}\n" \
               f"📍 Giá vào ngay: {curr_price}\n\n" \
               f"👉 LỆNH THEO H1: {act}\n" \
               f"🛑 SL chân sóng an toàn: {sl_calc} (-{sl_pips} pips / {risk_dist} giá)\n" \
               f"🎯 TP1 (Target 1:1.5): {tp1_calc} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
               f"🚀 TP2 (Target 1:2.5): {tp2_calc} (+{tp2_pips} pips - BSL/SSL)\n\n" \
               f"💡 NHẬN ĐỊNH CỦA TJR AI:\n{ai_comment}"
               
    print("Đang gửi báo cáo Bot #2 về Telegram...")
    send_telegram_photo(chart_file, tele_msg)
    
    if os.path.exists(chart_file):
        os.remove(chart_file)

if __name__ == "__main__":
    main()

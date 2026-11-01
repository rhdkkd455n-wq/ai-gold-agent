import os
import time
import base64
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
from datetime import datetime, timezone, timedelta
from groq import Groq

# 1. ƯU TIÊN LẤY TOKEN CỦA CON BOT MỚI (TJR_BOT_TOKEN)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TJR_TOKEN = os.environ.get("TJR_BOT_TOKEN", "").strip()
MAIN_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_BOT_TOKEN = TJR_TOKEN if TJR_TOKEN else MAIN_TOKEN

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

print(f"👉 Đang gửi qua Bot có Token đuôi: ...{TELEGRAM_BOT_TOKEN[-6:]}")

client = Groq(api_key=GROQ_API_KEY)
VN_TZ = timezone(timedelta(hours=7))

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
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY trong GitHub Secrets!")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=15min&outputsize=70&apikey={TWELVE_DATA_API_KEY}"
    res = requests.get(url).json()
    if "values" not in res:
        raise ValueError(f"Twelve Data báo lỗi: {res.get('message', 'Không rõ lỗi')}")
        
    df = pd.DataFrame(res['values'])
    df = df.iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(VN_TZ)
    df.set_index('datetime', inplace=True)
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df['Volume'] = 1000
    df_m15 = df.iloc[:-1]
    
    df_h1 = df_m15.resample('1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    h1_is_bull = df_h1['Close'].iloc[-1] > df_h1['EMA21'].iloc[-1]
    
    h_l = df_m15['High'] - df_m15['Low']
    h_c = np.abs(df_m15['High'] - df_m15['Close'].shift())
    l_c = np.abs(df_m15['Low'] - df_m15['Close'].shift())
    df_m15['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    return df_m15.tail(60), h1_is_bull

def find_ifvg_levels(df):
    ifvg_list = []
    for i in range(2, len(df) - 3):
        c1_h, c1_l = df['High'].iloc[i-2], df['Low'].iloc[i-2]
        c3_h, c3_l = df['High'].iloc[i], df['Low'].iloc[i]
        if c3_h < c1_l:
            top_box, bottom_box = c1_l, c3_h
            for j in range(i+1, len(df)):
                if df['Close'].iloc[j] > top_box:
                    ifvg_list.append({'type': 'BULLISH', 'idx_start': i-2, 'idx_end': len(df)-1, 'top': top_box, 'bottom': bottom_box, 'color': 'cyan'})
                    break
        elif c3_l > c1_h:
            top_box, bottom_box = c3_l, c1_h
            for j in range(i+1, len(df)):
                if df['Close'].iloc[j] < bottom_box:
                    ifvg_list.append({'type': 'BEARISH', 'idx_start': i-2, 'idx_end': len(df)-1, 'top': top_box, 'bottom': bottom_box, 'color': 'magenta'})
                    break
    return ifvg_list[-3:]

def generate_tjr_chart(df, ifvg_list, title_text):
    chart_path = "xauusd_tjr_chart.png"
    fig, axlist = mpf.plot(df, type='candle', style='binance', title=f"\n{title_text}", figsize=(14, 7), returnfig=True)
    ax = axlist[0]
    for box_data in ifvg_list:
        w = box_data['idx_end'] - box_data['idx_start'] + 1
        h = box_data['top'] - box_data['bottom']
        rect = patches.Rectangle((box_data['idx_start'], box_data['bottom']), w, h, edgecolor=box_data['color'], facecolor=box_data['color'], alpha=0.35)
        ax.add_patch(rect)
    fig.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return chart_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# HÀM GỬI TELEGRAM AN TOÀN (CHỐNG LỖI QUÁ 1024 KÝ TỰ & IN LOG CHI TIẾT)
def send_telegram_safe(photo_path, text):
    caption_text = text if len(text) <= 1000 else text[:980] + "\n..."
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    with open(photo_path, 'rb') as photo:
        res = requests.post(url_photo, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption_text}, files={'photo': photo})
        print(f"👉 KẾT QUẢ GỬI TELEGRAM BOT 2: Mã {res.status_code} - {res.text}")
        
        # Nếu gửi ảnh thất bại, tự động gửi lại dưới dạng tin nhắn chữ
        if res.status_code != 200:
            print("⚠️ Gửi ảnh bị lỗi, đang chuyển sang gửi tin nhắn văn bản...")
            url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            res2 = requests.post(url_msg, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})
            print(f"👉 KẾT QUẢ GỬI TEXT DỰ PHÒNG: {res2.status_code} - {res2.text}")

PROMPT_TJR = """
Bạn là Chuyên gia Giao dịch TJR Trades (H1 Trend + M15 IFVG Trigger).
Nến M15 Vàng Spot (XAU/USD). Hộp Cyan = Bullish IFVG Hỗ trợ, Hộp Magenta = Bearish IFVG Kháng cự.
H1 Trend: {H1_Trend}. Phiên: {KZ_Name}. Giá: {Price}.

HÃY ĐÁNH GIÁ NGẮN GỌN (Dưới 250 ký tự):
1. M15 retest IFVG đúng hướng H1 chưa?
2. Có Liquidity Sweep lấy đà chưa?
3. QUYẾT ĐỊNH: [BUY NGAY] / [SELL NGAY] HOẶC [ĐỨNG NGOÀI].
"""

def main():
    print("=== BẮT ĐẦU QUÉT BOT #2 ===")
    df_m15, h1_is_bull = get_spot_gold_m15()
    last_time = df_m15.index[-1] + timedelta(minutes=15)
    time_str = last_time.strftime('%H:%M - %d/%m')
    curr_price = round(df_m15['Close'].iloc[-1], 2)
    
    in_kz, kz_name = is_killzone(last_time)
    h1_trend_str = "TĂNG" if h1_is_bull else "GIẢM"
    
    ifvg_levels = find_ifvg_levels(df_m15)
    chart_title = f"[SPOT XAUUSD M15] H1 Trend: {h1_trend_str} - {kz_name}"
    chart_file = generate_tjr_chart(df_m15, ifvg_levels, chart_title)
    base64_img = encode_image(chart_file)
    
    recent_bars = 10
    recent_high = df_m15['High'].iloc[-recent_bars:].max()
    recent_low = df_m15['Low'].iloc[-recent_bars:].min()
    
    if h1_is_bull:
        act = "BUY VÀNG SPOT (MARKET)"
        sl_calc = round(recent_low - 1.0, 2)
        risk_dist = round(curr_price - sl_calc, 1)
        if risk_dist < 6.5: sl_calc = round(curr_price - 6.5, 2); risk_dist = 6.5
        elif risk_dist > 8.5: sl_calc = round(curr_price - 8.5, 2); risk_dist = 8.5
        tp1_calc = round(curr_price + (risk_dist * 1.5), 2)
        tp2_calc = round(curr_price + (risk_dist * 2.5), 2)
    else:
        act = "SELL VÀNG SPOT (MARKET)"
        sl_calc = round(recent_high + 1.0, 2)
        risk_dist = round(sl_calc - curr_price, 1)
        if risk_dist < 6.5: sl_calc = round(curr_price + 6.5, 2); risk_dist = 6.5
        elif risk_dist > 8.5: sl_calc = round(curr_price - 8.5, 2); risk_dist = 8.5
        tp1_calc = round(curr_price - (risk_dist * 1.5), 2)
        tp2_calc = round(curr_price - (risk_dist * 2.5), 2)
        
    sl_pips = int(risk_dist * 10)
    tp1_pips = int(risk_dist * 1.5 * 10)
    tp2_pips = int(risk_dist * 2.5 * 10)
    
    prompt = PROMPT_TJR.replace("{H1_Trend}", h1_trend_str)\
                       .replace("{KZ_Name}", kz_name)\
                       .replace("{Price}", str(curr_price))
                       
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
        ]}],
        max_tokens=250
    )
    ai_comment = response.choices[0].message.content.strip()
    
    tele_msg = f"🎯 [BOT #2: TJR M15 IFVG - SPOT GOLD]\n" \
               f"⏰ Nến M15 đóng lúc: {time_str} (Khớp MT5)\n" \
               f"📊 H1 Trend: {h1_trend_str} | ⌛ {kz_name}\n" \
               f"📍 Giá đóng: {curr_price} (Spot XAU/USD)\n\n" \
               f"👉 LỆNH THEO H1: {act}\n" \
               f"🛑 SL: {sl_calc} (-{sl_pips} pips)\n" \
               f"🎯 TP1: {tp1_calc} (+{tp1_pips} pips) -> Dời BE\n" \
               f"🚀 TP2: {tp2_calc} (+{tp2_pips} pips)\n\n" \
               f"💡 NHẬN ĐỊNH TJR:\n{ai_comment}"
               
    send_telegram_safe(chart_file, tele_msg)
    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

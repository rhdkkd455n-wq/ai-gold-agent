import os
import json
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
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

TICKER = "GC=F"
STATE_FILE = "trade_state.json"
client = Groq(api_key=GROQ_API_KEY)

# HÀM ĐỌC VÀ LƯU BỘ NHỚ LỆNH (TRACKER)
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except:
            return None
    return None

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def clear_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

def get_market_data():
    gold = yf.Ticker(TICKER)
    df_h1 = gold.history(period="10d", interval="1h")
    df_h1 = df_h1[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df_h1.index = pd.to_datetime(df_h1.index)
    df_h1 = df_h1.iloc[:-1]
    
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    high_low = df_h1['High'] - df_h1['Low']
    high_close = np.abs(df_h1['High'] - df_h1['Close'].shift())
    low_close = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_h1['ATR'] = tr.rolling(14).mean()
    
    df_h4 = df_h1.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    
    last_h1_time = df_h1.index[-1]
    hour_in_4h = (last_h1_time.hour % 4) + 1
    is_4h_closed = (hour_in_4h == 4)
    
    return df_h1.tail(60), df_h4.tail(40), is_4h_closed, hour_in_4h

def find_smc_levels(df):
    fvg_list = []
    for i in range(2, len(df)):
        c1_high, c1_low = df['High'].iloc[i-2], df['Low'].iloc[i-2]
        c3_high, c3_low = df['High'].iloc[i], df['Low'].iloc[i]
        if c3_low > c1_high:
            fvg_list.append({'idx_start': i-2, 'idx_end': len(df)-1, 'top': c3_low, 'bottom': c1_high, 'color': 'green'})
        elif c3_high < c1_low:
            fvg_list.append({'idx_start': i-2, 'idx_end': len(df)-1, 'top': c1_low, 'bottom': c3_high, 'color': 'red'})
    return fvg_list[-3:]

def generate_chart(df, fvg_list, title_text):
    chart_path = "xauusd_chart.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    
    fig, axlist = mpf.plot(
        df, type='candle', style='binance', addplot=ema_plot,
        title=f"\n{title_text}", figsize=(14, 7), returnfig=True
    )
    ax = axlist[0]
    for fvg in fvg_list:
        w = fvg['idx_end'] - fvg['idx_start'] + 1
        h = fvg['top'] - fvg['bottom']
        rect = patches.Rectangle((fvg['idx_start'], fvg['bottom']), w, h, edgecolor=fvg['color'], facecolor=fvg['color'], alpha=0.25)
        ax.add_patch(rect)
        
    fig.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return chart_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

def send_telegram_photo(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})

def check_active_trade(df_h1):
    """KIỂM TRA XEM LỆNH CŨ CÓ CẮN TP1, TP2 HOẶC SL CHƯA"""
    trade = load_state()
    if not trade:
        return False
        
    latest_candle = df_h1.iloc[-1]
    c_high = latest_candle['High']
    c_low = latest_candle['Low']
    action = trade['action']
    
    if action == "BUY":
        # Chạm SL
        if c_low <= trade['sl']:
            send_telegram_msg(f"🛑 [BÁO CÁO DÍNH SL]\nKèo BUY Vàng đã chạm mức cắt lỗ tại {trade['sl']} (-{trade['sl_pips']} pips).\nBảo toàn vốn để chờ kèo mới!")
            clear_state()
            return True
        # Chạm TP2 (Full Target)
        elif c_high >= trade['tp2']:
            send_telegram_msg(f"🚀 [FULL TARGET: CHẠM TP2 VÀNG!]\nKèo BUY Vàng đã cắn trọn mốc TP2 tại {trade['tp2']} (+{trade['tp2_pips']} PIPS)!\nĐóng toàn bộ vị thế, chúc mừng anh em!")
            clear_state()
            return True
        # Chạm TP1
        elif c_high >= trade['tp1'] and not trade.get('tp1_hit', False):
            trade['tp1_hit'] = True
            save_state(trade)
            send_telegram_msg(f"🎉 [TING TING: ĐÃ ĐẠT TP1 VÀNG!]\nGiá đã chạm mốc TP1 tại {trade['tp1']} (+{trade['tp1_pips']} PIPS)!\n👉 Hãy đóng bớt 50% khối lượng và KÉO SL VỀ ENTRY HÒA VỐN (BE) để gồng tiếp TP2!")
            return True

    elif action == "SELL":
        # Chạm SL
        if c_high >= trade['sl']:
            send_telegram_msg(f"🛑 [BÁO CÁO DÍNH SL]\nKèo SELL Vàng đã chạm mức cắt lỗ tại {trade['sl']} (-{trade['sl_pips']} pips).\nBảo toàn vốn để chờ kèo mới!")
            clear_state()
            return True
        # Chạm TP2 (Full Target)
        elif c_low <= trade['tp2']:
            send_telegram_msg(f"🚀 [FULL TARGET: CHẠM TP2 VÀNG!]\nKèo SELL Vàng đã cắn trọn mốc TP2 tại {trade['tp2']} (+{trade['tp2_pips']} PIPS)!\nĐóng toàn bộ vị thế, chúc mừng anh em!")
            clear_state()
            return True
        # Chạm TP1
        elif c_low <= trade['tp1'] and not trade.get('tp1_hit', False):
            trade['tp1_hit'] = True
            save_state(trade)
            send_telegram_msg(f"🎉 [TING TING: ĐÃ ĐẠT TP1 VÀNG!]\nGiá đã chạm mốc TP1 tại {trade['tp1']} (+{trade['tp1_pips']} PIPS)!\n👉 Hãy đóng bớt 50% khối lượng và KÉO SL VỀ ENTRY HÒA VỐN (BE) để gồng tiếp TP2!")
            return True

    return False

PROMPT_HOURLY = """
Bạn là Phù thủy Giao dịch SMC & Price Action TraderViet.
Quan sát biểu đồ nến H1 của Vàng vừa đóng cửa. Giá: {Price}. Xu hướng lớn H4: {H4_Trend}.
Cây nến 4H đang ở giờ thứ {Hour_In_4H}/4.

HÃY ĐÁNH GIÁ (Dưới 400 ký tự):
1. Nhận xét cấu trúc nến H1: Có tín hiệu rút râu từ chối cản/FVG đẹp hay đang nén giá?
2. Dự đoán nến 4H trong 1 giờ tới: Sẽ test vùng cản nào tiếp theo?
"""

def main():
    print("=== BẮT ĐẦU QUÉT VÀNG & THEO DÕI TP1/TP2 ===")
    df_h1, df_h4, is_4h_closed, hour_in_4h = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    
    # 1. KIỂM TRA XEM LỆNH CŨ ĐÃ CẮN TP1 HOẶC TP2 CHƯA
    hit_something = check_active_trade(df_h1)
    
    # 2. TÍNH TOÁN KÈO MỚI THEO PIP VÀ ATR
    raw_atr = df_h1['ATR'].iloc[-1]
    risk_pts = round((raw_atr if not np.isnan(raw_atr) else 8.5) * 1.2, 1) # Điểm giá SL
    sl_pips = int(risk_pts * 10) # 1 giá = 10 pips
    
    tp1_pts = round(risk_pts * 1.5, 1)
    tp1_pips = int(tp1_pts * 10)
    
    tp2_pts = round(risk_pts * 3.0, 1)
    tp2_pips = int(tp2_pts * 10)
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    h4_trend_str = "TĂNG" if h4_is_bull else "GIẢM"
    action = "BUY" if h4_is_bull else "SELL"
    
    # VÙNG ENTRY: Khoảng đệm 1.5 giá (15 pips) quanh giá hiện tại
    if action == "BUY":
        entry_from = curr_price
        entry_to = round(curr_price + 1.5, 2)
        sl_val = round(curr_price - risk_pts, 2)
        tp1_val = round(curr_price + tp1_pts, 2)
        tp2_val = round(curr_price + tp2_pts, 2)
    else:
        entry_from = round(curr_price - 1.5, 2)
        entry_to = curr_price
        sl_val = round(curr_price + risk_pts, 2)
        tp1_val = round(curr_price - tp1_pts, 2)
        tp2_val = round(curr_price - tp2_pts, 2)
        
    fvgs = find_smc_levels(df_h1)
    chart_file = generate_chart(df_h1, fvgs, f"[XAUUSD H1] H4 Trend: {h4_trend_str} - Pips Tracker")
    base64_img = encode_image(chart_file)
    
    prompt = PROMPT_HOURLY.replace("{Price}", str(curr_price))\
                          .replace("{H4_Trend}", h4_trend_str)\
                          .replace("{Hour_In_4H}", str(hour_in_4h))
                          
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
        ]}],
        max_tokens=400
    )
    ai_analysis = response.choices[0].message.content.strip()
    
    # ĐÓNG GÓI BẢN TIN TÍN HIỆU CHUẨN VIP SIGNAL CÓ ENTRY RANGE, SL & TP1, TP2 THEO PIPS
    tele_msg = f"⚡ [CHIẾN LƯỢC VÀNG: XAUUSD]\n" \
               f"👉 LỆNH: {action} VÀNG (MARKET)\n\n" \
               f"📍 Vùng Entry: {entry_from} - {entry_to}\n" \
               f"🛑 Cắt lỗ (SL): {sl_val} (-{sl_pips} pips)\n\n" \
               f"🎯 Mục tiêu 1 (TP1): {tp1_val} (+{tp1_pips} pips)\n" \
               f"   *(Chốt 50% khối lượng, dời SL về hòa vốn)*\n" \
               f"🚀 Mục tiêu 2 (TP2): {tp2_val} (+{tp2_pips} pips)\n" \
               f"   *(Gồng trọn sóng H4 - Tỷ lệ R:R 1:3)*\n\n" \
               f"⏳ Tiến trình nến 4H: Đang ở giờ thứ {hour_in_4h}/4\n" \
               f"💡 NHẬN ĐỊNH CỦA AI:\n{ai_analysis}"
               
    send_telegram_photo(chart_file, tele_msg)
    
    # LƯU TRẠNG THÁI VÀO BỘ NHỚ ĐỂ THEO DÕI CẮN TP1 / TP2
    current_trade = load_state()
    if not current_trade:
        save_state({
            "action": action,
            "entry": curr_price,
            "sl": sl_val,
            "tp1": tp1_val,
            "tp2": tp2_val,
            "sl_pips": sl_pips,
            "tp1_pips": tp1_pips,
            "tp2_pips": tp2_pips,
            "tp1_hit": False
        })
        
    if os.path.exists(chart_file):
        os.remove(chart_file)

if __name__ == "__main__":
    main()

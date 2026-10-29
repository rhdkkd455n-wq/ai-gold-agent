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

# 1. KHỞI TẠO VÀ LÀM SẠCH KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

TICKER = "GC=F"
STATE_FILE = "trade_state.json"
client = Groq(api_key=GROQ_API_KEY)

# 2. BỘ NHỚ THEO DÕI TP1, TP2 VÀ SL (TRADE TRACKER)
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

def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

def send_telegram_photo(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})

def check_tp_and_sl_hit(df_h1):
    """KIỂM TRA VÀ GỬI THÔNG BÁO TỰ ĐỘNG KHI ĐẠT TP1, TP2 HOẶC SL"""
    trade = load_state()
    if not trade:
        return
        
    last_candle = df_h1.iloc[-1]
    c_high = last_candle['High']
    c_low = last_candle['Low']
    action = trade['action']
    
    if action == "BUY":
        # 1. Chạm SL
        if c_low <= trade['sl']:
            send_telegram_msg(f"🛑 [BÁO CÁO CẮN SL VÀNG]\nKèo BUY đã chạm mức cắt lỗ tại {trade['sl']} (-{trade['sl_pips']} pips).\nBảo toàn vốn chờ cơ hội mới!")
            clear_state()
            return
        # 2. Chạm TP2 (Full Target)
        elif c_high >= trade['tp2']:
            send_telegram_msg(f"🚀 [FULL TARGET: ĐÃ ĐẠT TP2 VÀNG!]\nKèo BUY vừa cắn trọn mốc TP2 tại {trade['tp2']} (+{trade['tp2_pips']} PIPS)!\nĐóng toàn bộ vị thế, chúc mừng bạn!")
            clear_state()
            return
        # 3. Chạm TP1
        elif c_high >= trade['tp1'] and not trade.get('tp1_hit', False):
            trade['tp1_hit'] = True
            save_state(trade)
            send_telegram_msg(f"🎉 [TING TING: ĐÃ ĐẠT TP1 VÀNG!]\nGiá vừa cắn mốc TP1 tại {trade['tp1']} (+{trade['tp1_pips']} PIPS)!\n👉 Hãy đóng 50% khối lượng và KÉO SL VỀ HÒA VỐN (BE) để gồng tiếp TP2!")
            return

    elif action == "SELL":
        # 1. Chạm SL
        if c_high >= trade['sl']:
            send_telegram_msg(f"🛑 [BÁO CÁO CẮN SL VÀNG]\nKèo SELL đã chạm mức cắt lỗ tại {trade['sl']} (-{trade['sl_pips']} pips).\nBảo toàn vốn chờ cơ hội mới!")
            clear_state()
            return
        # 2. Chạm TP2 (Full Target)
        elif c_low <= trade['tp2']:
            send_telegram_msg(f"🚀 [FULL TARGET: ĐÃ ĐẠT TP2 VÀNG!]\nKèo SELL vừa cắn trọn mốc TP2 tại {trade['tp2']} (+{trade['tp2_pips']} PIPS)!\nĐóng toàn bộ vị thế, chúc mừng bạn!")
            clear_state()
            return
        # 3. Chạm TP1
        elif c_low <= trade['tp1'] and not trade.get('tp1_hit', False):
            trade['tp1_hit'] = True
            save_state(trade)
            send_telegram_msg(f"🎉 [TING TING: ĐÃ ĐẠT TP1 VÀNG!]\nGiá vừa cắn mốc TP1 tại {trade['tp1']} (+{trade['tp1_pips']} PIPS)!\n👉 Hãy đóng 50% khối lượng và KÉO SL VỀ HÒA VỐN (BE) để gồng tiếp TP2!")
            return

# 3. LẤY DỮ LIỆU NẾN H1 VÀ H4
def get_market_data():
    gold = yf.Ticker(TICKER)
    df_h1 = gold.history(period="15d", interval="1h")
    df_h1 = df_h1[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df_h1.index = pd.to_datetime(df_h1.index)
    df_h1 = df_h1.iloc[:-1] # Bỏ nến đang chạy
    
    # Tính EMA 21 & ATR 14 H1
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    h_l = df_h1['High'] - df_h1['Low']
    h_c = np.abs(df_h1['High'] - df_h1['Close'].shift())
    l_c = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    # Gộp nến H4
    df_h4 = df_h1.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    h_l4 = df_h4['High'] - df_h4['Low']
    h_c4 = np.abs(df_h4['High'] - df_h4['Close'].shift())
    l_c4 = np.abs(df_h4['Low'] - df_h4['Close'].shift())
    df_h4['ATR'] = pd.concat([h_l4, h_c4, l_c4], axis=1).max(axis=1).rolling(14).mean()
    
    # Xác định nến 4H đóng hay đang chạy
    last_h1_time = df_h1.index[-1]
    hour_in_4h = (last_h1_time.hour % 4) + 1
    is_4h_closed = (hour_in_4h == 4)
    
    return df_h1.tail(60), df_h4.tail(45), is_4h_closed, hour_in_4h

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

def generate_chart(df, fvg_list, tf_name):
    chart_path = f"xauusd_{tf_name.lower()}_chart.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    
    fig, axlist = mpf.plot(
        df, type='candle', style='binance', addplot=ema_plot,
        title=f"\n[XAUUSD {tf_name}] Price Action & SMC", figsize=(14, 7), returnfig=True
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

def query_groq(chart_file, prompt):
    base64_img = encode_image(chart_file)
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]}],
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Phân tích bám sát xu hướng thị trường: {e}"

def main():
    print("=== BẮT ĐẦU CHU KỲ QUÉT VÀNG CHUYÊN SÂU ===")
    df_h1, df_h4, is_4h_closed, hour_in_4h = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    
    # BƯỚC 1: TỰ ĐỘNG KIỂM TRA XEM LỆNH CŨ CÓ CẮN TP1, TP2 HOẶC SL CHƯA
    check_tp_and_sl_hit(df_h1)
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    h4_trend_str = "TĂNG" if h4_is_bull else "GIẢM"
    
    # ================= BƯỚC 2: PHÂN TÍCH NẾN 1H (GỬI MỖI 1 TIẾNG) =================
    atr_h1 = df_h1['ATR'].iloc[-1]
    risk_h1 = round((atr_h1 if not np.isnan(atr_h1) else 4.5) * 1.2, 1) # 4 - 6 giá
    sl_h1_pips = int(risk_h1 * 10)
    tp1_h1_pips = int(risk_h1 * 1.5 * 10)
    tp2_h1_pips = int(risk_h1 * 2.5 * 10)
    
    act_h1 = "BUY" if h4_is_bull else "SELL"
    if act_h1 == "BUY":
        e_from_h1, e_to_h1 = curr_price, round(curr_price + 1.0, 2)
        sl_h1 = round(curr_price - risk_h1, 2)
        tp1_h1 = round(curr_price + (risk_h1 * 1.5), 2)
        tp2_h1 = round(curr_price + (risk_h1 * 2.5), 2)
    else:
        e_from_h1, e_to_h1 = round(curr_price - 1.0, 2), curr_price
        sl_h1 = round(curr_price + risk_h1, 2)
        tp1_h1 = round(curr_price - (risk_h1 * 1.5), 2)
        tp2_h1 = round(curr_price - (risk_h1 * 2.5), 2)
        
    chart_h1 = generate_chart(df_h1, find_smc_levels(df_h1), "H1")
    
    prompt_h1 = f"""
    Bạn là Chuyên gia Price Action & SMC TraderViet.
    Quan sát biểu đồ nến H1 của Vàng vừa đóng cửa. Giá: {curr_price}. Xu hướng H4: {h4_trend_str}.
    Tiến trình nến 4H: Đang ở giờ thứ {hour_in_4h}/4.
    
    HÃY TRẢ LỜI ĐÚNG 2 MỤC SAU (Dưới 350 ký tự):
    1. QUYẾT ĐỊNH NẾN 1H: [VÀO LỆNH NGAY ({act_h1})] hay [KHÔNG VÀO LỆNH (ĐỨNG NGOÀI)]? Nêu lý do vì sao nến 1H này đủ/chưa đủ điều kiện.
    2. DỰ ĐOÁN NẾN 4H TRONG 1 GIỜ TỚI: Dự đoán nến 4H sẽ di chuyển ra sao trong 1 tiếng tiếp theo và kiểm tra mức giá nào?
    """
    ai_h1 = query_groq(chart_h1, prompt_h1)
    
    msg_h1 = f"🟢 [BÁO CÁO NẾN 1H & DỰ ĐOÁN 4H]\n" \
             f"⏰ Giá nến 1H vừa đóng: {curr_price}\n" \
             f"⏳ Tiến trình nến 4H: Giờ thứ {hour_in_4h}/4\n\n" \
             f"📍 Vùng Entry H1: {e_from_h1} - {e_to_h1}\n" \
             f"🛑 SL: {sl_h1} (-{sl_h1_pips} pips ngắn)\n" \
             f"🎯 TP1: {tp1_h1} (+{tp1_h1_pips} pips) -> Dời hòa vốn\n" \
             f"🚀 TP2: {tp2_h1} (+{tp2_h1_pips} pips) -> Gồng hết sóng\n\n" \
             f"{ai_h1}"
             
    print("Gửi báo cáo H1 về Telegram...")
    send_telegram_photo(chart_h1, msg_h1)
    if os.path.exists(chart_h1): os.remove(chart_h1)
    
    # Lưu lệnh vào bộ nhớ nếu có quyết định vào lệnh và chưa có lệnh nào đang chạy
    if "[VÀO LỆNH NGAY" in ai_h1 and not load_state():
        save_state({
            "action": act_h1, "entry": curr_price, "sl": sl_h1,
            "tp1": tp1_h1, "tp2": tp2_h1, "sl_pips": sl_h1_pips,
            "tp1_pips": tp1_h1_pips, "tp2_pips": tp2_h1_pips, "tp1_hit": False
        })
        
    time.sleep(2)
    
    # ================= BƯỚC 3: KHI NẾN 4H CHÍNH THỨC ĐÓNG CỬA (MỖI 4 TIẾNG) =================
    if is_4h_closed:
        print("👑 NẾN 4H VỪA ĐÓNG CỬA! GỬI BÁO CÁO SWING H4...")
        atr_h4 = df_h4['ATR'].iloc[-1]
        risk_h4 = round((atr_h4 if not np.isnan(atr_h4) else 14.0) * 1.2, 1) # 14 - 18 giá
        sl_h4_pips = int(risk_h4 * 10)
        tp1_h4_pips = int(risk_h4 * 1.5 * 10)
        tp2_h4_pips = int(risk_h4 * 3.0 * 10)
        
        act_h4 = "BUY" if h4_is_bull else "SELL"
        if act_h4 == "BUY":
            e_from_h4, e_to_h4 = curr_price, round(curr_price + 2.0, 2)
            sl_h4 = round(curr_price - risk_h4, 2)
            tp1_h4 = round(curr_price + (risk_h4 * 1.5), 2)
            tp2_h4 = round(curr_price + (risk_h4 * 3.0), 2)
        else:
            e_from_h4, e_to_h4 = round(curr_price - 2.0, 2), curr_price
            sl_h4 = round(curr_price + risk_h4, 2)
            tp1_h4 = round(curr_price - (risk_h4 * 1.5), 2)
            tp2_h4 = round(curr_price - (risk_h4 * 3.0), 2)
            
        chart_h4 = generate_chart(df_h4, find_smc_levels(df_h4), "H4")
        prompt_h4 = f"""
        Bạn là Chuyên gia Giao dịch SMC & Price Action TraderViet.
        Quan sát biểu đồ nến 4H của Vàng VỪA ĐÓNG CỬA. Giá: {curr_price}.
        Hãy phân tích ngắn gọn (dưới 350 ký tự):
        1. Giải phẫu cây nến 4H vừa kết thúc: Là nến gì? Phản ứng cản/FVG ra sao?
        2. Kế hoạch Swing 4H: Đưa ra chiến lược cho cây nến 4H mới vừa ra đời.
        """
        ai_h4 = query_groq(chart_h4, prompt_h4)
        
        msg_h4 = f"👑 [BẢN TIN ĐẶC BIỆT: NẾN 4H VỪA ĐÓNG CỬA]\n" \
                 f"⏰ Mức giá đóng nến 4H: {curr_price}\n" \
                 f"📊 Cấu trúc xu hướng lớn: {h4_trend_str}\n\n" \
                 f"📍 Vùng Entry Swing H4: {e_from_h4} - {e_to_h4}\n" \
                 f"🛑 SL: {sl_h4} (-{sl_h4_pips} pips H4)\n" \
                 f"🎯 TP1: {tp1_h4} (+{tp1_h4_pips} pips)\n" \
                 f"🚀 TP2: {tp2_h4} (+{tp2_h4_pips} pips - R:R 1:3)\n\n" \
                 f"{ai_h4}"
                 
        print("Gửi bản tin H4 về Telegram...")
        send_telegram_photo(chart_h4, msg_h4)
        if os.path.exists(chart_h4): os.remove(chart_h4)

if __name__ == "__main__":
    main()

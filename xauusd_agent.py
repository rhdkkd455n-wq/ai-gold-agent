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
from datetime import datetime, timezone, timedelta
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

TICKER = "GC=F"
STATE_FILE = "trade_state.json"
client = Groq(api_key=GROQ_API_KEY)
VN_TZ = timezone(timedelta(hours=7))

# QUẢN LÝ BỘ NHỚ AN TOÀN (TỰ SỬA LỖI FORMAT CŨ)
def load_state():
    default_state = {"active_trades": []}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "active_trades" in data and isinstance(data["active_trades"], list):
                    return data
        except:
            pass
    return default_state

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

def send_telegram_photo(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})

def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    if not active_trades:
        return
        
    last_candle = df_h1.iloc[-1]
    c_high = last_candle['High']
    c_low = last_candle['Low']
    remaining_trades = []
    
    for trade in active_trades:
        t_id = trade['id']
        act = trade['action']
        tf = trade['timeframe']
        entry = trade['entry']
        sl = trade['sl']
        tp1 = trade['tp1']
        tp2 = trade['tp2']
        
        is_closed = False
        
        if act == "BUY":
            if c_low <= sl:
                send_telegram_msg(f"🛑 [CẮN SL] Kèo {act} {tf} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                is_closed = True
            elif c_high >= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} {tf} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LỢI NHUẬN: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1] Kèo {act} {tf} ({t_id})\n• Entry: {entry} -> Cắn TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về {entry} hòa vốn. Gồng tiếp TP2: {tp2}!")
        
        elif act == "SELL":
            if c_high >= sl:
                send_telegram_msg(f"🛑 [CẮN SL] Kèo {act} {tf} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                is_closed = True
            elif c_low <= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} {tf} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LỢI NHUẬN: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1] Kèo {act} {tf} ({t_id})\n• Entry: {entry} -> Cắn TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về {entry} hòa vốn. Gồng tiếp TP2: {tp2}!")

        if not is_closed:
            remaining_trades.append(trade)
            
    state['active_trades'] = remaining_trades
    save_state(state)

def get_market_data():
    gold = yf.Ticker(TICKER)
    df = gold.history(period="15d", interval="1h")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = df.index.tz_convert(VN_TZ)
    df = df.iloc[:-1] # Bỏ nến đang chạy
    
    # Chỉ số H1
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    h_l = df['High'] - df['Low']
    h_c = np.abs(df['High'] - df['Close'].shift())
    l_c = np.abs(df['Low'] - df['Close'].shift())
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    # Gộp nến H4 chuẩn MT5 (Đóng nến lúc: 01h, 05h, 09h, 13h, 17h, 21h)
    df_h4 = df.resample('4h', offset='1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    h_l4 = df_h4['High'] - df_h4['Low']
    h_c4 = np.abs(df_h4['High'] - df_h4['Close'].shift())
    l_c4 = np.abs(df_h4['Low'] - df_h4['Close'].shift())
    df_h4['ATR'] = pd.concat([h_l4, h_c4, l_c4], axis=1).max(axis=1).rolling(14).mean()
    
    # Giờ đóng nến = Giờ mở nến + 1 tiếng
    candle_open_time = df.index[-1]
    candle_close_time = candle_open_time + timedelta(hours=1)
    
    is_4h_closed = candle_close_time.hour in [1, 5, 9, 13, 17, 21]
    
    if is_4h_closed:
        hour_in_4h = 4
    else:
        hour_in_4h = ((candle_close_time.hour - 1) % 4)
        if hour_in_4h == 0: hour_in_4h = 4
        
    return df.tail(60), df_h4.tail(45), is_4h_closed, hour_in_4h, candle_close_time

def find_smc_levels(df):
    fvg_list = []
    for i in range(2, len(df)):
        c1_h, c1_l = df['High'].iloc[i-2], df['Low'].iloc[i-2]
        c3_h, c3_l = df['High'].iloc[i], df['Low'].iloc[i]
        if c3_l > c1_h:
            fvg_list.append({'idx_start': i-2, 'idx_end': len(df)-1, 'top': c3_l, 'bottom': c1_h, 'color': 'green'})
        elif c3_h < c1_l:
            fvg_list.append({'idx_start': i-2, 'idx_end': len(df)-1, 'top': c1_l, 'bottom': c3_h, 'color': 'red'})
    return fvg_list[-3:]

def generate_chart(df, fvg_list, tf_name):
    chart_path = f"xauusd_{tf_name.lower()}_chart.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    
    fig, axlist = mpf.plot(
        df, type='candle', style='binance', addplot=ema_plot,
        title=f"\n[XAUUSD {tf_name}] MT5 Timeframe Sync", figsize=(14, 7), returnfig=True
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
        res = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]}],
            max_tokens=400
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Phân tích thị trường ổn định: {e}"

def main():
    print("=== ĐỒNG BỘ CHUẨN GIỜ ĐÓNG NẾN MT5 & KÍCH HOẠT NẾN 4H ===")
    df_h1, df_h4, is_4h_closed, hour_in_4h, close_time = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    
    # 1. KIỂM TRA TP / SL CỦA CÁC LỆNH ĐANG THEO DÕI
    check_tp_and_sl_hit(df_h1)
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    h4_trend_str = "TĂNG" if h4_is_bull else "GIẢM"
    
    # ================= 2. BÁO CÁO NẾN 1H (GỬI MỖI TIẾNG) =================
    atr_h1 = df_h1['ATR'].iloc[-1]
    risk_h1 = round((atr_h1 if not np.isnan(atr_h1) else 4.5) * 1.2, 1)
    sl_h1_pips = int(risk_h1 * 10)
    tp1_h1_pips = int(risk_h1 * 1.5 * 10)
    tp2_h1_pips = int(risk_h1 * 2.5 * 10)
    
    act_h1 = "BUY" if h4_is_bull else "SELL"
    t_id_h1 = f"#H1_{close_time.strftime('%d%m_%H')}H"
    
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
    Bạn là Chuyên gia SMC & Price Action TraderViet.
    Nến H1 Vàng VỪA ĐÓNG CỬA lúc {time_str} (Giờ VN). Giá: {curr_price}. Xu hướng H4: {h4_trend_str}.
    Tiến trình nến 4H: Đang ở giờ {hour_in_4h}/4.
    
    HÃY ĐÁNH GIÁ (Dưới 300 ký tự):
    1. QUYẾT ĐỊNH NẾN 1H: [VÀO LỆNH NGAY ({act_h1})] hay [KHÔNG VÀO LỆNH (ĐỨNG NGOÀI)]? Nêu lý do.
    2. DỰ ĐOÁN NẾN 4H TRONG 1 GIỜ TỚI: Hướng di chuyển tiếp theo.
    """
    ai_h1 = query_groq(chart_h1, prompt_h1)
    
    msg_h1 = f"🟢 [BÁO CÁO NẾN 1H & DỰ ĐOÁN 4H]\n" \
             f"🎫 Mã vé lệnh: {t_id_h1}\n" \
             f"⏰ Nến H1 đóng lúc: {time_str} (Chuẩn MT5)\n" \
             f"⏳ Tiến trình nến 4H: Giờ thứ {hour_in_4h}/4\n\n" \
             f"📍 Vùng Entry H1: {e_from_h1} - {e_to_h1}\n" \
             f"🛑 SL: {sl_h1} (-{sl_h1_pips} pips)\n" \
             f"🎯 TP1: {tp1_h1} (+{tp1_h1_pips} pips) -> Dời hòa vốn\n" \
             f"🚀 TP2: {tp2_h1} (+{tp2_h1_pips} pips)\n\n" \
             f"{ai_h1}"
             
    send_telegram_photo(chart_h1, msg_h1)
    if os.path.exists(chart_h1): os.remove(chart_h1)
    
    # GHI NHỚ LỆNH AN TOÀN (KHÔNG BAO GIỜ BỊ LỖI KEY)
    if "[VÀO LỆNH NGAY" in ai_h1:
        st = load_state()
        if "active_trades" not in st or not isinstance(st["active_trades"], list):
            st["active_trades"] = []
        st['active_trades'].append({
            "id": t_id_h1, "timeframe": "H1", "action": act_h1,
            "entry": curr_price, "sl": sl_h1, "tp1": tp1_h1, "tp2": tp2_h1,
            "sl_pips": sl_h1_pips, "tp1_pips": tp1_h1_pips, "tp2_pips": tp2_h1_pips,
            "tp1_hit": False
        })
        save_state(st)
        
    time.sleep(2)
    
    # ================= 3. BẢN TIN ĐẶC BIỆT KHI NẾN 4H CHÍNH THỨC ĐÓNG CỬA =================
    if is_4h_closed:
        print("👑 NẾN 4H ĐÓNG CỬA CHUẨN GIỜ MT5! GỬI BÁO CÁO SWING H4...")
        atr_h4 = df_h4['ATR'].iloc[-1]
        risk_h4 = round((atr_h4 if not np.isnan(atr_h4) else 14.0) * 1.2, 1)
        sl_h4_pips = int(risk_h4 * 10)
        tp1_h4_pips = int(risk_h4 * 1.5 * 10)
        tp2_h4_pips = int(risk_h4 * 3.0 * 10)
        
        act_h4 = "BUY" if h4_is_bull else "SELL"
        t_id_h4 = f"#H4_{close_time.strftime('%d%m_%H')}H"
        
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
        Bạn là Chuyên gia SMC & Price Action TraderViet.
        Nến 4H Vàng VỪA CHÍNH THỨC ĐÓNG lúc {time_str} (Giờ VN - Chuẩn MT5). Giá: {curr_price}.
        Hãy nhận định ngắn (dưới 300 ký tự):
        1. Giải phẫu cây nến 4H vừa kết thúc.
        2. Kế hoạch Swing cho cây nến 4H MỚI vừa mở cửa.
        """
        ai_h4 = quer

import os
import json
import time
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

STATE_FILE = "trade_state.json"
client = Groq(api_key=GROQ_API_KEY)
VN_TZ = timezone(timedelta(hours=7))

def load_state():
    default_state = {"active_trades": [], "calendar_sent_date": ""}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "active_trades" not in data: data["active_trades"] = []
                    if "calendar_sent_date" not in data: data["calendar_sent_date"] = ""
                    return data
        except: pass
    return default_state

def save_state(state):
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=4)

def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text}, timeout=10)

def send_telegram_photo(photo_path, text):
    caption_text = text if len(text) <= 1000 else text[:980] + "\n..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption_text}, files={'photo': photo}, timeout=15)

def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    if not active_trades: return
    last_c = df_h1.iloc[-1]
    c_high, c_low = last_c['High'], last_c['Low']
    remaining = []
    for trade in active_trades:
        t_id, act = trade['id'], trade['action']
        entry, sl, tp1, tp2 = trade['entry'], trade['sl'], trade['tp1'], trade['tp2']
        is_closed = False
        if act == "BUY":
            if c_low <= sl:
                send_telegram_msg(f"🛑 [CẮN SL] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips. Bảo toàn vốn!")
                is_closed = True
            elif c_high >= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
        elif act == "SELL":
            if c_high >= sl:
                send_telegram_msg(f"🛑 [CẮN SL] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips. Bảo toàn vốn!")
                is_closed = True
            elif c_low <= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
        if not is_closed: remaining.append(trade)
    state['active_trades'] = remaining
    save_state(state)

def get_economic_calendar():
    events_today = []
    now_vn = datetime.now(VN_TZ)
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        res = requests.get(url, timeout=5).json()
        for item in res:
            if item.get("country") == "USD" and item.get("impact") == "High":
                event_dt = datetime.fromisoformat(item.get("date")).astimezone(VN_TZ)
                if event_dt.date() == now_vn.date():
                    t_str = event_dt.strftime('%H:%M')
                    events_today.append(f"⚠️ {t_str} - {item.get('title')} (Dự báo: {item.get('forecast','N/A')})")
    except: pass
    return events_today

def get_market_data():
    if not TWELVE_DATA_API_KEY: raise ValueError("Chưa có TWELVE_DATA_API_KEY!")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    res = requests.get(url, timeout=10).json()
    if "values" not in res: raise ValueError(f"Twelve Data lỗi: {res.get('message')}")
    df = pd.DataFrame(res['values']).iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df['Volume'] = 1000
    df_h1 = df.iloc[:-1].copy()
    
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    h_l, h_c, l_c = df_h1['High'] - df_h1['Low'], np.abs(df_h1['High'] - df_h1['Close'].shift()), np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    df_h4 = df_h1.resample('4h', offset='1h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    
    candle_open = df_h1.index[-1]
    candle_close = candle_open + timedelta(hours=1)
    is_4h_closed = candle_close.hour in [1, 5, 9, 13, 17, 21]
    prog = "NẾN 4H VỪA ĐÓNG CỬA (Giờ 1/4)" if is_4h_closed else f"Nến 4H đang ở giờ {((candle_close.hour - 1) % 4) + 1}/4"
    return df_h1.tail(60), df_h4.tail(45), is_4h_closed, prog, candle_close

def generate_chart(df, title_text):
    path = "xauusd_clean.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    mpf.plot(df, type='candle', style='binance', addplot=ema_plot, title=f"\n{title_text}", figsize=(13, 6.5), savefig=path)
    return path

def build_candle_dossier(df_h1):
    dossier = []
    for t, row in df_h1.tail(8).iterrows():
        rng = row['High'] - row['Low']
        u_w = row['High'] - max(row['Open'], row['Close'])
        l_w = min(row['Open'], row['Close']) - row['Low']
        direct = "XANH" if row['Close'] >= row['Open'] else "ĐỎ"
        dossier.append(f"• {t.strftime('%H:%M')}: {direct} | C={row['Close']:.1f}, H={row['High']:.1f}, L={row['Low']:.1f} (Râu trên={u_w:.1f}, Râu dưới={l_w:.1f})")
    return "\n".join(dossier)

def ai_deep_analysis(h4_trend, curr_price, ema21_now, atr_now, candle_dossier):
    prompt = f"""
Bạn là Chuyên gia Giao dịch Price Action & SMC TraderViet (Gold Specialist).
DỮ LIỆU 8 NẾN H1 GẦN NHẤT:
{candle_dossier}

THÔNG SỐ THỊ TRƯỜNG:
- Giá Spot hiện tại: {curr_price} | EMA 21 H1: {ema21_now}
- Xu hướng lớn H4: {h4_trend} | ATR H1: {atr_now:.1f} giá

HÃY PHÂN TÍCH CHUYÊN SÂU CHI TIẾT THEO ĐÚNG 3 PHẦN:
**1. ĐỌC VỊ HÀNH VI GIÁ (PRICE ACTION)**
* Nhận định chi tiết về lực nến: Có hiện tượng xả mạnh (dump), gom hàng, rút râu từ chối cản, hay nến đi ngang?

**2. ĐÁNH GIÁ CẤU TRÚC H4 & CÁC VÙNG CẢN**
* Tương quan xu hướng H4 và nến H1. Vùng hỗ trợ/kháng cự then chốt tiếp theo mà giá sẽ kiểm định.

**3. KỊCH BẢN GIÁ & HÀNH ĐỘNG**
* Kế hoạch hành động cụ thể cho trader.

KẾT THÚC BẰNG DÒNG LỆNH:
[QUYET_DINH]: BUY hoặc SELL hoặc STAND_ASIDE
"""
    models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    for m in models:
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=650,
                temperature=0.3,
                timeout=20.0
            )
            raw = res.choices[0].message.content.strip()
            act = "STAND_ASIDE"
            if "[QUYET_DINH]:" in raw:
                part = raw.split("[QUYET_DINH]:")[1].strip()
                if "BUY" in part: act = "BUY"
                elif "SELL" in part: act = "SELL"
            clean_text = raw.replace(f"[QUYET_DINH]: {act}", "").replace("[QUYET_DINH]:", "").strip()
            return act, clean_text
        except:
            time.sleep(1)
    return "STAND_ASIDE", "Thị trường đang kiểm định vùng cản kỹ thuật. Khuyến nghị theo dõi thêm."

def main():
    print("=== CHẠY BOT #1: TRADERVIET H1/H4 ===")
    state = load_state()
    now_vn = datetime.now(VN_TZ)
    today_str = now_vn.strftime('%d/%m/%Y')
    
    # Gửi lịch tin sáng (7h - 9h30)
    if state.get("calendar_sent_date") != today_str and (7 <= now_vn.hour <= 9):
        evs = get_economic_calendar()
        if evs:
            send_telegram_msg(f"📅 [LỊCH TIN ĐỎ KINH TẾ MỸ HÔM NAY - {today_str}]\n\n" + "\n".join(evs) + "\n\n👉 Khuyến nghị: Né giờ tin đỏ 15 phút!")
        state["calendar_sent_date"] = today_str
        save_state(state)
        time.sleep(1)
        
    df_h1, df_h4, is_4h_closed, prog, c_close = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = c_close.strftime('%H:%M - %d/%m')
    t_id = f"#H1_{c_close.strftime('%d%m_%H')}H"
    check_tp_and_sl_hit(df_h1)
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    h4_trend_str = "TĂNG" if h4_is_bull else "GIẢM"
    c_last = df_h1.iloc[-1]
    ema21_now = round(c_last['EMA21'], 2)
    atr_now = round(df_h1['ATR'].iloc[-1], 1)
    if atr_now < 5.5: atr_now = 6.0
    
    candle_dossier = build_candle_dossier(df_h1)
    chart_file = generate_chart(df_h1, f"[SPOT XAUUSD H1] H4 Trend: {h4_trend_str}")
    
    act, analysis = ai_deep_analysis(h4_trend_str, curr_price, ema21_now, atr_now, candle_dossier)
    
    header = f"🟢 [BOT #1: TRADERVIET H1/H4]\n" \
             f"🎫 Mã vé: {t_id}\n" \
             f"⏰ Nến H1 đóng: {time_str} (Khớp MT5)\n" \
             f"📊 H4 Trend: {h4_trend_str} | ⏳ {prog}\n" \
             f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n"
             
    risk_dist = max(atr_now * 1.2, 6.5)
    sl_pips = int(risk_dist * 10)
    
    if act in ["BUY", "SELL"]:
        sl_val = round(curr_price - risk_dist if act=="BUY" else curr_price + risk_dist, 2)
        tp1_val = round(curr_price + risk_dist * 2.0 if act=="BUY" else curr_price - risk_dist * 2.0, 2)
        tp2_val = round(curr_price + risk_dist * 3.0 if act=="BUY" else curr_price - risk_dist * 3.0, 2)
        tp1_pips = int(risk_dist * 20)
        tp2_pips = int(risk_dist * 30)
        
        caption = f"{header}" \
                  f"🚨 QUYẾT ĐỊNH: VÀO {act} VÀNG (MARKET)\n" \
                  f"📍 Entry: {curr_price}\n" \
                  f"🛑 Cắt lỗ (SL): {sl_val} (-{sl_pips} pips)\n" \
                  f"🎯 TP1 (Chuẩn R:R 1:2): {tp1_val} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                  f"🚀 TP2 (Chuẩn R:R 1:3): {tp2_val} (+{tp2_pips} pips)"
                  
        st = load_state()
        if "active_trades" not in st or not isinstance(st["active_trades"], list): st["active_trades"] = []
        st['active_trades'].append({
            "id": t_id, "action": act, "timeframe": "H1",
            "entry": curr_price, "sl": sl_val, "tp1": tp1_val, "tp2": tp2_val,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
    else:
        caption = f"{header}" \
                  f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ KÈO A+)\n\n" \
                  f"👉 Lưu ý: AI đánh giá chưa đủ hội tụ điều kiện để vào lệnh."
                  
    send_telegram_photo(chart_file, caption)
    time.sleep(1)
    send_telegram_msg(f"💡 [NHẬN ĐỊNH CHUYÊN SÂU TỪ AI - {t_id}]\n\n{analysis}")
    
    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

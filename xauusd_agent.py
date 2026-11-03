import os
import re
import json
import time
import base64
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

# 1. KHỞI TẠO KEY BẢO MẬT
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
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

def send_telegram_photo(photo_path, text):
    caption_text = text if len(text) <= 1000 else text[:980] + "\n..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption_text}, files={'photo': photo})

# ================= 2. SĂN ĐỘ LỆCH SỐ LIỆU FOREXFACTORY =================
def clean_num(val_str):
    if not val_str or val_str == "N/A": return None
    s = re.sub(r'[^\d.-]', '', val_str)
    try: return float(s)
    except: return None

def scan_news_deviation():
    now_vn = datetime.now(VN_TZ)
    news_signals = []
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        res = requests.get(url, timeout=5).json()
        for item in res:
            if item.get("country") == "USD" and item.get("impact") == "High":
                event_dt = datetime.fromisoformat(item.get("date")).astimezone(VN_TZ)
                diff_sec = (now_vn - event_dt).total_seconds()
                if 0 <= diff_sec <= 7200:
                    act_raw, fc_raw = item.get("actual"), item.get("forecast")
                    title = item.get("title", "")
                    if act_raw and fc_raw:
                        actual = clean_num(act_raw)
                        forecast = clean_num(fc_raw)
                        if actual is not None and forecast is not None:
                            dev = actual - forecast
                            is_inv = "Unemployment" in title or "Claims" in title
                            if (dev > 0 and not is_inv) or (dev < 0 and is_inv):
                                bias = "SELL"
                                desc = f"Thực tế {act_raw} TỐT HƠN dự báo {fc_raw} -> USD Mạnh -> VÀNG GIẢM"
                            else:
                                bias = "BUY"
                                desc = f"Thực tế {act_raw} YẾU HƠN dự báo {fc_raw} -> USD Suy yếu -> VÀNG TĂNG"
                            news_signals.append({
                                "title": title, "actual": act_raw, "forecast": fc_raw,
                                "bias": bias, "desc": desc, "time": event_dt.strftime('%H:%M')
                            })
    except Exception as e:
        print(f"Lỗi cào tin tức: {e}")
    return news_signals

# ================= 3. THEO DÕI TP / SL =================
def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    if not active_trades: return
    last_candle = df_h1.iloc[-1]
    c_high, c_low = last_candle['High'], last_candle['Low']
    remaining_trades = []
    
    for trade in active_trades:
        t_id, act, tf = trade['id'], trade['action'], trade['timeframe']
        entry, sl, tp1, tp2 = trade['entry'], trade['sl'], trade['tp1'], trade['tp2']
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
                
        if not is_closed: remaining_trades.append(trade)
    state['active_trades'] = remaining_trades
    save_state(state)

# ================= 4. LẤY NẾN SPOT XAU/USD =================
def get_market_data():
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY trong GitHub Secrets!")
        
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    res = requests.get(url).json()
    if "values" not in res: raise ValueError(f"Lỗi Twelve Data: {res.get('message')}")
        
    df = pd.DataFrame(res['values'])
    df = df.iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df['Volume'] = 1000
    df_h1 = df.iloc[:-1] # Bỏ nến đang chạy
    
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    h_l, h_c, l_c = df_h1['High'] - df_h1['Low'], np.abs(df_h1['High'] - df_h1['Close'].shift()), np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    df_h4 = df_h1.resample('4h', offset='1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    h_l4, h_c4, l_c4 = df_h4['High'] - df_h4['Low'], np.abs(df_h4['High'] - df_h4['Close'].shift()), np.abs(df_h4['Low'] - df_h4['Close'].shift())
    df_h4['ATR'] = pd.concat([h_l4, h_c4, l_c4], axis=1).max(axis=1).rolling(14).mean()
    
    candle_open_time = df_h1.index[-1]
    candle_close_time = candle_open_time + timedelta(hours=1)
    is_4h_closed = candle_close_time.hour in [1, 5, 9, 13, 17, 21]
    if is_4h_closed: progress_text = "NẾN 4H VỪA ĐÓNG CỬA (Bắt đầu nến 4H mới: Giờ 1/4)"
    else: progress_text = f"Nến 4H đang ở giờ thứ {((candle_close_time.hour - 1) % 4) + 1}/4"
        
    return df_h1.tail(60), df_h4.tail(45), is_4h_closed, progress_text, candle_close_time

def generate_clean_chart(df, title_text):
    chart_path = "xauusd_clean_chart.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    mpf.plot(df, type='candle', style='binance', addplot=ema_plot, title=f"\n{title_text}", figsize=(14, 7), savefig=chart_path)
    return chart_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file: return base64.b64encode(image_file.read()).decode('utf-8')

PROMPT_H1_STANDARD = """
Bạn là Chuyên gia Giao dịch Price Action & SMC TraderViet (Gold Specialist).
Biểu đồ nến H1 Vàng Spot (XAU/USD) kèm đường EMA 21 cam.
Xu hướng lớn H4: {H4_Trend}. Giá hiện tại: {Price} lúc {Time_Str} (Khớp MT5 & TradingView).

HÃY ĐÁNH GIÁ CHUYÊN NGHIỆP VÀ CHỌN ĐÚNG 1 TRONG 2:
1. QUYẾT ĐỊNH:
   - [CÓ KÈO VÀO LỆNH: BUY] hoặc [CÓ KÈO VÀO LỆNH: SELL] (Nếu nến H1 rút râu từ chối cản, quét thanh khoản hoặc nến bứt phá dứt khoát).
   - [ĐỨNG NGOÀI QUAN SÁT] (Nếu nến doji nhỏ, chưa rõ cấu trúc).
2. PHÂN TÍCH & DỰ ĐOÁN 4H: Nhận định ngắn gọn về cấu trúc đỉnh đáy và hướng đi của nến 4H tiếp theo (Dưới 200 ký tự).
"""

def main():
    print("=== BOT #1: QUÉT NẾN VÀNG & RADA TIN TỨC ===")
    df_h1, df_h4, is_4h_closed, progress_text, close_time = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    check_tp_and_sl_hit(df_h1)
    
    # 1. QUÉT ĐỘ LỆCH SỐ LIỆU TIN TỨC VỪA CÔNG BỐ
    news_items = scan_news_deviation()
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    h4_trend_str = "TĂNG (ƯU TIÊN BUY)" if h4_is_bull else "GIẢM (ƯU TIÊN SELL)"
    
    chart_h1 = generate_clean_chart(df_h1, f"[SPOT XAUUSD H1] H4 Trend: {h4_trend_str}")
    base64_img = encode_image(chart_h1)
    
    # SL AN TOÀN H1 (6.0 - 9.5 GIÁ)
    raw_atr = df_h1['ATR'].iloc[-1]
    risk_h1 = (raw_atr if not np.isnan(raw_atr) else 5.5) * 1.2
    if risk_h1 < 6.0: risk_h1 = 6.5
    elif risk_h1 > 9.5: risk_h1 = 8.5
    risk_h1 = round(risk_h1, 1)
    
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

    # ================= TRƯỜNG HỢP A: CÓ TIN ĐỎ VỪA RA (BÃO TIN) =================
    if news_items:
        latest_news = news_items[0]
        news_info_str = f"• Tin: {latest_news['title']} (Lúc {latest_news['time']})\n• Thực tế: {latest_news['actual']} vs Dự báo: {latest_news['forecast']}\n• Đánh giá: {latest_news['desc']}"
        
        tele_msg = f"🔥 [BÃO TIN ĐỎ: ĐỘ LỆCH SỐ LIỆU VĨ MÔ]\n" \
                   f"⏰ Thời gian: {time_str} (Khớp MT5)\n\n" \
                   f"📊 1. SỐ LIỆU KINH TẾ:\n{news_info_str}\n\n" \
                   f"⚡ 2. HƯỚNG LỆNH ĐỀ XUẤT: {latest_news['bias']} VÀNG (MARKET)\n" \
                   f"📍 Entry: {curr_price}\n" \
                   f"🛑 SL an toàn: {sl_h1} (-{sl_h1_pips} pips)\n" \
                   f"🎯 TP1: {tp1_h1} (+{tp1_h1_pips} pips) -> Dời hòa vốn\n" \
                   f"🚀 TP2: {tp2_h1} (+{tp2_h1_pips} pips)\n\n" \
                   f"👉 LƯU Ý: Chờ nến M15 sau tin rút râu xác nhận để vào lệnh an toàn nhất!"
        send_telegram_photo(chart_h1, tele_msg)

    # ================= TRƯỜNG HỢP B: KHÔNG CÓ TIN ĐỎ (LUÔN BÁO CÁO H1/H4) =================
    else:
        prompt = PROMPT_H1_STANDARD.replace("{H4_Trend}", h4_trend_str)\
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
        
        if "[CÓ KÈO VÀO LỆNH" in ai_verdict:
            msg_h1 = f"🚨 [KÈO H1 ĐẠT CHUẨN: {act_h1} VÀNG]\n" \
                     f"🎫 Mã vé lệnh: {t_id_h1}\n" \
                     f"⏰ Nến H1 đóng lúc: {time_str} (Khớp MT5 & TradingView)\n" \
                     f"📊 H4 Trend: {h4_trend_str}\n\n" \
                     f"📍 Vùng Entry: {e_from_h1} - {e_to_h1}\n" \
                     f"🛑 Cắt lỗ (SL): {sl_h1} (-{sl_h1_pips} pips / {risk_h1} giá)\n" \
                     f"🎯 TP1: {tp1_h1} (+{tp1_h1_pips} pips) -> Dời hòa vốn\n" \
                     f"🚀 TP2: {tp2_h1} (+{tp2_h1_pips} pips - R:R 1:2.5)\n\n" \
                     f"💡 PHÂN TÍCH:\n{ai_verdict}"
                     
            st = load_state()
            if "active_trades" not in st or not isinstance(st["active_trades"], list): st["active_trades"] = []
            st['active_trades'].append({
                "id": t_id_h1, "timeframe": "H1", "action": act_h1,
                "entry": curr_price, "sl": sl_h1, "tp1": tp1_h1, "tp2": tp2_h1,
                "sl_pips": sl_h1_pips, "tp1_pips": tp1_h1_pips, "tp2_pips": tp2_h1_pips, "tp1_hit": False
            })
            save_state(st)
        else:
            msg_h1 = f"🟢 [BÁO CÁO THỊ TRƯỜNG H1: SPOT GOLD]\n" \
                     f"⏰ Thời điểm: {time_str} (Khớp MT5 & TradingView)\n" \
                     f"📊 Xu hướng lớn H4: {h4_trend_str}\n" \
                     f"⏳ Tiến trình: {progress_text}\n" \
                     f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n" \
                     f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ KÈO ĐẸP)\n\n" \
                     f"💡 NHẬN ĐỊNH TỪ AI:\n{ai_verdict}"
                     
        send_telegram_photo(chart_h1, msg_h1)

    if os.path.exists(chart_h1): os.remove(chart_h1)

if __name__ == "__main__":
    main()

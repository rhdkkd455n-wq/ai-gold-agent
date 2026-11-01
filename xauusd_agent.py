import os
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
    caption_text = text if len(text) <= 1000 else text[:980] + "\n..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption_text}, files={'photo': photo})

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
    """LẤY NẾN SPOT XAU/USD TỪ TWELVE DATA - KHỚP 100% SÀN MT5"""
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY trong GitHub Secrets!")
        
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&apikey={TWELVE_DATA_API_KEY}"
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
    df_h1 = df.iloc[:-1]
    
    # Chỉ báo H1
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    h_l = df_h1['High'] - df_h1['Low']
    h_c = np.abs(df_h1['High'] - df_h1['Close'].shift())
    l_c = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    # Gộp nến H4 chuẩn MT5 (01, 05, 09, 13, 17, 21h)
    df_h4 = df_h1.resample('4h', offset='1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    h_l4 = df_h4['High'] - df_h4['Low']
    h_c4 = np.abs(df_h4['High'] - df_h4['Close'].shift())
    l_c4 = np.abs(df_h4['Low'] - df_h4['Close'].shift())
    df_h4['ATR'] = pd.concat([h_l4, h_c4, l_c4], axis=1).max(axis=1).rolling(14).mean()
    
    candle_open_time = df_h1.index[-1]
    candle_close_time = candle_open_time + timedelta(hours=1)
    
    is_4h_closed = candle_close_time.hour in [1, 5, 9, 13, 17, 21]
    if is_4h_closed:
        progress_text = "NẾN 4H VỪA ĐÓNG CỬA (Bắt đầu cây nến 4H mới: Giờ 1/4)"
    else:
        h_offset = ((candle_close_time.hour - 1) % 4) + 1
        progress_text = f"Cây nến 4H đang ở giờ thứ {h_offset}/4"
        
    return df_h1.tail(60), df_h4.tail(45), is_4h_closed, progress_text, candle_close_time

def generate_clean_chart(df, title_text):
    chart_path = "xauusd_clean_chart.png"
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

PROMPT_H1_STRICT = """
Bạn là Chuyên gia Giao dịch Price Action & SMC TraderViet (Gold Specialist).
Quan sát biểu đồ nến H1 Vàng Spot (XAU/USD) kèm đường EMA 21 cam.
Xu hướng chủ đạo H4: {H4_Trend}.
Giá nến H1 vừa đóng: {Price} lúc {Time_Str} (Khớp MT5).

HÃY ĐÁNH GIÁ CHÍNH XÁC VÀ DÙNG THUẬT NGỮ CHUYÊN NGHIỆP:
1. QUYẾT ĐỊNH: Bắt buộc chọn đúng 1 trong 2:
   - [CÓ KÈO VÀO LỆNH: BUY] hoặc [CÓ KÈO VÀO LỆNH: SELL] (Nếu nến H1 có rút râu từ chối cản, quét thanh khoản hoặc nến bứt phá).
   - [ĐỨNG NGOÀI QUAN SÁT] (Nếu nến đi ngang, doji nhỏ, chưa rõ cấu trúc, hoặc thị trường đóng cửa cuối tuần).

2. PHÂN TÍCH & DỰ ĐOÁN 4H: Nêu ngắn gọn cấu trúc đỉnh đáy và hướng đi của nến 4H tiếp theo (Dưới 250 ký tự).
"""

def main():
    print("=== BOT #1: ĐỒNG BỘ CHUẨN XÁC GIÁ & LỌC TÍN HIỆU ===")
    df_h1, df_h4, is_4h_closed, progress_text, close_time = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    
    check_tp_and_sl_hit(df_h1)
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    h4_trend_str = "TĂNG (ƯU TIÊN BUY)" if h4_is_bull else "GIẢM (ƯU TIÊN SELL)"
    
    # KHÓA CHẶN MỨC SL AN TOÀN CHO VÀNG H1: TỐI THIỂU 6.0 GIÁ (60 PIPS), TỐI ĐA 9.5 GIÁ
    raw_atr = df_h1['ATR'].iloc[-1]
    risk_h1 = (raw_atr if not np.isnan(raw_atr) else 5.5) * 1.2
    if risk_h1 < 6.0:  risk_h1 = 6.5  # Tuyệt đối không để SL dưới 6 giá!
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
        
    chart_h1 = generate_clean_chart(df_h1, f"[SPOT XAUUSD H1] H4 Trend: {h4_trend_str}")
    base64_img = encode_image(chart_h1)
    
    prompt = PROMPT_H1_STRICT.replace("{H4_Trend}", h4_trend_str)\
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
    
    # TRƯỜNG HỢP 1: CÓ KÈO A+ THẬT SỰ -> HIỆN ĐỦ ENTRY, SL, TP CHUẨN PIPS
    if "[CÓ KÈO VÀO LỆNH" in ai_verdict:
        msg_h1 = f"🚨 [KÈO H1 ĐẠT CHUẨN: {act_h1} VÀNG]\n" \
                 f"🎫 Mã vé lệnh: {t_id_h1}\n" \
                 f"⏰ Nến H1 đóng lúc: {time_str} (Khớp MT5)\n" \
                 f"📊 H4 Trend: {h4_trend_str}\n\n" \
                 f"📍 Vùng Entry: {e_from_h1} - {e_to_h1}\n" \
                 f"🛑 Cắt lỗ (SL): {sl_h1} (-{sl_h1_pips} pips / {risk_h1} giá an toàn)\n" \
                 f"🎯 TP1: {tp1_h1} (+{tp1_h1_pips} pips) -> Dời hòa vốn\n" \
                 f"🚀 TP2: {tp2_h1} (+{tp2_h1_pips} pips - R:R 1:2.5)\n\n" \
                 f"💡 PHÂN TÍCH TỪ AI:\n{ai_verdict}"
                 
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

    # TRƯỜNG HỢP 2: CHƯA CÓ KÈO (ĐỨNG NGOÀI) -> ẨN HẾT THÔNG SỐ ENTRY/SL/TP ĐỂ KHỎI RỐI MẮT!
    else:
        msg_h1 = f"🟢 [BÁO CÁO THỊ TRƯỜNG H1: SPOT GOLD]\n" \
                 f"⏰ Thời điểm: {time_str} (Khớp MT5)\n" \
                 f"📊 Xu hướng lớn H4: {h4_trend_str}\n" \
                 f"⏳ Tiến trình: {progress_text}\n" \
                 f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n" \
                 f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ KÈO)\n\n" \
                 f"💡 NHẬN ĐỊNH TỪ AI:\n{ai_verdict}"
                 
    send_telegram_photo(chart_h1, msg_h1)
    if os.path.exists(chart_h1): os.remove(chart_h1)

if __name__ == "__main__":
    main()

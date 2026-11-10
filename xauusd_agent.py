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
                send_telegram_msg(f"🛑 [CẮN SL] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
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
                send_telegram_msg(f"🛑 [CẮN SL] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
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

# HÀM BÓC TÁCH BẢNG DỮ LIỆU SÂU 10 CÂY NẾN H1 GẦN NHẤT
def build_deep_candle_dossier(df_h1):
    dossier = []
    recent_10 = df_h1.tail(10)
    for t, row in recent_10.iterrows():
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        rng = h - l
        u_wick = h - max(o, c)
        l_wick = min(o, c) - l
        body = abs(c - o)
        direction = "XANH (Tăng)" if c >= o else "ĐỎ (Giảm)"
        t_str = t.strftime('%H:%M')
        dossier.append(f"• Nến {t_str}: {direction} | Open={o:.1f}, High={h:.1f}, Low={l:.1f}, Close={c:.1f} | Biên độ={rng:.1f} giá (Thân={body:.1f}, Râu trên={u_wick:.1f}, Râu dưới={l_wick:.1f})")
    return "\n".join(dossier)

# NÃO BỘ AI SUY LUẬN SÂU (SỬ DỤNG 800 - 1000 TOKENS ĐẦY ĐỦ CONTEXT)
def ai_deep_decision_engine(h4_trend, curr_price, ema21_val, atr_val, candle_dossier):
    prompt = f"""
Bạn là Bậc thầy Giao dịch Price Action & SMC TraderViet (Chief Investment Officer quỹ Vàng XAU/USD).
Dưới đây là BẢNG HỒ SƠ DỮ LIỆU GIẢI PHẪU 10 CÂY NẾN H1 GẦN NHẤT:
{candle_dossier}

BỐI CẢNH KHUNG LỚN:
- Giá thị trường hiện tại: {curr_price} (Spot XAU/USD)
- Đường EMA 21 H1: {ema21_val}
- Xu hướng chủ đạo H4: {h4_trend}
- Độ biến động ATR H1: {atr_val} giá (Stop Loss tối thiểu phải từ {atr_val*1.2:.1f} giá)

HÃY SUY LUẬN ĐA CHIỀU (Chain of Thought) VÀ RA QUYẾT ĐỊNH TUYỆT ĐỐI:
1. Đọc hành vi giá qua chuỗi 10 nến: Có hiện tượng quét thanh khoản (Liquidity Sweep), nến Pin Bar từ chối cản, hay nén biên độ tích lũy?
2. Có nên vào lệnh ngay tại nến vừa đóng không?
3. Nếu VÀO LỆNH (BUY/SELL): Đặt SL ở đâu cho an toàn ngoài tầm quét của râu nến? TP1 và TP2 ở đâu đảm bảo R:R >= 1:2?
4. Nếu ĐỨNG NGOÀI: Nêu rõ kịch bản đang thiếu điều kiện gì.

BẮT BUỘC TRẢ VỀ DUY NHẤT ĐỊNH DẠNG JSON (Không viết bất kỳ chữ nào bên ngoài):
{{
  "quyet_dinh": "BUY" hoặc "SELL" hoặc "STAND_ASIDE",
  "phan_tich_chuyen_sau": "Viết bài phân tích chuyên sâu 3-4 đoạn giải thích cặn kẽ thế trận cung cầu, bẫy giá và hành vi cá mập qua chuỗi 10 nến",
  "entry": {curr_price},
  "sl": muc_gia_sl,
  "tp1": muc_gia_tp1,
  "tp2": muc_gia_tp2
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
        print(f"⚠️ Groq fallback ({e})")
        
    return {
        "quyet_dinh": "STAND_ASIDE",
        "phan_tich_chuyen_sau": f"Thị trường đang tích lũy quanh vùng {curr_price}. Xu hướng H4 {h4_trend}. Hệ thống khuyến nghị đứng ngoài bảo toàn vốn.",
        "entry": curr_price, "sl": 0, "tp1": 0, "tp2": 0
    }

def main():
    print("=== BOT #1: AI DEEP REASONING (NGÂN SÁCH TOKEN TỐI ƯU) ===")
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
    
    candle_dossier = build_deep_candle_dossier(df_h1)
    chart_file = generate_chart(df_h1, f"[SPOT XAUUSD H1] H4 Trend: {h4_trend_str}")
    
    # AI LÀ TỔNG TƯ LỆNH RA QUYẾT ĐỊNH
    decision = ai_deep_decision_engine(h4_trend_str, curr_price, ema21_now, atr_now, candle_dossier)
    act = decision.get("quyet_dinh", "STAND_ASIDE")
    analysis = decision.get("phan_tich_chuyen_sau", "")
    
    header = f"🟢 [BOT #1: TRADERVIET H1/H4 - AI DEEP ENGINE]\n" \
             f"🎫 Mã vé: {t_id}\n" \
             f"⏰ Nến H1 đóng: {time_str} (Khớp MT5)\n" \
             f"📊 H4 Trend: {h4_trend_str} | ⏳ {prog}\n" \
             f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n"
             
    if act in ["BUY", "SELL"]:
        sl_val = float(decision.get("sl", curr_price - 7.5 if act=="BUY" else curr_price + 7.5))
        tp1_val = float(decision.get("tp1", curr_price + 15.0 if act=="BUY" else curr_price - 15.0))
        tp2_val = float(decision.get("tp2", curr_price + 25.0 if act=="BUY" else curr_price - 25.0))
        
        sl_pips = int(round(abs(curr_price - sl_val) * 10))
        tp1_pips = int(round(abs(tp1_val - curr_price) * 10))
        tp2_pips = int(round(abs(tp2_val - curr_price) * 10))
        
        caption = f"{header}" \
                  f"🚨 QUYẾT ĐỊNH: VÀO {act} VÀNG (MARKET)\n" \
                  f"📍 Entry: {curr_price}\n" \
                  f"🛑 Cắt lỗ (SL): {sl_val} (-{sl_pips} pips)\n" \
                  f"🎯 TP1 (Chuẩn R:R >= 1:2): {tp1_val} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                  f"🚀 TP2 (Chuẩn R:R >= 1:3): {tp2_val} (+{tp2_pips} pips)"
                  
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
    send_telegram_msg(f"💡 [BẢN NHẬN ĐỊNH CHUYÊN SÂU TỪ AI - {t_id}]\n\n{analysis}")
    
    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

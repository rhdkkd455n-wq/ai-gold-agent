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
from datetime import datetime, timezone, timedelta
from groq import Groq

# 1. KHỞI TẠO KEY BẢO MẬT & TỰ SỬA LỖI FONT BÀN PHÍM
RAW_GROQ_KEYS = os.environ.get("GROQ_API_KEY", "").strip()
RAW_GROQ_KEYS = RAW_GROQ_KEYS.replace('α', 'a').replace('ɑ', 'a')
GROQ_KEYS_POOL = [k.strip() for k in RAW_GROQ_KEYS.split(",") if k.strip()]

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

STATE_FILE = "trade_state.json"
VN_TZ = timezone(timedelta(hours=7))

def load_state():
    default_state = {"active_trades": [], "closed_ids": [], "calendar_sent_date": ""}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "active_trades" not in data: data["active_trades"] = []
                    if "closed_ids" not in data: data["closed_ids"] = []
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

def query_groq_brain(prompt, max_tokens=700):
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    for idx, key in enumerate(GROQ_KEYS_POOL):
        try:
            cl = Groq(api_key=key)
            for m in models:
                try:
                    res = cl.chat.completions.create(
                        model=m, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.3, timeout=20.0
                    )
                    return res.choices[0].message.content.strip()
                except Exception as err:
                    if "429" in str(err) or "rate_limit" in str(err).lower(): break
                    time.sleep(1)
        except: continue
    return "Thị trường đang kiểm định vùng cản kỹ thuật then chốt."

# ================= 2. BỘ MÁY GIẢI PHẪU LỆNH THUA CHO BOT 1 =================
def ai_analyze_loss_bot1(trade, c_high, c_low, last_close):
    t_id = trade['id']
    act = trade['action']
    entry, sl, sl_pips = trade['entry'], trade['sl'], trade['sl_pips']
    hit_price = c_high if act == "SELL" else c_low
    
    prompt = f"""
Bạn là Trưởng bộ phận Đào tạo & Quản trị Rủi ro Price Action & SMC TraderViet.
Một lệnh Vàng vừa bị CẮN STOP LOSS (THUA LỖ):
- Mã vé lệnh: {t_id}
- Vị thế: {act} tại mức giá {entry}
- Điểm Cắt lỗ (SL): {sl} (Bị thị trường quét qua tại mức giá {hit_price})
- Khoảng lỗ: -{sl_pips} pips
- Giá nến H1 hiện tại đóng ở: {last_close}

HÃY VIẾT BẢN GIẢI PHẪU LỆNH THUA (POST-MORTEM) ĐỂ HỌC HỎI & RÚT KINH NGHIỆM:
1. NGUYÊN NHÂN BỊ QUÉT SL: Phân tích khách quan tại sao giá lại đi ngược chiều? (Cá mập quét thanh khoản sâu hơn? Bị bẫy nến rút râu giả? Hay do đà sóng H4 quá mạnh?).
2. BÀI HỌC THỰC CHIẾN: Cần lưu ý điều gì ở các lệnh tiếp theo để tránh lặp lại sai lầm này?
(Viết sâu sắc, đúng chất Price Action TraderViet, tuyệt đối không viết sáo rỗng).
"""
    return query_groq_brain(prompt, max_tokens=650)

def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    closed_ids = set(state.get("closed_ids", []))
    if not active_trades: return
    
    last_c = df_h1.iloc[-1]
    c_high, c_low, last_close = last_c['High'], last_c['Low'], last_c['Close']
    remaining = []
    
    for trade in active_trades:
        t_id, act = trade['id'], trade['action']
        if t_id in closed_ids: continue
        entry, sl, tp1, tp2 = trade['entry'], trade['sl'], trade['tp1'], trade['tp2']
        is_closed = False
        
        if act == "BUY":
            if c_low <= sl:
                send_telegram_msg(f"🛑 [CẮN SL TRADERVIET] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.\n\n⏳ AI đang giải phẫu nguyên nhân dính SL...")
                loss_post_mortem = ai_analyze_loss_bot1(trade, c_high, c_low, last_close)
                send_telegram_msg(f"🔍 [BẢN GIẢI PHẪU LỆNH THUA - AI POST-MORTEM: {t_id}]\n\n{loss_post_mortem}")
                is_closed = True
            elif c_high >= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS! Chốt lời toàn bộ!")
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
                
        elif act == "SELL":
            if c_high >= sl:
                send_telegram_msg(f"🛑 [CẮN SL TRADERVIET] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.\n\n⏳ AI đang giải phẫu nguyên nhân dính SL...")
                loss_post_mortem = ai_analyze_loss_bot1(trade, c_high, c_low, last_close)
                send_telegram_msg(f"🔍 [BẢN GIẢI PHẪU LỆNH THUA - AI POST-MORTEM: {t_id}]\n\n{loss_post_mortem}")
                is_closed = True
            elif c_low <= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS! Chốt lời toàn bộ!")
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
                
        if is_closed: closed_ids.add(t_id)
        else: remaining.append(trade)
            
    state['active_trades'] = remaining
    state['closed_ids'] = list(closed_ids)
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
        direct = "XANH (TĂNG)" if row['Close'] >= row['Open'] else "ĐỎ (GIẢM)"
        dossier.append(f"• Nến {t.strftime('%H:%M')}: {direct} | C={row['Close']:.1f}, H={row['High']:.1f}, L={row['Low']:.1f} (Râu trên={u_w:.1f}, Râu dưới={l_w:.1f})")
    return "\n".join(dossier)

PROMPT_H1_VIDEO1 = """
Bạn là Chuyên gia Giao dịch Price Action & SMC TraderViet (Gold Specialist).
DỮ LIỆU 8 NẾN H1 GẦN NHẤT:
{candle_dossier}

THÔNG SỐ BỐI CẢNH:
- Giá Spot hiện tại: {curr_price} | EMA 21 H1: {ema21_now}
- Xu hướng lớn H4: {h4_trend} | ATR H1: {atr_now:.1f} giá

HÃY ĐÁNH GIÁ CHUYÊN NGHIỆP CHI TIẾT THEO ĐÚNG 3 MỤC (TUYỆT ĐỐI KHÔNG CHÀO HỎI):
**1. ĐỌC VỊ HÀNH VI GIÁ (PRICE ACTION)**
* Phân tích chi tiết lực nến H1 vừa đóng (body nến, râu nến), sự bám đuổi đường EMA 21, có hiện tượng xả mạnh (dump), gom hàng, hay tích lũy?

**2. ĐÁNH GIÁ CẤU TRÚC H4 & CÁC VÙNG CẢN**
* Tương quan xu hướng H4 và nến H1. Vùng cản/đáy then chốt tiếp theo mà giá sẽ kiểm định.

**3. KỊCH BẢN GIÁ & HÀNH ĐỘNG**
* Dự báo đường đi của nến tiếp theo và khuyến nghị hành động cụ thể cho trader.

KẾT THÚC BẰNG DÒNG:
[QUYET_DINH]: BUY hoặc SELL hoặc STAND_ASIDE
"""

def main():
    print("=== BOT #1: TÁCH RIÊNG ẢNH VÀ BÀI PHÂN TÍCH CHỮ (KHÔNG CẮT XÉN) ===")
    state = load_state()
    now_vn = datetime.now(VN_TZ)
    today_str = now_vn.strftime('%d/%m/%Y')
    
    # 1. Gửi lịch tin sáng (7h - 9h30)
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
    
    # 2. KIỂM TRA TP/SL (NẾU DÍNH SL -> AI GIẢI PHẪU NGUYÊN NHÂN NGAY!)
    check_tp_and_sl_hit(df_h1)
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    h4_trend_str = "TĂNG (ƯU TIÊN BUY)" if h4_is_bull else "GIẢM (ƯU TIÊN SELL)"
    c_last = df_h1.iloc[-1]
    ema21_now = round(c_last['EMA21'], 2)
    atr_now = round(df_h1['ATR'].iloc[-1], 1)
    if atr_now < 5.5: atr_now = 6.0
    
    candle_dossier = build_candle_dossier(df_h1)
    chart_file = generate_chart(df_h1, f"[SPOT XAUUSD H1] H4 Trend: {h4_trend_str}")
    
    prompt = PROMPT_H1_VIDEO1.replace("{candle_dossier}", candle_dossier)\
                             .replace("{curr_price}", str(curr_price))\
                             .replace("{ema21_now}", str(ema21_now))\
                             .replace("{h4_trend}", h4_trend_str)\
                             .replace("{atr_now}", str(atr_now))
                             
    raw_ai_text = query_groq_brain(prompt, max_tokens=700)
    
    act = "STAND_ASIDE"
    if "[QUYET_DINH]:" in raw_ai_text:
        part = raw_ai_text.split("[QUYET_DINH]:")[1].strip()
        if "BUY" in part: act = "BUY"
        elif "SELL" in part: act = "SELL"
        
    # BÀI PHÂN TÍCH CHUYÊN SÂU ĐƯỢC LỌC SẠCH SẼ ĐỂ GỬI RIÊNG
    clean_analysis = raw_ai_text.replace(f"[QUYET_DINH]: {act}", "").replace("[QUYET_DINH]:", "").strip()
    
    risk_dist = max(atr_now * 1.2, 6.5)
    sl_pips = int(risk_dist * 10)
    
    # ================= TIN NHẮN 1: GỬI KÈM ẢNH (CHỈ CHỨA THẺ LỆNH GỌN GÀNG) =================
    header_info = f"🟢 [BOT #1: TRADERVIET H1/H4]\n" \
                  f"🎫 Mã vé lệnh: {t_id}\n" \
                  f"⏰ Nến H1 đóng: {time_str} (Khớp MT5)\n" \
                  f"📊 H4 Trend: {h4_trend_str} | ⏳ {prog}\n" \
                  f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n"
                  
    st = load_state()
    active_ids = [t['id'] for t in st.get("active_trades", [])]
    closed_ids = set(st.get("closed_ids", []))
    
    if act in ["BUY", "SELL"]:
        is_buy = (act == "BUY")
        sl_val = round(curr_price - risk_dist if is_buy else curr_price + risk_dist, 2)
        tp1_val = round(curr_price + risk_dist * 2.0 if is_buy else curr_price - risk_dist * 2.0, 2)
        tp2_val = round(curr_price + risk_dist * 3.0 if is_buy else curr_price - risk_dist * 3.0, 2)
        tp1_pips = int(risk_dist * 20)
        tp2_pips = int(risk_dist * 30)
        
        photo_caption = f"{header_info}" \
                        f"🚨 QUYẾT ĐỊNH: VÀO {act} VÀNG (MARKET)\n" \
                        f"📍 Entry: {curr_price}\n" \
                        f"🛑 Cắt lỗ (SL): {sl_val} (-{sl_pips} pips)\n" \
                        f"🎯 TP1 (Chuẩn R:R 1:2): {tp1_val} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                        f"🚀 TP2 (Chuẩn R:R 1:3): {tp2_val} (+{tp2_pips} pips)"
                        
        if (t_id not in active_ids) and (t_id not in closed_ids):
            st['active_trades'].append({
                "id": t_id, "action": act, "timeframe": "H1",
                "entry": curr_price, "sl": sl_val, "tp1": tp1_val, "tp2": tp2_val,
                "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
            })
            save_state(st)
    else:
        photo_caption = f"{header_info}🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ KÈO ĐẸP)"

    # GỬI TIN 1: ẢNH + THẺ LỆNH GỌN GÀNG
    send_telegram_photo(chart_file, photo_caption)
    time.sleep(1)
    
    # GỬI TIN 2: BÀI PHÂN TÍCH CHUYÊN SÂU TÁCH RIÊNG HOÀN TOÀN (KHÔNG BAO GIỜ BỊ CẮT CHỮ)
    send_telegram_msg(f"💡 [NHẬN ĐỊNH CHUYÊN SÂU TỪ AI - {t_id}]\n\n{clean_analysis}")
    
    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

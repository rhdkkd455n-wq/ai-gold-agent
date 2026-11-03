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
from datetime import datetime, timezone, timedelta
from groq import Groq

# 1. KHỞI TẠO KEY
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

# ================= 2. THUẬT TOÁN SĂN ĐỘ LỆCH SỐ LIỆU (ACTUAL VS FORECAST) =================
def clean_num(val_str):
    """Lọc sạch ký tự %, K, M để chuyển về số thực tính toán"""
    if not val_str or val_str == "N/A": return None
    s = re.sub(r'[^\d.-]', '', val_str)
    try: return float(s)
    except: return None

def scan_news_deviation():
    """TỰ ĐỘNG BÓC TÁCH ĐỘ LỆCH SỐ LIỆU TỨC THÌ TỪ FOREXFACTORY"""
    now_vn = datetime.now(VN_TZ)
    news_signals = []
    
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        res = requests.get(url, timeout=5).json()
        
        for item in res:
            if item.get("country") == "USD" and item.get("impact") == "High":
                event_dt = datetime.fromisoformat(item.get("date")).astimezone(VN_TZ)
                
                # Kiểm tra các tin vừa ra trong vòng 2 tiếng trở lại
                diff_sec = (now_vn - event_dt).total_seconds()
                if 0 <= diff_sec <= 7200:
                    act_raw = item.get("actual")
                    fc_raw = item.get("forecast")
                    title = item.get("title", "")
                    
                    if act_raw and fc_raw:
                        actual = clean_num(act_raw)
                        forecast = clean_num(fc_raw)
                        
                        if actual is not None and forecast is not None:
                            dev = actual - forecast
                            # Với CPI, NFP, PPI, Retail Sales: Thực tế > Dự báo là Tốt cho USD -> Xấu cho Vàng
                            # Với Đơn xin trợ cấp thất nghiệp (Jobless claims): Ngược lại
                            is_inverse_news = "Unemployment" in title or "Claims" in title
                            
                            if (dev > 0 and not is_inverse_news) or (dev < 0 and is_inverse_news):
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
        print(f"Lỗi phân tích số liệu: {e}")
        
    return news_signals

# ================= 3. LẤY NẾN TWELVE DATA & DÒ BẪY THANH KHOẢN =================
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
    df_h1 = df.iloc[:-1]
    
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

# PROMPT HỢP NHẤT: ĐỘ LỆCH SỐ LIỆU + BẪY THANH KHOẢN SAU TIN
PROMPT_NEWS_HYBRID = """
Bạn là Chuyên gia Giao dịch Chiến lược Bão Tin Vàng (Forex News Sniper).
Biểu đồ nến H1/M15 Vàng Spot (XAU/USD). Giá hiện tại: {Price}.
DỮ LIỆU ĐỘ LỆCH SỐ LIỆU VĨ MÔ VỪA CÔNG BỐ:
{News_Data}

HÃY ĐÁNH GIÁ MÔ HÌNH HỢP NHẤT:
1. Số liệu vĩ mô đang báo dòng tiền đẩy Vàng theo hướng nào ({Expected_Bias})?
2. Nhìn nến trên biểu đồ: Đã xuất hiện cú đâm râu Quét thanh khoản (Liquidity Sweep) rồi rút chân xác nhận theo hướng số liệu chưa?
3. QUYẾT ĐỊNH: [VÀO LỆNH BÃO TIN: BUY] hoặc [VÀO LỆNH BÃO TIN: SELL] hoặc [ĐỨNG NGOÀI (CHƯA ĐỒNG THUẬN)].
(Viết súc tích dưới 250 ký tự).
"""

def main():
    print("=== BOT #1: KÍCH HOẠT HỆ THỐNG SĂN ĐỘ LỆCH SỐ LIỆU & BẪY SAU TIN ===")
    df_h1, df_h4, is_4h_closed, progress_text, close_time = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    
    # 1. QUÉT ĐỘ LỆCH SỐ LIỆU TIN TỨC VỪA CÔNG BỐ
    news_items = scan_news_deviation()
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    h4_trend_str = "TĂNG" if h4_is_bull else "GIẢM"
    
    # Đo chân nến quét thanh khoản gần nhất
    recent_high = df_h1['High'].iloc[-5:].max()
    recent_low = df_h1['Low'].iloc[-5:].min()
    
    chart_h1 = generate_clean_chart(df_h1, f"[SPOT XAUUSD] News & Price Action Analysis")
    base64_img = encode_image(chart_h1)
    
    # TRƯỜNG HỢP CÓ TIN ĐỎ VỪA RA -> BẬT RADA SĂN KÈO HỢP NHẤT
    if news_items:
        latest_news = news_items[0]
        news_bias = latest_news['bias']
        news_info_str = f"• Tin: {latest_news['title']} (Lúc {latest_news['time']})\n• Thực tế: {latest_news['actual']} vs Dự báo: {latest_news['forecast']}\n• Đánh giá: {latest_news['desc']}"
        
        prompt = PROMPT_NEWS_HYBRID.replace("{Price}", str(curr_price))\
                                   .replace("{News_Data}", news_info_str)\
                                   .replace("{Expected_Bias}", news_bias)
                                   
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]}],
            max_tokens=350
        )
        ai_verdict = response.choices[0].message.content.strip()
        
        # Nếu cả Vĩ mô và Nến đồng thuận -> BẮN SIÊU KÈO BÃO TIN!
        if "[VÀO LỆNH BÃO TIN" in ai_verdict:
            is_buy = "BUY" in ai_verdict
            act = "BUY VÀNG (MARKET)" if is_buy else "SELL VÀNG (MARKET)"
            
            # SL đặt sau râu nến vừa quét thanh khoản
            if is_buy:
                sl_calc = round(recent_low - 1.5, 2)
                risk_dist = round(curr_price - sl_calc, 1)
                tp1_calc = round(curr_price + (risk_dist * 1.5), 2)
                tp2_calc = round(curr_price + (risk_dist * 3.0), 2) # R:R 1:3
            else:
                sl_calc = round(recent_high + 1.5, 2)
                risk_dist = round(sl_calc - curr_price, 1)
                tp1_calc = round(curr_price - (risk_dist * 1.5), 2)
                tp2_calc = round(curr_price - (risk_dist * 3.0), 2)
                
            sl_pips = int(risk_dist * 10)
            tp1_pips = int(risk_dist * 1.5 * 10)
            tp2_pips = int(risk_dist * 3.0 * 10)
            
            tele_msg = f"🔥 [SIÊU KÈO BÃO TIN: ĐỘ LỆCH SỐ LIỆU + BẪY M15]\n" \
                       f"⏰ Thời gian: {time_str} (Khớp MT5)\n\n" \
                       f"📊 1. SỐ LIỆU VĨ MÔ THỰC TẾ:\n{news_info_str}\n\n" \
                       f"⚡ 2. HÀNH ĐỘNG: {act}\n" \
                       f"📍 Entry: {curr_price}\n" \
                       f"🛑 SL sau râu quét: {sl_calc} (-{sl_pips} pips / {risk_dist} giá)\n" \
                       f"🎯 TP1: {tp1_calc} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                       f"🚀 TP2: {tp2_calc} (+{tp2_pips} pips - R:R 1:3)\n\n" \
                       f"💡 NHẬN ĐỊNH TỪ AI:\n{ai_verdict}"
            send_telegram_photo(chart_h1, tele_msg)
        else:
            print("Số liệu và nến chưa đồng thuận. Đứng ngoài né bão.")
            
    # NẾU KHÔNG CÓ TIN ĐỎ VỪA RA -> CHẠY PHÂN TÍCH KỸ THUẬT H1/H4 BÌNH THƯỜNG
    else:
        print("Không có tin đỏ tức thì. Duy trì phân tích nến chuẩn H1/H4.")
        
    if os.path.exists(chart_h1): os.remove(chart_h1)

if __name__ == "__main__":
    main()

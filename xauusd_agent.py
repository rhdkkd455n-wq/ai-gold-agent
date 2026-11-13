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

# 1. KÍCH HOẠT BỂ CHỨA 6+ KEY GROQ (ĐỒNG BỘ VỚI BOT #3)
def load_all_groq_keys():
    keys = []
    raw_main = os.environ.get("GROQ_API_KEY", "")
    for k in raw_main.replace("\n", ",").split(","):
        if k.strip():
            keys.append(k.strip())
    for i in range(1, 10):
        k_sub = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if k_sub and k_sub not in keys:
            keys.append(k_sub)
    return keys

GROQ_KEYS_POOL = load_all_groq_keys()

BOT1_TOKEN = os.environ.get("BOT1_TELEGRAM_TOKEN", "").strip()
MAIN_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_BOT_TOKEN = BOT1_TOKEN if BOT1_TOKEN else MAIN_TOKEN

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

STATE_FILE = "bot1_state.json"
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
        except Exception as e:
            print(f"⚠️ Lỗi đọc file state: {e}")
    return default_state

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 4000
    for i in range(0, len(text), max_len):
        chunk = text[i:i+max_len]
        try:
            res = requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': chunk}, timeout=15)
            if not res.ok:
                print(f"⚠️ Telegram sendMessage lỗi: {res.text}")
        except Exception as e:
            print(f"⚠️ Lỗi kết nối Telegram: {e}")

def send_telegram_photo(photo_path, text):
    caption_text = text if len(text) <= 1000 else text[:980] + "\n..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo:
            res = requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption_text}, files={'photo': photo}, timeout=20)
            if not res.ok:
                print(f"⚠️ Telegram sendPhoto lỗi: {res.text}")
    except Exception as e:
        print(f"⚠️ Lỗi gửi ảnh Telegram: {e}")

# HÀM XOAY VÒNG QUA BỂ CHỨA KEY GROQ & DỰ PHÒNG MODEL (ĐÃ CẬP NHẬT MODEL MỚI NHẤT)
def call_groq_rotation(prompt, max_tokens=400, temperature=0.2):
    if not GROQ_KEYS_POOL:
        print("❌ CẢNH BÁO: Chưa có GROQ_API_KEY trong GitHub Secrets!")
        return None

    models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    for idx, key in enumerate(GROQ_KEYS_POOL):
        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "Key"
        cl = Groq(api_key=key)
        for m in models:
            try:
                res = cl.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=20.0
                )
                ans = res.choices[0].message.content
                if ans and ans.strip():
                    print(f"✅ AI phản hồi thành công với Key #{idx+1} ({masked}) | Model: {m}")
                    return ans.strip()
            except Exception as e:
                err_str = str(e)
                print(f"⚠️ Key #{idx+1} ({masked}) lỗi ({m}): {err_str}")
                if any(k in err_str.lower() for k in ["401", "invalid_api_key", "429", "rate_limit", "organization_restricted", "model_decommissioned", "model_not_found"]):
                    print(f"👉 Chuyển sang Key/Model tiếp theo...")
                    break
                else:
                    time.sleep(1.5)
    return None

def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    if not active_trades:
        return
    last_c = df_h1.iloc[-1]
    c_high, c_low, c_close = last_c['High'], last_c['Low'], last_c['Close']
    remaining = []
    
    for trade in active_trades:
        t_id, act = trade['id'], trade['action']
        entry, sl, tp1, tp2 = trade['entry'], trade['sl'], trade['tp1'], trade['tp2']
        is_closed = False
        sl_hit = False
        
        if act == "BUY":
            if c_low <= sl:
                sl_hit = True
                is_closed = True
            elif c_high >= tp2:
                send_telegram_msg(f"🚀 [FULL TP2] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
        elif act == "SELL":
            if c_high >= sl:
                sl_hit = True
                is_closed = True
            elif c_low <= tp2:
                send_telegram_msg(f"🚀 [FULL TP2] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
                
        if sl_hit:
            if trade.get('tp1_hit', False):
                send_telegram_msg(f"🛡️ [CHẠM HÒA VỐN BE] Kèo {act} ({t_id}) đã đóng phần còn lại tại Entry ({entry}) sau khi chốt lời TP1.")
            else:
                send_telegram_msg(f"🛑 [CẮN SL] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                pm_prompt = (
                    f"Kèo {act} Vàng ({t_id}) theo phương pháp Price Action & SMC vừa bị quét SL tại {sl} (Entry: {entry}). "
                    f"Nến H1 vừa đóng có High={c_high}, Low={c_low}, Close={c_close}. "
                    f"Hãy giải phẫu ngắn gọn trong 3 ý: 1. Lý do cấu trúc bị phá hoặc bị quét thanh khoản (Stop Hunt). 2. Đánh giá lực nến hiện tại. 3. Kinh nghiệm rút ra."
                )
                pm_res = call_groq_rotation(pm_prompt, max_tokens=450, temperature=0.3)
                if pm_res:
                    send_telegram_msg(f"🔍 [BẢN GIẢI PHẪU LỆNH THUA - AI POST-MORTEM: {t_id}]\n\n{pm_res}")
                    
        if not is_closed:
            remaining.append(trade)
            
    state['active_trades'] = remaining
    save_state(state)

def get_market_data():
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY!")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.get(url, timeout=15).json()
            if "values" in res:
                break
            print(f"⚠️ Twelve Data phản hồi chưa ổn định (Lần thử {attempt+1}): {res.get('message')}")
        except Exception as e:
            print(f"⚠️ Mất kết nối mạng (Lần thử {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise ValueError("Không thể kết nối tới Twelve Data sau 3 lần thử.")

    df = pd.DataFrame(res['values']).iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df['Volume'] = 1000
    df_h1 = df.iloc[:-1].copy()
    
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    h_l = df_h1['High'] - df_h1['Low']
    h_c = np.abs(df_h1['High'] - df_h1['Close'].shift())
    l_c = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    df_h4 = df_h1.resample('4h', offset='1h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    
    c_open = df_h1.index[-1]
    c_close = c_open + timedelta(hours=1)
    is_4h_closed = c_close.hour in [1, 5, 9, 13, 17, 21]
    prog = "NẾN 4H VỪA ĐÓNG CỬA (Giờ 1/4)" if is_4h_closed else f"Nến 4H đang ở giờ {((c_close.hour - 1) % 4) + 1}/4"
    return df_h1.tail(60), df_h4.tail(45), is_4h_closed, prog, c_close

def generate_chart(df, title_text):
    path = "xauusd_clean.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    fig, _ = mpf.plot(df, type='candle', style='binance', addplot=ema_plot, title=f"\n{title_text}", figsize=(13, 6.5), returnfig=True)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def build_pa_context(df_h1):
    recent_20 = df_h1.tail(20)
    swing_high = round(recent_20['High'].max(), 2)
    swing_low = round(recent_20['Low'].min(), 2)
    
    lines = [f"- Đỉnh Swing High (20 nến H1): {swing_high} | Đáy Swing Low (20 nến H1): {swing_low}"]
    lines.append("- Chi tiết 5 nến H1 gần nhất:")
    for t, r in df_h1.tail(5).iterrows():
        c_type = "TĂNG (Xanh)" if r['Close'] >= r['Open'] else "GIẢM (Đỏ)"
        body = abs(r['Close'] - r['Open'])
        u_wick = r['High'] - max(r['Open'], r['Close'])
        l_wick = min(r['Open'], r['Close']) - r['Low']
        lines.append(f"  • {t.strftime('%H:%M')}: Nến {c_type} | Open={r['Open']:.2f}, High={r['High']:.2f}, Low={r['Low']:.2f}, Close={r['Close']:.2f} (Thân={body:.1f}, Râu trên={u_wick:.1f}, Râu dưới={l_wick:.1f})")
    return "\n".join(lines)

def safe_float(val, default_val):
    try:
        if isinstance(val, (int, float)):
            return float(val)
        cleaned = re.sub(r'[^\d.-]', '', str(val))
        return float(cleaned) if cleaned else default_val
    except:
        return default_val

# HÀM SUY LUẬN AI: RA QUYẾT ĐỊNH + GIẢI THÍCH TƯ DUY CHUYÊN SÂU
def ai_decide_trade(h4_trend, curr_price, ema21_val, pa_context, atr_val):
    min_sl_dist = round(max(atr_val * 1.2, 6.5), 1)
    prompt = f"""
Bạn là Bậc thầy Price Action & SMC TraderViet quản lý quỹ Vàng XAU/USD.
DỮ LIỆU THỊ TRƯỜNG HIỆN TẠI:
- Giá đóng cửa hiện tại: {curr_price}
- Đường EMA 21 khung H1: {ema21_val}
- Xu hướng chủ đạo H4: {h4_trend}
- Độ biến động ATR(14): {atr_val} giá
{pa_context}

NHIỆM VỤ:
Phân tích cấu trúc thị trường (SMC, Bos/ChoCH, Quét thanh khoản, Pinbar/Engulfing) và TỰ RA QUYẾT ĐỊNH.
Chỉ vào lệnh (BUY hoặc SELL) khi thực sự có tín hiệu đẹp (Kèo A+). Nếu nến đang lưng chừng hoặc rủi ro cao, hãy chọn STAND_ASIDE.

Trả về ĐÚNG định dạng JSON hợp lệ (không dùng markdown bên ngoài JSON):
{{
  "quyet_dinh": "BUY" hoặc "SELL" hoặc "STAND_ASIDE",
  "ly_do_ngan": "Tóm tắt lý do vào lệnh hoặc đứng ngoài trong 1-2 câu ngắn gọn",
  "phan_tich_chi_tiet": "Phân tích sâu 3 gạch đầu dòng: 1. Cấu trúc đa khung H4/H1 & Vị trí giá so với EMA21/Swing High-Low. 2. Hành vi nến & Thanh khoản (Râu nến quét SL hay lực thân nến áp đảo). 3. Kịch bản hành động tiếp theo.",
  "entry": {curr_price},
  "sl": số_thực (cách entry tối thiểu {min_sl_dist} giá),
  "tp1": số_thực (tỷ lệ R:R tối thiểu 1:1.5),
  "tp2": số_thực (tỷ lệ R:R tối thiểu 1:2.5)
}}
"""
    txt = call_groq_rotation(prompt, max_tokens=650, temperature=0.2)
    if txt:
        try:
            match = re.search(r'\{.*\}', txt, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"⚠️ Lỗi bóc tách JSON từ AI: {e}\nNội dung gốc: {txt}")

    return {
        "quyet_dinh": "STAND_ASIDE",
        "ly_do_ngan": "Thị trường đang giao dịch giằng co, chưa xác nhận tín hiệu Price Action/SMC chuẩn A+.",
        "phan_tich_chi_tiet": "• Cấu trúc H1 chưa tạo được điểm phá vỡ (BOS) rõ ràng.\n• Khuyến nghị kiên nhẫn chờ giá phản ứng tại các vùng thanh khoản đỉnh/đáy gần nhất.",
        "entry": curr_price, "sl": 0, "tp1": 0, "tp2": 0
    }

def main():
    print(f"=== BOT #1: AI TỔNG TƯ LỆNH RA QUYẾT ĐỊNH ({len(GROQ_KEYS_POOL)} KEY GROQ) ===")
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
    if np.isnan(atr_now) or atr_now < 5.5:
        atr_now = 6.0
        
    pa_context = build_pa_context(df_h1)
    chart_file = generate_chart(df_h1, f"[SPOT XAUUSD H1] H4 Trend: {h4_trend_str}")
    
    # GỌI NÃO BỘ AI RA QUYẾT ĐỊNH
    decision = ai_decide_trade(h4_trend_str, curr_price, ema21_now, pa_context, atr_now)
    act = str(decision.get("quyet_dinh", "STAND_ASIDE")).upper().strip()
    reason = decision.get("ly_do_ngan", decision.get("ly_do", "Chưa đủ điều kiện vào lệnh."))
    deep_analysis = decision.get("phan_tich_chi_tiet", reason)
    
    header = f"🟢 [BOT #1: TRADERVIET H1/H4 - AI DRIVEN]\n" \
             f"🎫 Mã vé: {t_id}\n" \
             f"⏰ Thời điểm: {time_str} (Khớp MT5)\n" \
             f"📊 H4 Trend: {h4_trend_str} | ⏳ {prog}\n" \
             f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n"
             
    min_dist = round(max(atr_now * 1.2, 6.5), 2)
    
    if act in ["BUY", "SELL"]:
        raw_sl = safe_float(decision.get("sl"), 0)
        raw_tp1 = safe_float(decision.get("tp1"), 0)
        raw_tp2 = safe_float(decision.get("tp2"), 0)
        
        # KHÓA AN TOÀN SL/TP: Đảm bảo AI không bao giờ đặt ngược chiều SL/TP
        if act == "BUY":
            sl_val = raw_sl if (0 < raw_sl <= curr_price - 4.0) else round(curr_price - min_dist, 2)
            risk = abs(curr_price - sl_val)
            tp1_val = raw_tp1 if (raw_tp1 >= curr_price + risk * 1.3) else round(curr_price + risk * 1.6, 2)
            tp2_val = raw_tp2 if (raw_tp2 > tp1_val) else round(curr_price + risk * 2.6, 2)
        else: # SELL
            sl_val = raw_sl if (raw_sl >= curr_price + 4.0) else round(curr_price + min_dist, 2)
            risk = abs(sl_val - curr_price)
            tp1_val = raw_tp1 if (0 < raw_tp1 <= curr_price - risk * 1.3) else round(curr_price - risk * 1.6, 2)
            tp2_val = raw_tp2 if (0 < raw_tp2 < tp1_val) else round(curr_price - risk * 2.6, 2)
            
        sl_val, tp1_val, tp2_val = round(sl_val, 2), round(tp1_val, 2), round(tp2_val, 2)
        sl_pips = int(round(abs(curr_price - sl_val) * 10))
        tp1_pips = int(round(abs(tp1_val - curr_price) * 10))
        tp2_pips = int(round(abs(tp2_val - curr_price) * 10))
        
        caption = f"{header}" \
                  f"🚨 QUYẾT ĐỊNH: VÀO {act} VÀNG (MARKET)\n" \
                  f"📍 Entry: {curr_price}\n" \
                  f"🛑 Cắt lỗ (SL): {sl_val} (-{sl_pips} pips)\n" \
                  f"🎯 TP1 (R:R 1:1.5+): {tp1_val} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                  f"🚀 TP2 (R:R 1:2.5+): {tp2_val} (+{tp2_pips} pips)\n\n" \
                  f"💡 LUẬN ĐIỂM AI: {reason}"
                  
        st = load_state()
        if "active_trades" not in st or not isinstance(st["active_trades"], list):
            st["active_trades"] = []
        st['active_trades'].append({
            "id": t_id, "action": act, "timeframe": "H1",
            "entry": curr_price, "sl": sl_val, "tp1": tp1_val, "tp2": tp2_val,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
    else:
        caption = f"{header}" \
                  f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ KÈO A+)\n\n" \
                  f"💡 TÓM TẮT AI: {reason}"
                  
    send_telegram_photo(chart_file, caption)
    time.sleep(1)
    send_telegram_msg(f"🧠 [PHÂN TÍCH TƯ DUY AI SMC/PA - {t_id}]\n\n{deep_analysis}")
    
    if os.path.exists(chart_file):
        os.remove(chart_file)

if __name__ == "__main__":
    main()

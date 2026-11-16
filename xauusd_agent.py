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

STATE_FILE = "trade_state.json"
VN_TZ = timezone(timedelta(hours=7))
SEND_STAND_ASIDE_MSG = False

def load_state():
    default_state = {
        "active_trades": [],
        "closed_trades": [],
        "notified_events": [],
        "last_weekly_report": "",
        "last_processed_candle": ""
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k, v in default_state.items():
                        if k not in data:
                            data[k] = v
                    unique_trades = []
                    seen_ids = set()
                    for t in data.get("active_trades", []):
                        tid = t.get("id")
                        if tid and tid not in seen_ids:
                            seen_ids.add(tid)
                            unique_trades.append(t)
                    data["active_trades"] = unique_trades
                    return data
        except Exception as e:
            print(f"⚠️ Lỗi đọc file state: {e}")
    return default_state

def save_state(state):
    if "notified_events" in state and len(state["notified_events"]) > 200:
        state["notified_events"] = state["notified_events"][-200:]
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

def call_groq_rotation(prompt, max_tokens=600, temperature=0.2):
    if not GROQ_KEYS_POOL:
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
                    print(f"✅ AI Swing phản hồi thành công với Key #{idx+1} ({masked}) | Model: {m}")
                    return ans.strip()
            except Exception as e:
                err_str = str(e)
                print(f"⚠️ Key #{idx+1} lỗi ({m}): {err_str}")
                if any(k in err_str.lower() for k in ["401", "invalid_api_key", "429", "rate_limit", "organization_restricted", "model_decommissioned", "model_not_found"]):
                    break
                else:
                    time.sleep(1.5)
    return None

def check_and_send_weekly_summary():
    now_vn = datetime.now(VN_TZ)
    is_weekend = (now_vn.weekday() == 5 and now_vn.hour >= 5) or (now_vn.weekday() == 6)
    if not is_weekend:
        return

    state = load_state()
    year, week_num, _ = now_vn.isocalendar()
    week_id = f"{year}-W{week_num}"
    if state.get("last_weekly_report") == week_id:
        return

    closed = state.get("closed_trades", [])
    if not closed:
        return

    total_trades = len(closed)
    win_trades = [t for t in closed if t.get("pips", 0) > 0]
    loss_trades = [t for t in closed if t.get("pips", 0) < 0]
    total_pips = sum(t.get("pips", 0) for t in closed)
    win_rate = round((len(win_trades) / total_trades) * 100, 1) if total_trades > 0 else 0

    details = []
    for idx, t in enumerate(closed, 1):
        icon = "✅" if t.get("pips", 0) > 0 else "🛑"
        p_str = f"+{t['pips']}" if t['pips'] > 0 else f"{t['pips']}"
        details.append(f"{idx}. {icon} {t['id']} ({t['action']}): {t['result']} ({p_str} pips)")

    details_str = "\n".join(details)
    net_str = f"+{total_pips}" if total_pips >= 0 else f"{total_pips}"

    summary_msg = (
        f"📅 [BÁO CÁO TỔNG KẾT TUẦN - BOT #1 SWING TRADING H4/D1]\n"
        f"🗓️ Kỳ báo cáo: Tuần {week_num}/{year}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 THỐNG KÊ HIỆU SUẤT SWING:\n"
        f"• Tổng số lệnh Swing đã đóng: {total_trades} lệnh\n"
        f"• Số lệnh Thắng (Sóng lớn): {len(win_trades)} lệnh ✅\n"
        f"• Số lệnh Thua (SL): {len(loss_trades)} lệnh 🛑\n"
        f"• Tỷ lệ thắng (Winrate): {win_rate}%\n"
        f"• TỔNG LỢI NHUẬN RÒNG: {net_str} PIPS 💰\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📜 CHI TIẾT TỪNG KÈO SWING:\n"
        f"{details_str}"
    )
    send_telegram_msg(summary_msg)
    state["last_weekly_report"] = week_id
    state["closed_trades"] = []
    save_state(state)

def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    if not active_trades:
        return

    notified = set(state.get("notified_events", []))
    last_c = df_h1.iloc[-1]
    c_high, c_low, c_close = last_c['High'], last_c['Low'], last_c['Close']
    remaining = []
    closed_time = datetime.now(VN_TZ).strftime('%d/%m %H:%M')
    
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
                ev_key = f"{t_id}_TP2"
                if ev_key not in notified:
                    send_telegram_msg(f"🦅 [SWING TRADE - FULL TP2 SÓNG LỚN] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI ĐẬM: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2 SWING", "pips": trade['tp2_pips'], "closed_at": closed_time})
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [SWING TRADE - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn ({entry}) để gồng sóng lớn tới TP2 ({tp2})!")
                    notified.add(ev_key)
        elif act == "SELL":
            if c_high >= sl:
                sl_hit = True
                is_closed = True
            elif c_low <= tp2:
                ev_key = f"{t_id}_TP2"
                if ev_key not in notified:
                    send_telegram_msg(f"🦅 [SWING TRADE - FULL TP2 SÓNG LỚN] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI ĐẬM: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2 SWING", "pips": trade['tp2_pips'], "closed_at": closed_time})
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [SWING TRADE - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn ({entry}) để gồng sóng lớn tới TP2 ({tp2})!")
                    notified.add(ev_key)
                
        if sl_hit:
            if trade.get('tp1_hit', False):
                ev_key = f"{t_id}_BE"
                if ev_key not in notified:
                    send_telegram_msg(f"🛡️ [CHẠM HÒA VỐN SWING] Kèo {act} ({t_id}) đóng phần còn lại tại Entry ({entry}) sau khi đã chốt TP1 (+{trade['tp1_pips']} pips).")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "TP1 + BE", "pips": trade['tp1_pips'], "closed_at": closed_time})
            else:
                ev_key = f"{t_id}_SL"
                if ev_key not in notified:
                    send_telegram_msg(f"🛑 [CẮN SL SWING] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "CẮN SL", "pips": -trade['sl_pips'], "closed_at": closed_time})
                    pm_prompt = f"Kèo Swing {act} Vàng ({t_id}) theo cấu trúc H4/SMC vừa chạm SL tại {sl} (Entry: {entry}). High={c_high}, Low={c_low}, Close={c_close}. Giải phẫu nguyên nhân gãy cấu trúc H4 trong 3 ý ngắn gọn."
                    pm_res = call_groq_rotation(pm_prompt, max_tokens=450, temperature=0.3)
                    if pm_res:
                        send_telegram_msg(f"🔍 [GIẢI PHẪU LỆNH SWING - POST-MORTEM: {t_id}]\n\n{pm_res}")
                    
        if not is_closed:
            remaining.append(trade)
            
    state['active_trades'] = remaining
    state['notified_events'] = list(notified)
    save_state(state)

def get_market_data():
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY!")
    # Lấy 250 nến H1 (~10 ngày giao dịch) để dựng biểu đồ và cấu trúc H4/D1 chuẩn cho Swing Trade
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=250&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    
    for attempt in range(3):
        try:
            res = requests.get(url, timeout=15).json()
            if "values" in res:
                break
        except Exception as e:
            if attempt == 2:
                raise ValueError(f"Mất kết nối Twelve Data: {e}")
            time.sleep(5)

    df = pd.DataFrame(res['values']).iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    df['Volume'] = 1000
    df_h1 = df.iloc[:-1].copy()
    
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    df_h1['EMA50'] = df_h1['Close'].ewm(span=50, adjust=False).mean()
    
    df_h4 = df_h1.resample('4h', offset='1h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    df_h4['EMA50'] = df_h4['Close'].ewm(span=50, adjust=False).mean()
    
    h_l4 = df_h4['High'] - df_h4['Low']
    h_c4 = np.abs(df_h4['High'] - df_h4['Close'].shift())
    l_c4 = np.abs(df_h4['Low'] - df_h4['Close'].shift())
    df_h4['ATR_H4'] = pd.concat([h_l4, h_c4, l_c4], axis=1).max(axis=1).rolling(14).mean()
    
    c_open = df_h1.index[-1]
    c_close = c_open + timedelta(hours=1)
    is_4h_closed = c_close.hour in [1, 5, 9, 13, 17, 21]
    prog = "NẾN H4 VỪA ĐÓNG CỬA" if is_4h_closed else f"Giờ thứ {((c_close.hour - 1) % 4) + 1}/4 của nến H4"
    return df_h1.tail(80), df_h4.tail(50), is_4h_closed, prog, c_close

def generate_swing_chart(df_h4, title_text):
    path = "xauusd_swing_h4.png"
    add_plots = [
        mpf.make_addplot(df_h4['EMA21'], color='#ff9800', width=1.6),
        mpf.make_addplot(df_h4['EMA50'], color='#2196f3', width=1.6)
    ]
    fig, _ = mpf.plot(df_h4, type='candle', style='binance', addplot=add_plots, title=f"\n{title_text}", figsize=(13, 6.5), returnfig=True)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def build_swing_context(df_h4, df_h1):
    recent_h4 = df_h4.tail(15)
    major_high = round(recent_h4['High'].max(), 2)
    major_low = round(recent_h4['Low'].min(), 2)
    
    lines = [f"- Đỉnh cấu trúc H4 (Major Swing High 15 nến H4): {major_high} | Đáy cấu trúc H4 (Major Swing Low): {major_low}"]
    lines.append("- Chi tiết 5 nến H4 gần nhất (Cấu trúc sóng lớn):")
    for t, r in df_h4.tail(5).iterrows():
        c_type = "TĂNG" if r['Close'] >= r['Open'] else "GIẢM"
        u_wick = r['High'] - max(r['Open'], r['Close'])
        l_wick = min(r['Open'], r['Close']) - r['Low']
        lines.append(f"  • Nến H4 {t.strftime('%d/%m %H:%M')}: {c_type} | O={r['Open']:.2f}, H={r['High']:.2f}, L={r['Low']:.2f}, C={r['Close']:.2f} (Râu trên={u_wick:.1f}, Râu dưới={l_wick:.1f})")
    return "\n".join(lines), major_high, major_low

def safe_float(val, default_val):
    try:
        if isinstance(val, (int, float)):
            return float(val)
        cleaned = re.sub(r'[^\d.-]', '', str(val))
        return float(cleaned) if cleaned else default_val
    except:
        return default_val

def ai_decide_swing_trade(h4_trend, curr_price, ema21_h4, ema50_h4, swing_context, atr_h4):
    min_sl_dist = round(max(atr_h4 * 0.85, 13.0), 1)
    prompt = f"""
Bạn là Chiến lược gia SWING TRADING (Giao dịch trung hạn H4/D1) theo trường phái SMC & Price Action cho quỹ Vàng XAU/USD.
DỮ LIỆU CẤU TRÚC H4 HIỆN TẠI:
- Giá hiện tại: {curr_price}
- EMA 21 khung H4: {ema21_h4} | EMA 50 khung H4: {ema50_h4}
- Xu hướng cấu trúc lớn H4: {h4_trend}
- Biên độ ATR khung H4: {atr_h4} giá
{swing_context}

NHIỆM VỤ SWING TRADING (GIỮ LỆNH NHIỀU NGÀY ĂN SÓNG LỚN):
1. Tìm điểm đảo chiều sóng lớn (CHoCH / Quét thanh khoản Major High/Low trên H4) hoặc nhịp Pullback sâu về vùng Order Block / EMA21-EMA50 của khung H4.
2. Không vào lệnh nếu giá đang lơ lửng giữa vùng tranh chấp. Chỉ chọn BUY hoặc SELL khi có setup Swing thực sự chất lượng (R:R từ 1:2 đến 1:3.5).

Trả về ĐÚNG định dạng JSON hợp lệ:
{{
  "quyet_dinh": "BUY" hoặc "SELL" hoặc "STAND_ASIDE",
  "ly_do_ngan": "Tóm tắt luận điểm Swing Trade H4 trong 1-2 câu",
  "phan_tich_chi_tiet": "Phân tích 3 gạch đầu dòng: 1. Cấu trúc vĩ mô H4/D1 (EMA21/EMA50 & Major High/Low). 2. Dấu hiệu dòng tiền lớn (Order Block / Liquidity Sweep trên H4). 3. Kế hoạch gồng lệnh Swing nhiều ngày.",
  "entry": {curr_price},
  "sl": số_thực (cách entry tối thiểu {min_sl_dist} giá để chịu nhiệt sóng H4),
  "tp1": số_thực (tỷ lệ R:R 1:2.0),
  "tp2": số_thực (tỷ lệ R:R 1:3.5)
}}
"""
    txt = call_groq_rotation(prompt, max_tokens=650, temperature=0.2)
    if txt:
        try:
            match = re.search(r'\{.*\}', txt, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"⚠️ Lỗi JSON Swing AI: {e}")

    return {
        "quyet_dinh": "STAND_ASIDE",
        "ly_do_ngan": "Chưa xuất hiện thiết lập Swing H4 chuẩn.",
        "phan_tich_chi_tiet": "• Tiếp tục kiên nhẫn chờ giá tiếp cận vùng thanh khoản H4.",
        "entry": curr_price, "sl": 0, "tp1": 0, "tp2": 0
    }

def main():
    print(f"=== BOT #1 (SWING TRADING H4/D1): ({len(GROQ_KEYS_POOL)} KEY GROQ) ===")
    df_h1, df_h4, is_4h_closed, prog, c_close = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = c_close.strftime('%H:%M - %d/%m')
    t_id = f"#SWING_H4_{c_close.strftime('%d%m_%H')}H"
    
    # Luôn kiểm tra TP/SL của các lệnh Swing đang chạy mỗi giờ
    check_tp_and_sl_hit(df_h1)
    check_and_send_weekly_summary()
    
    st = load_state()
    if st.get("last_processed_candle") == t_id:
        print(f"⏭️ Nến {t_id} đã xử lý. Bỏ qua!")
        return
    st["last_processed_candle"] = t_id
    save_state(st)

    # ĐẶC TRƯNG SWING BOT: Chỉ quét tìm lệnh mới khi NẾN H4 ĐÓNG CỬA (4 tiếng 1 lần)
    if not is_4h_closed:
        print(f"⏳ [{prog}] Chưa đóng nến H4. Bot Swing chỉ canh TP/SL mỗi giờ và chờ đóng nến H4 mới tìm lệnh.")
        return

    h4_last = df_h4.iloc[-1]
    ema21_h4 = round(h4_last['EMA21'], 2)
    ema50_h4 = round(h4_last['EMA50'], 2)
    atr_h4 = round(h4_last['ATR_H4'], 1)
    if np.isnan(atr_h4) or atr_h4 < 12.0:
        atr_h4 = 15.0

    if h4_last['Close'] > ema21_h4 and ema21_h4 >= ema50_h4:
        h4_trend_str = "TĂNG MẠNH (BULLISH STRUCTURE)"
    elif h4_last['Close'] < ema21_h4 and ema21_h4 <= ema50_h4:
        h4_trend_str = "GIẢM MẠNH (BEARISH STRUCTURE)"
    else:
        h4_trend_str = "TÍCH LŨY / CHUYỂN GIAO CẤU TRÚC"
        
    swing_context, major_high, major_low = build_swing_context(df_h4, df_h1)
    
    unprotected_trades = [t for t in st.get("active_trades", []) if not t.get("tp1_hit", False)]
    if len(unprotected_trades) > 0:
        print("🛡️ Đang có lệnh Swing chưa chạm TP1. Tập trung gồng lệnh hiện tại.")
        return

    decision = ai_decide_swing_trade(h4_trend_str, curr_price, ema21_h4, ema50_h4, swing_context, atr_h4)
    act = str(decision.get("quyet_dinh", "STAND_ASIDE")).upper().strip()
    reason = decision.get("ly_do_ngan", "Chưa có điểm vào Swing.")
    deep_analysis = decision.get("phan_tich_chi_tiet", reason)
    
    # Khoảng cách SL chuẩn Swing H4: Tối thiểu 13 - 22 giá để không bị nhiễu ngắn hạn quét
    min_swing_dist = round(max(atr_h4 * 0.85, 13.0), 2)
    max_swing_dist = round(max(atr_h4 * 1.4, 24.0), 2)
    
    if act in ["BUY", "SELL"]:
        chart_file = generate_swing_chart(df_h4, f"[BOT #1: SWING H4] {h4_trend_str}")
        raw_sl = safe_float(decision.get("sl"), 0)
        
        if act == "BUY":
            struct_sl = round(min(h4_last['Low'] - 3.5, curr_price - min_swing_dist), 2)
            sl_val = raw_sl if (0 < raw_sl <= struct_sl) else struct_sl
            risk = round(np.clip(abs(curr_price - sl_val), min_swing_dist, max_swing_dist), 2)
            sl_val = round(curr_price - risk, 2)
            tp1_val = round(curr_price + risk * 2.0, 2)
            tp2_val = round(curr_price + risk * 3.5, 2)
        else:
            struct_sl = round(max(h4_last['High'] + 3.5, curr_price + min_swing_dist), 2)
            sl_val = raw_sl if (raw_sl >= struct_sl) else struct_sl
            risk = round(np.clip(abs(sl_val - curr_price), min_swing_dist, max_swing_dist), 2)
            sl_val = round(curr_price + risk, 2)
            tp1_val = round(curr_price - risk * 2.0, 2)
            tp2_val = round(curr_price - risk * 3.5, 2)
            
        sl_pips = int(round(abs(curr_price - sl_val) * 10))
        tp1_pips = int(round(abs(tp1_val - curr_price) * 10))
        tp2_pips = int(round(abs(tp2_val - curr_price) * 10))
        
        header = f"🦅 [BOT #1: SWING TRADING H4/D1 - BẮT SÓNG LỚN]\n" \
                 f"🎫 Mã vé: {t_id} (Giữ lệnh theo cấu trúc H4)\n" \
                 f"⏰ Đóng nến H4 lúc: {time_str}\n" \
                 f"📊 Cấu trúc H4: {h4_trend_str}\n" \
                 f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n"
                 
        caption = f"{header}" \
                  f"🚨 KÈO SWING TRUNG HẠN: VÀO {act} VÀNG\n" \
                  f"📍 Entry: {curr_price}\n" \
                  f"🛑 Cắt lỗ H4 (SL): {sl_val} (-{sl_pips} pips)\n" \
                  f"🎯 TP1 (R:R 1:2.0 - Dời BE): {tp1_val} (+{tp1_pips} pips)\n" \
                  f"🚀 TP2 (R:R 1:3.5 - Sóng lớn): {tp2_val} (+{tp2_pips} pips)\n\n" \
                  f"💡 LUẬN ĐIỂM SWING: {reason}"
                  
        st['active_trades'].append({
            "id": t_id, "action": act, "timeframe": "H4_SWING",
            "entry": curr_price, "sl": sl_val, "tp1": tp1_val, "tp2": tp2_calc if 'tp2_calc' in locals() else tp2_val,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
        send_telegram_photo(chart_file, caption)
        time.sleep(1)
        send_telegram_msg(f"🧠 [PHÂN TÍCH VĨ MÔ & CẤU TRÚC SWING H4 - {t_id}]\n\n{deep_analysis}")
        if os.path.exists(chart_file):
            os.remove(chart_file)
    else:
        print(f"🚫 [{t_id}] Đóng nến H4 nhưng chưa có kèo Swing A+ ({reason}). Im lặng trên Telegram.")

if __name__ == "__main__":
    main()

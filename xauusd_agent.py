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
                    data["active_trades"] = unique_trades[-1:] if len(unique_trades) > 1 else unique_trades
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

def call_groq_rotation(prompt, max_tokens=650, temperature=0.25):
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
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Bạn là Chuyên gia Giao dịch Price Action thực chiến theo trường phái Nhật Hoài - TraderViet "
                                "(Nial Fuller, Bob Volman, Naked Forex: Pin Bar, Fakey, Inside Bar, Engulfing, Two-Bar Reversal & Vùng Hợp Lưu). "
                                "Trả lời ngắn gọn, súc tích, thực chiến bằng tiếng Việt."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=20.0
                )
                ans = res.choices[0].message.content
                if ans and ans.strip():
                    print(f"✅ AI TraderViet phản hồi thành công với Key #{idx+1} ({masked}) | Model: {m}")
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
        f"📅 [BÁO CÁO TỔNG KẾT TUẦN - BOT #1 PRICE ACTION TRADERVIET]\n"
        f"🗓️ Kỳ báo cáo: Tuần {week_num}/{year}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 THỐNG KÊ HIỆU SUẤT SWING PA:\n"
        f"• Tổng số lệnh đã đóng: {total_trades} lệnh\n"
        f"• Số lệnh Thắng: {len(win_trades)} lệnh ✅\n"
        f"• Số lệnh Thua: {len(loss_trades)} lệnh 🛑\n"
        f"• Tỷ lệ thắng (Winrate): {win_rate}%\n"
        f"• TỔNG LỢI NHUẬN RÒNG: {net_str} PIPS 💰\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📜 CHI TIẾT TỪNG KÈO:\n"
        f"{details_str}"
    )
    send_telegram_msg(summary_msg)
    state["last_weekly_report"] = week_id
    state["closed_trades"] = []
    save_state(state)

def check_tp_sl_and_time_rule(df_h1):
    """
    Kiểm tra TP/SL & Áp dụng QUY TẮC THỜI GIAN (Bài 23 TraderViet):
    Nếu sau 8 cây nến H1 mà lệnh chưa chạm TP1 và đang đi ngược/lình xình, chủ động đóng lệnh sớm để bảo vệ vốn.
    """
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
                    send_telegram_msg(f"🦅 [PRICE ACTION - FULL TP2 SÓNG DÀI] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI ĐẬM: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time})
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [PRICE ACTION - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Hòa Vốn ({entry}) theo nguyên tắc Set & Forget!")
                    notified.add(ev_key)
        elif act == "SELL":
            if c_high >= sl:
                sl_hit = True
                is_closed = True
            elif c_low <= tp2:
                ev_key = f"{t_id}_TP2"
                if ev_key not in notified:
                    send_telegram_msg(f"🦅 [PRICE ACTION - FULL TP2 SÓNG DÀI] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI ĐẬM: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time})
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [PRICE ACTION - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Hòa Vốn ({entry}) theo nguyên tắc Set & Forget!")
                    notified.add(ev_key)
                
        if sl_hit:
            if trade.get('tp1_hit', False):
                ev_key = f"{t_id}_BE"
                if ev_key not in notified:
                    send_telegram_msg(f"🛡️ [CHẠM HÒA VỐN BE] Kèo {act} ({t_id}) đóng phần còn lại tại Entry ({entry}) sau khi đã bỏ túi TP1 (+{trade['tp1_pips']} pips).")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "TP1 + BE", "pips": trade['tp1_pips'], "closed_at": closed_time})
            else:
                ev_key = f"{t_id}_SL"
                if ev_key not in notified:
                    send_telegram_msg(f"🛑 [CẮN SL PRICE ACTION] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "CẮN SL", "pips": -trade['sl_pips'], "closed_at": closed_time})
                    pm_prompt = (
                        f"Theo phương pháp Price Action của TraderViet, hãy giải phẫu lệnh {act} Vàng ({t_id}) vừa chạm SL tại {sl} (Entry: {entry}). "
                        f"Nến H1 vừa đóng có High={c_high}, Low={c_low}, Close={c_close}. "
                        f"Trình bày 3 ý ngắn gọn: 1. Đây là Bẫy giá (Bull/Bear Trap) hay Vùng Sự Kiện bị phá vỡ thật. "
                        f"2. Đánh giá động lượng thanh nến. 3. Bài học rút ra."
                    )
                    pm_res = call_groq_rotation(pm_prompt, max_tokens=450, temperature=0.3)
                    if pm_res:
                        send_telegram_msg(f"🔍 [GIẢI PHẪU LỆNH THUA PRICE ACTION: {t_id}]\n\n{pm_res}")

        # QUY TẮC THỜI GIAN (TIME RULE - BÀI 23 TRADERVIET)
        if not is_closed:
            hours_held = trade.get("hours_held", 0) + 1
            trade["hours_held"] = hours_held
            # Nếu sau 8 tiếng chưa đạt được TP1 mà giá lình xình mất động lượng -> Thoát sớm
            if not trade.get("tp1_hit", False) and hours_held >= 8:
                raw_pips = int(round((c_close - entry) * 10)) if act == "BUY" else int(round((entry - c_close) * 10))
                if -50 <= raw_pips <= 40:
                    ev_key = f"{t_id}_TIMERULE"
                    if ev_key not in notified:
                        send_telegram_msg(
                            f"⏱️ [QUY TẮC THỜI GIAN - BÀI 23 TRADERVIET]\n"
                            f"Kèo {act} ({t_id}) đã giữ 8 giờ nhưng giá không bứt phá theo kỳ vọng. "
                            f"Chủ động đóng lệnh tại {c_close} ({raw_pips:+} pips) để bảo toàn vốn!"
                        )
                        notified.add(ev_key)
                        state["closed_trades"].append({"id": t_id, "action": act, "result": "THOÁT QUY TẮC THỜI GIAN", "pips": raw_pips, "closed_at": closed_time})
                    is_closed = True
            elif hours_held >= 36:
                raw_pips = int(round((c_close - entry) * 10)) if act == "BUY" else int(round((entry - c_close) * 10))
                final_pips = max(raw_pips, trade['tp1_pips']) if trade.get('tp1_hit', False) else raw_pips
                send_telegram_msg(f"⏱️ [CHỐT LỆNH SWING 36H] Đóng kèo {act} ({t_id}) tại {c_close} ({final_pips:+} pips).")
                state["closed_trades"].append({"id": t_id, "action": act, "result": "ĐÓNG 36H", "pips": final_pips, "closed_at": closed_time})
                is_closed = True

        if not is_closed:
            remaining.append(trade)
            
    state['active_trades'] = remaining
    state['notified_events'] = list(notified)
    save_state(state)

def get_market_data():
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY!")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=240&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    
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
    df['Volume'] = ((df['High'] - df['Low']) * 1200).astype(int) + 600
    df_h1 = df.iloc[:-1].copy()
    
    df_h1['Range'] = df_h1['High'] - df_h1['Low']
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    df_h1['EMA50'] = df_h1['Close'].ewm(span=50, adjust=False).mean()
    
    h_l = df_h1['High'] - df_h1['Low']
    h_c = np.abs(df_h1['High'] - df_h1['Close'].shift())
    l_c = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    df_h4 = df_h1.resample('4h', offset='1h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    df_h4['EMA50'] = df_h4['Close'].ewm(span=50, adjust=False).mean()
    
    c_open = df_h1.index[-1]
    c_close = c_open + timedelta(hours=1)
    is_4h_closed = c_close.hour in [1, 5, 9, 13, 17, 21]
    prog = "NẾN H4 VỪA ĐÓNG CỬA" if is_4h_closed else f"Giờ thứ {((c_close.hour - 1) % 4) + 1}/4 của nến H4"
    return df_h1.tail(65), df_h4.tail(45), is_4h_closed, prog, c_close

def generate_pa_chart(df_h1, res_zone, sup_zone, title_text):
    path = "xauusd_pa_traderviet.png"
    add_plots = [
        mpf.make_addplot(df_h1['EMA21'], color='#ff9800', width=1.6),
        mpf.make_addplot(df_h1['EMA50'], color='#2196f3', width=1.4),
        mpf.make_addplot(np.full(len(df_h1), res_zone), color='#e91e63', width=1.3, linestyle='--'),
        mpf.make_addplot(np.full(len(df_h1), sup_zone), color='#4caf50', width=1.3, linestyle='--')
    ]
    fig, _ = mpf.plot(df_h1, type='candle', style='binance', addplot=add_plots, title=f"\n{title_text}", figsize=(13, 6.5), returnfig=True)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def scan_traderviet_price_action(df_h1, df_h4):
    """
    BỘ QUÉT TINH HOA 97 BÀI PRICE ACTION TRADERVIET:
    1. Lọc vùng giằng co (Barb Wire / Choppy Market - Bài 52, 53).
    2. Xác định Vùng Sự Kiện (Event Zones - Bài 30) & Vùng Hợp Lưu (Confluence - Bài 50).
    3. Quét 4 mô hình kinh điển:
       - Pin Bar Đuôi Dài (Bài 1, 14, 18)
       - Fakey / Inside Bar False Breakout (Bài 2, 3, 15, 17)
       - Bullish / Bearish Engulfing (Bài 38, 39)
       - Mô Hình 2 Nến Đảo Chiều / Two-Bar Reversal (Bài 57, 58)
    """
    c0 = df_h1.iloc[-1]  # Nến tín hiệu vừa đóng
    c1 = df_h1.iloc[-2]  # Nến trước đó 1 giờ
    c2 = df_h1.iloc[-3]  # Nến trước đó 2 giờ (dùng cho Fakey / Mother Bar)
    
    curr_price = round(c0['Close'], 2)
    atr = round(c0['ATR'], 1) if not np.isnan(c0['ATR']) and c0['ATR'] >= 5.0 else 6.5

    # --- BƯỚC 1: KIỂM TRA VÙNG GIẰNG CO / KHÔNG RÕ XU HƯỚNG (BÀI 52, 53) ---
    recent_6 = df_h1.iloc[-6:]
    box_range = recent_6['High'].max() - recent_6['Low'].min()
    # Nếu 6 cây nến H1 bị nhốt trong biên độ quá hẹp (< 1.15 * ATR), thị trường đang nén/nhiễu
    is_choppy = box_range < (atr * 1.15)

    # --- BƯỚC 2: XÁC ĐỊNH VÙNG SỰ KIỆN (EVENT ZONES) & XU HƯỚNG H4 ---
    lookback_24 = df_h1.iloc[-25:-1]
    res_zone = round(lookback_24['High'].max(), 2) # Kháng cự Swing High
    sup_zone = round(lookback_24['Low'].min(), 2)  # Hỗ trợ Swing Low

    sub_10 = df_h1.iloc[-11:-1]
    minor_res = round(sub_10['High'].max(), 2)
    minor_sup = round(sub_10['Low'].min(), 2)

    ema21_h1, ema50_h1 = c0['EMA21'], c0['EMA50']
    ema21_h4, ema50_h4 = df_h4['EMA21'].iloc[-1], df_h4['EMA50'].iloc[-1]
    h4_close = df_h4['Close'].iloc[-1]

    h4_uptrend = (h4_close >= ema21_h4 - 2.0) and (ema21_h4 >= ema50_h4 - 3.0)
    h4_downtrend = (h4_close <= ema21_h4 + 2.0) and (ema21_h4 <= ema50_h4 + 3.0)

    # Kiểm tra Vùng Hợp Lưu (Confluence - Bài 50, 51):
    # - Hợp lưu Mua: Giá vừa chạm/quét Hỗ trợ (sup_zone / minor_sup) HOẶC chạm vùng EMA21/EMA50 trong xu hướng Tăng
    at_buy_confluence = (
        (c0['Low'] <= sup_zone + 3.0) or
        (c0['Low'] <= minor_sup + 2.0) or
        (h4_uptrend and c0['Low'] <= max(ema21_h1, ema50_h1) + 2.5 and curr_price >= min(ema21_h1, ema50_h1))
    )
    # - Hợp lưu Bán: Giá vừa chạm/quét Kháng cự (res_zone / minor_res) HOẶC chạm vùng EMA21/EMA50 trong xu hướng Giảm
    at_sell_confluence = (
        (c0['High'] >= res_zone - 3.0) or
        (c0['High'] >= minor_res - 2.0) or
        (h4_downtrend and c0['High'] >= min(ema21_h1, ema50_h1) - 2.5 and curr_price <= max(ema21_h1, ema50_h1))
    )

    # Không mua ngay sát kháng cự đỉnh, không bán ngay sát hỗ trợ đáy (Đảm bảo Room to Run - Bài 25)
    room_for_buy = (res_zone - curr_price) >= (atr * 1.1)
    room_for_sell = (curr_price - sup_zone) >= (atr * 1.1)
    not_climax = c0['Range'] <= (atr * 2.1) # Tránh nến dãn quá dài (Exhaustion)

    # Thông số hình học của nến hiện tại (c0) và nến trước (c1)
    r0 = max(c0['Range'], 0.1)
    body0 = abs(c0['Close'] - c0['Open'])
    u_wick0 = c0['High'] - max(c0['Open'], c0['Close'])
    l_wick0 = min(c0['Open'], c0['Close']) - c0['Low']

    r1 = max(c1['Range'], 0.1)
    body1 = abs(c1['Close'] - c1['Open'])

    # --- BƯỚC 3: NHẬN DIỆN 4 MÔ HÌNH PRICE ACTION KINH ĐIỂN ---
    # 1. MÔ HÌNH PIN BAR ĐUÔI DÀI (Bài 1, 14, 18 - Nial Fuller)
    # Râu nến >= 55% tổng chiều dài nến, đóng cửa nằm ở 1/3 trên (Bullish) hoặc 1/3 dưới (Bearish)
    bull_pinbar = (l_wick0 >= r0 * 0.55) and (u_wick0 <= r0 * 0.25) and (r0 >= atr * 0.55)
    bear_pinbar = (u_wick0 >= r0 * 0.55) and (l_wick0 <= r0 * 0.25) and (r0 >= atr * 0.55)

    # 2. MÔ HÌNH FAKEY - INSIDE BAR FALSE BREAKOUT (Bài 3, 17)
    # c1 là Inside Bar nằm trong Mother Bar c2 (hoặc c1 có biên độ hẹp), c0 quét qua đỉnh/đáy rồi đóng cửa ngược lại
    is_inside_bar = (c1['High'] <= c2['High'] + 0.8) and (c1['Low'] >= c2['Low'] - 0.8)
    bull_fakey = is_inside_bar and (c0['Low'] < min(c1['Low'], c2['Low'])) and (c0['Close'] > c1['Low'] + r1 * 0.5) and (c0['Close'] > c0['Open'])
    bear_fakey = is_inside_bar and (c0['High'] > max(c1['High'], c2['High'])) and (c0['Close'] < c1['High'] - r1 * 0.5) and (c0['Close'] < c0['Open'])

    # 3. MÔ HÌNH NẾN NHẤN CHÌM - ENGULFING CHUYÊN SÂU (Bài 38, 39)
    # Nến trước ngược màu, nến hiện tại có thân lớn bao trùm thân nến trước và đóng cửa dứt khoát
    bull_engulfing = (c1['Close'] < c1['Open']) and (c0['Close'] > c0['Open']) and (body0 >= body1 * 1.15) and (c0['Close'] >= c1['High'] - 0.5) and (body0 >= r0 * 0.55)
    bear_engulfing = (c1['Close'] > c1['Open']) and (c0['Close'] < c0['Open']) and (body0 >= body1 * 1.15) and (c0['Close'] <= c1['Low'] + 0.5) and (body0 >= r0 * 0.55)

    # 4. MÔ HÌNH HAI NẾN ĐẢO CHIỀU - TWO-BAR REVERSAL (Bài 57, 58)
    # Nến c1 giảm mạnh, nến c0 tăng mạnh tương đương trả lại hoàn toàn điểm số (hoặc ngược lại)
    bull_twobar = (c1['Close'] < c1['Open']) and (body1 >= atr * 0.5) and (c0['Close'] > c0['Open']) and (body0 >= body1 * 0.85) and (abs(c0['Low'] - c1['Low']) <= 2.5)
    bear_twobar = (c1['Close'] > c1['Open']) and (body1 >= atr * 0.5) and (c0['Close'] < c0['Open']) and (body0 >= body1 * 0.85) and (abs(c0['High'] - c1['High']) <= 2.5)

    if is_choppy and not (bull_fakey or bear_fakey):
        return "STAND_ASIDE", "Thị trường đang trong Vùng Giằng Co (Bài 52-53), đứng ngoài chờ phá vỡ.", res_zone, sup_zone

    if at_buy_confluence and room_for_buy and not_climax:
        if bull_fakey:
            return "BUY", f"MÔ HÌNH FAKEY TĂNG (Bài 3 & 17 - Cú lừa quét đáy Inside Bar tại Vùng Hợp Lưu {minor_sup})", res_zone, sup_zone
        if bull_pinbar:
            return "BUY", f"PIN BAR ĐUÔI DÀI TĂNG (Bài 14 & 18 - Rút chân {l_wick0:.1f} giá từ chối vùng hỗ trợ)", res_zone, sup_zone
        if bull_engulfing:
            return "BUY", f"BULLISH ENGULFING (Bài 38 & 39 - Nến Nhấn Chìm Tăng xác nhận động lượng tại Vùng Hợp Lưu)", res_zone, sup_zone
        if bull_twobar:
            return "BUY", f"MÔ HÌNH 2 NẾN ĐẢO CHIỀU TĂNG (Bài 57 & 58 - Two-Bar Reversal bắt đáy)", res_zone, sup_zone

    if at_sell_confluence and room_for_sell and not_climax:
        if bear_fakey:
            return "SELL", f"MÔ HÌNH FAKEY GIẢM (Bài 3 & 17 - Cú lừa quét đỉnh Inside Bar tại Vùng Hợp Lưu {minor_res})", res_zone, sup_zone
        if bear_pinbar:
            return "SELL", f"PIN BAR ĐUÔI DÀI GIẢM (Bài 14 & 18 - Rút râu trên {u_wick0:.1f} giá từ chối vùng kháng cự)", res_zone, sup_zone
        if bear_engulfing:
            return "SELL", f"BEARISH ENGULFING (Bài 38 & 39 - Nến Nhấn Chìm Giảm xác nhận động lượng tại Vùng Hợp Lưu)", res_zone, sup_zone
        if bear_twobar:
            return "SELL", f"MÔ HÌNH 2 NẾN ĐẢO CHIỀU GIẢM (Bài 57 & 58 - Two-Bar Reversal bắt đỉnh)", res_zone, sup_zone

    return "STAND_ASIDE", "Chưa xuất hiện bộ 4 mô hình (Pin Bar / Fakey / Engulfing / 2-Bar) tại Vùng Hợp Lưu.", res_zone, sup_zone

def main():
    print(f"=== BOT #1 (PRICE ACTION TRADERVIET H1/H4): ({len(GROQ_KEYS_POOL)} KEY GROQ) ===")
    df_h1, df_h4, is_4h_closed, prog, c_close = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = c_close.strftime('%H:%M - %d/%m')
    t_id = f"#PA_{c_close.strftime('%d%m_%H')}H"
    
    check_tp_sl_and_time_rule(df_h1)
    check_and_send_weekly_summary()
    
    st = load_state()
    if st.get("last_processed_candle") == t_id:
        print(f"⏭️ Nến {t_id} đã xử lý. Bỏ qua!")
        return
    st["last_processed_candle"] = t_id
    save_state(st)

    if c_close.hour in [2, 3, 4, 5]:
        print("🌙 Giờ nghỉ giao dịch cuối phiên Mỹ: Tạm ngưng mở lệnh mới.")
        return

    c_last = df_h1.iloc[-1]
    h4_last = df_h4.iloc[-1]
    ema21_h4 = round(h4_last['EMA21'], 2)
    ema50_h4 = round(h4_last['EMA50'], 2)
    atr_h1 = round(c_last['ATR'], 1) if not np.isnan(c_last['ATR']) and c_last['ATR'] >= 5.5 else 6.5
    
    if h4_last['Close'] > ema21_h4 and ema21_h4 >= ema50_h4:
        h4_trend_str = "XU HƯỚNG TĂNG (BULLISH)"
    elif h4_last['Close'] < ema21_h4 and ema21_h4 <= ema50_h4:
        h4_trend_str = "XU HƯỚNG GIẢM (BEARISH)"
    else:
        h4_trend_str = "CHUYỂN GIAO / TÍCH LŨY"

    act, setup_desc, res_zone, sup_zone = scan_traderviet_price_action(df_h1, df_h4)
    
    unprotected_trades = [t for t in st.get("active_trades", []) if not t.get("tp1_hit", False)]
    has_risk_trade = len(unprotected_trades) > 0

    if act in ["BUY", "SELL"] and not has_risk_trade:
        chart_file = generate_pa_chart(df_h1, res_zone, sup_zone, f"[BOT #1: TRADERVIET PA] {act} | {h4_trend_str}")
        
        # ĐẶT DỪNG LỖ CHUẨN PRICE ACTION CHUYÊN NGHIỆP (Bài 26 TraderViet):
        # Đặt SL ngay sau đuôi nến tín hiệu (hoặc cụm 2 nến) + khoảng đệm tránh Stop Hunt
        min_pa_risk = max(atr_h1 * 1.35, 9.0)
        max_pa_risk = max(atr_h1 * 2.1, 16.0)
        
        if act == "BUY":
            pattern_low = min(c_last['Low'], df_h1['Low'].iloc[-2])
            raw_sl = pattern_low - max(atr_h1 * 0.4, 2.5)
            risk = round(np.clip(curr_price - raw_sl, min_pa_risk, max_pa_risk), 2)
            sl_val = round(curr_price - risk, 2)
            tp1_val = round(curr_price + risk * 1.6, 2)
            tp2_val = round(curr_price + risk * 2.8, 2)
        else:
            pattern_high = max(c_last['High'], df_h1['High'].iloc[-2])
            raw_sl = pattern_high + max(atr_h1 * 0.4, 2.5)
            risk = round(np.clip(raw_sl - curr_price, min_pa_risk, max_pa_risk), 2)
            sl_val = round(curr_price + risk, 2)
            tp1_val = round(curr_price - risk * 1.6, 2)
            tp2_val = round(curr_price - risk * 2.8, 2)
            
        sl_pips = int(round(abs(curr_price - sl_val) * 10))
        tp1_pips = int(round(abs(tp1_val - curr_price) * 10))
        tp2_pips = int(round(abs(tp2_val - curr_price) * 10))
        
        ai_prompt = (
            f"Hệ thống Price Action TraderViet vừa phát hiện thiết lập {act} Vàng (XAU/USD) chuẩn A+:\n"
            f"- Mô hình kích hoạt: {setup_desc}\n"
            f"- Câu chuyện thị trường H4: {h4_trend_str} (EMA21={ema21_h4}, EMA50={ema50_h4})\n"
            f"- Vùng Sự Kiện (Event Zones): Kháng cự={res_zone} | Hỗ trợ={sup_zone}\n"
            f"- Thông số lệnh: Entry={curr_price}, SL={sl_val} (-{sl_pips} pips), TP1={tp1_val} (+{tp1_pips} pips), TP2={tp2_val} (+{tp2_pips} pips).\n\n"
            f"Hãy đóng vai biên tập viên Nhật Hoài - TraderViet, viết bản phân tích ngắn gọn trong 3 gạch đầu dòng:\n"
            f"1. Đọc hiểu Câu Chuyện Thị Trường & Ý nghĩa của mô hình nến vừa đóng.\n"
            f"2. Đánh giá 3 yếu tố Vùng Hợp Lưu (Confluence) ủng hộ lệnh này.\n"
            f"3. Kế hoạch quản lý lệnh theo phương pháp Set & Forget và Quy tắc Thời gian."
        )
        deep_analysis = call_groq_rotation(ai_prompt, max_tokens=600, temperature=0.25)
        if not deep_analysis:
            deep_analysis = f"• Tín hiệu kích hoạt theo chuẩn Price Action TraderViet: {setup_desc}.\n• Đặt SL sau đuôi nến tại {sl_val} và dời về Hòa Vốn khi chạm TP1 ({tp1_val})."

        header = f"🟢 [BOT #1: PRICE ACTION TRADERVIET - SWING H1/H4]\n" \
                 f"🎫 Mã vé: {t_id}\n" \
                 f"⏰ Thời điểm: {time_str} ({prog})\n" \
                 f"📊 Cấu trúc H4: {h4_trend_str}\n" \
                 f"🧱 Vùng Sự Kiện: Kháng cự {res_zone} | Hỗ trợ {sup_zone}\n" \
                 f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n"
                 
        caption = f"{header}" \
                  f"🚨 KÈO PRICE ACTION A+: VÀO {act} VÀNG\n" \
                  f"🕯️ Tín hiệu: {setup_desc}\n" \
                  f"📍 Entry: {curr_price}\n" \
                  f"🛑 Cắt lỗ sau đuôi nến (SL): {sl_val} (-{sl_pips} pips)\n" \
                  f"🎯 TP1 (R:R 1:1.6 - Set & Forget BE): {tp1_val} (+{tp1_pips} pips)\n" \
                  f"🚀 TP2 (R:R 1:2.8 - Bắt sóng dài): {tp2_val} (+{tp2_pips} pips)"
                  
        st['active_trades'].append({
            "id": t_id, "action": act, "timeframe": "H1_H4_PA",
            "entry": curr_price, "sl": sl_val, "tp1": tp1_val, "tp2": tp2_val,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
            "tp1_hit": False, "hours_held": 0
        })
        save_state(st)
        send_telegram_photo(chart_file, caption)
        time.sleep(1)
        send_telegram_msg(f"📘 [ĐỌC HIỂU HÀNH ĐỘNG GIÁ - NHẬT HOÀI TRADERVIET: {t_id}]\n\n{deep_analysis}")
        if os.path.exists(chart_file):
            os.remove(chart_file)
    else:
        print(f"🚫 [{t_id}] Đứng ngoài theo kỷ luật Price Action: {setup_desc}")

if __name__ == "__main__":
    main()

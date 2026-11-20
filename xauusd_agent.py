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

MAX_SL_DIST = 16.0
MIN_SL_DIST = 8.5

def load_state():
    default_state = {
        "active_trades": [],
        "watch_zones": [],
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
    if "notified_events" in state and len(state["notified_events"]) > 300:
        state["notified_events"] = state["notified_events"][-300:]
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
        cl = Groq(api_key=key)
        for m in models:
            try:
                res = cl.chat.completions.create(
                    model=m,
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là Chuyên gia Giao dịch Price Action thực chiến theo trường phái Nhật Hoài - TraderViet. Trả lời ngắn gọn, súc tích bằng tiếng Việt."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=20.0
                )
                ans = res.choices[0].message.content
                if ans and ans.strip():
                    return ans.strip()
            except Exception as e:
                err_str = str(e)
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
        f"📅 [BÁO CÁO TỔNG KẾT TUẦN - BOT #1 PRICE ACTION]\n"
        f"🗓️ Kỳ báo cáo: Tuần {week_num}/{year}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"• Tổng số lệnh đã đóng: {total_trades} lệnh\n"
        f"• Thắng: {len(win_trades)} ✅ | Thua: {len(loss_trades)} 🛑\n"
        f"• Winrate: {win_rate}% | Ròng: {net_str} PIPS 💰\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{details_str}"
    )
    send_telegram_msg(summary_msg)
    state["last_weekly_report"] = week_id
    state["closed_trades"] = []
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
    df_h1['EMA10'] = df_h1['Close'].ewm(span=10, adjust=False).mean()
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

def generate_pa_chart(df_h1, res_zone, sup_zone, entry_level, title_text):
    path = "xauusd_pa_traderviet.png"
    add_plots = [
        mpf.make_addplot(df_h1['EMA10'], color='#9c27b0', width=1.1, linestyle=':'),
        mpf.make_addplot(df_h1['EMA21'], color='#ff9800', width=1.6),
        mpf.make_addplot(df_h1['EMA50'], color='#2196f3', width=1.4),
        mpf.make_addplot(np.full(len(df_h1), res_zone), color='#e91e63', width=1.2, linestyle='--'),
        mpf.make_addplot(np.full(len(df_h1), sup_zone), color='#4caf50', width=1.2, linestyle='--'),
        mpf.make_addplot(np.full(len(df_h1), entry_level), color='#00bcd4', width=1.8)
    ]
    fig, _ = mpf.plot(df_h1, type='candle', style='binance', addplot=add_plots, title=f"\n{title_text}", figsize=(13, 6.5), returnfig=True)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def check_active_and_watch_zones(df_h1, df_h4, time_str):
    state = load_state()
    notified = set(state.get("notified_events", []))
    last_c = df_h1.iloc[-1]
    c_high, c_low, c_close = last_c['High'], last_c['Low'], last_c['Close']
    closed_time = datetime.now(VN_TZ).strftime('%d/%m %H:%M')

    # --- 1. KIỂM TRA TP / SL / HÒA VỐN CỦA LỆNH ĐANG CHẠY ---
    active_trades = state.get("active_trades", [])
    remaining_active = []
    for trade in active_trades:
        t_id, act = trade['id'], trade['action']
        entry, sl, tp1, tp2 = float(trade['entry']), float(trade['sl']), float(trade['tp1']), float(trade['tp2'])
        is_closed = False
        sl_hit = False

        if act == "BUY":
            if c_low <= sl:
                sl_hit = True
                is_closed = True
            elif c_high >= tp2:
                ev_key = f"{t_id}_TP2"
                if ev_key not in notified:
                    send_telegram_msg(f"🚀 [BOT #1 - FULL TP2] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time})
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [BOT #1 - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Hòa Vốn ({entry})!")
                    notified.add(ev_key)
        elif act == "SELL":
            if c_high >= sl:
                sl_hit = True
                is_closed = True
            elif c_low <= tp2:
                ev_key = f"{t_id}_TP2"
                if ev_key not in notified:
                    send_telegram_msg(f"🚀 [BOT #1 - FULL TP2] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time})
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [BOT #1 - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Hòa Vốn ({entry})!")
                    notified.add(ev_key)

        if sl_hit:
            if trade.get('tp1_hit', False):
                ev_key = f"{t_id}_BE"
                if ev_key not in notified:
                    send_telegram_msg(f"🛡️ [BOT #1 - CHẠM HÒA VỐN BE] Kèo {act} ({t_id}) đóng tại Entry ({entry}) sau khi đạt TP1 (+{trade['tp1_pips']} pips).")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "TP1 + BE", "pips": trade['tp1_pips'], "closed_at": closed_time})
            else:
                ev_key = f"{t_id}_SL"
                if ev_key not in notified:
                    send_telegram_msg(f"🛑 [BOT #1 - CHẠM SL] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "CẮN SL", "pips": -trade['sl_pips'], "closed_at": closed_time})
                    pm_prompt = f"Giải phẫu lệnh {act} Vàng ({t_id}) theo Price Action TraderViet vừa chạm SL tại {sl} (Entry: {entry}). High={c_high}, Low={c_low}, Close={c_close}. Viết ngắn gọn 3 gạch đầu dòng."
                    pm_res = call_groq_rotation(pm_prompt, max_tokens=400, temperature=0.3)
                    if pm_res:
                        send_telegram_msg(f"🔍 [GIẢI PHẪU LỆNH THUA BOT #1: {t_id}]\n\n{pm_res}")

        if not is_closed:
            hours_held = trade.get("hours_held", 0) + 1
            trade["hours_held"] = hours_held
            if hours_held >= 48:
                raw_pips = int(round((c_close - entry) * 10)) if act == "BUY" else int(round((entry - c_close) * 10))
                final_pips = max(raw_pips, trade.get('tp1_pips', 0)) if trade.get('tp1_hit', False) else raw_pips
                ev_key = f"{t_id}_48H"
                if ev_key not in notified:
                    send_telegram_msg(f"⏱️ [BOT #1 - ĐÓNG LỆNH 48H] Đóng kèo {act} ({t_id}) tại {c_close} ({final_pips:+} pips).")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "ĐÓNG 48H", "pips": final_pips, "closed_at": closed_time})
                is_closed = True

        if not is_closed:
            remaining_active.append(trade)

    # --- 2. KIỂM TRA VÙNG CANH NGẦM (CHỈ BÁO VÀO LỆNH MARKET KHI GIÁ ĐẾN VÙNG & CẤU TRÚC CÒN NGUYÊN) ---
    watch_list = state.get("watch_zones", [])
    remaining_watch = []
    has_unprotected = any(not t.get("tp1_hit", False) for t in remaining_active)
    h4_close = df_h4['Close'].iloc[-1]
    ema21_h4 = df_h4['EMA21'].iloc[-1]

    for w in watch_list:
        w_id, w_act = w['id'], w['action']
        zone_price, sl = float(w['zone_price']), float(w['sl'])
        wait_hours = w.get("wait_hours", 0) + 1
        w["wait_hours"] = wait_hours

        if wait_hours > 6:
            continue

        if w_act == "BUY":
            # Nếu giá thủng luôn SL hoặc H4 đã đổi sang giảm mạnh -> Hủy vùng canh ngầm
            if c_low <= sl or h4_close < ema21_h4 - 8.0:
                continue
            # Giá đã hồi về chạm vùng canh VÀ đóng nến giữ trên SL (cấu trúc còn nguyên) -> BÁO VÀO MARKET NGAY!
            if c_low <= zone_price + 1.5 and c_close > sl + 2.0 and not has_unprotected:
                curr_entry = round(c_close, 2)
                risk = round(np.clip(curr_entry - sl, MIN_SL_DIST, MAX_SL_DIST), 2)
                opt_sl = round(curr_entry - risk, 2)
                tp1_val = round(curr_entry + risk * 1.5, 2)
                tp2_val = round(curr_entry + risk * 2.6, 2)
                sl_pips = int(round(risk * 10))
                tp1_pips = int(round(abs(tp1_val - curr_entry) * 10))
                tp2_pips = int(round(abs(tp2_val - curr_entry) * 10))

                chart_file = generate_pa_chart(df_h1, w['res_zone'], w['sup_zone'], curr_entry, f"[BOT #1: MARKET BUY] @ {curr_entry}")
                caption = (
                    f"🟢 [BOT #1: GIÁ ĐÃ VỀ VÙNG - VÀO BUY MARKET NGAY]\n"
                    f"🎫 Mã vé: {w_id} | ⏰ {time_str}\n"
                    f"🕯️ Xác nhận: Giá test vùng {zone_price} & giữ cấu trúc ({w['pattern']})\n"
                    f"📍 Entry Market ngay: {curr_entry}\n"
                    f"🛑 SL: {opt_sl} (-{sl_pips} pips)\n"
                    f"🎯 TP1: {tp1_val} (+{tp1_pips} pips) -> Dời BE\n"
                    f"🚀 TP2: {tp2_val} (+{tp2_pips} pips)"
                )
                send_telegram_photo(chart_file, caption)
                if os.path.exists(chart_file):
                    os.remove(chart_file)
                remaining_active.append({
                    "id": w_id, "action": "BUY", "timeframe": "H1_RETEST_MKT",
                    "entry": curr_entry, "sl": opt_sl, "tp1": tp1_val, "tp2": tp2_val,
                    "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
                    "tp1_hit": False, "hours_held": 0
                })
                has_unprotected = True
                continue

        elif w_act == "SELL":
            if c_high >= sl or h4_close > ema21_h4 + 8.0:
                continue
            if c_high >= zone_price - 1.5 and c_close < sl - 2.0 and not has_unprotected:
                curr_entry = round(c_close, 2)
                risk = round(np.clip(sl - curr_entry, MIN_SL_DIST, MAX_SL_DIST), 2)
                opt_sl = round(curr_entry + risk, 2)
                tp1_val = round(curr_entry - risk * 1.5, 2)
                tp2_val = round(curr_entry - risk * 2.6, 2)
                sl_pips = int(round(risk * 10))
                tp1_pips = int(round(abs(curr_entry - tp1_val) * 10))
                tp2_pips = int(round(abs(curr_entry - tp2_val) * 10))

                chart_file = generate_pa_chart(df_h1, w['res_zone'], w['sup_zone'], curr_entry, f"[BOT #1: MARKET SELL] @ {curr_entry}")
                caption = (
                    f"🔴 [BOT #1: GIÁ ĐÃ VỀ VÙNG - VÀO SELL MARKET NGAY]\n"
                    f"🎫 Mã vé: {w_id} | ⏰ {time_str}\n"
                    f"🕯️ Xác nhận: Giá test vùng {zone_price} & giữ cấu trúc ({w['pattern']})\n"
                    f"📍 Entry Market ngay: {curr_entry}\n"
                    f"🛑 SL: {opt_sl} (-{sl_pips} pips)\n"
                    f"🎯 TP1: {tp1_val} (+{tp1_pips} pips) -> Dời BE\n"
                    f"🚀 TP2: {tp2_val} (+{tp2_pips} pips)"
                )
                send_telegram_photo(chart_file, caption)
                if os.path.exists(chart_file):
                    os.remove(chart_file)
                remaining_active.append({
                    "id": w_id, "action": "SELL", "timeframe": "H1_RETEST_MKT",
                    "entry": curr_entry, "sl": opt_sl, "tp1": tp1_val, "tp2": tp2_val,
                    "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
                    "tp1_hit": False, "hours_held": 0
                })
                has_unprotected = True
                continue

        remaining_watch.append(w)

    state['active_trades'] = remaining_active
    state['watch_zones'] = remaining_watch
    state['notified_events'] = list(notified)
    save_state(state)

def scan_traderviet_price_action(df_h1, df_h4):
    c0 = df_h1.iloc[-1]
    c1 = df_h1.iloc[-2]
    c2 = df_h1.iloc[-3]

    curr_price = round(c0['Close'], 2)
    atr = round(np.clip(c0['ATR'] if not np.isnan(c0['ATR']) else 7.0, 5.5, 12.0), 1)

    recent_6 = df_h1.iloc[-6:]
    box_range = recent_6['High'].max() - recent_6['Low'].min()
    is_choppy = box_range < (atr * 0.95)

    lookback_24 = df_h1.iloc[-25:-1]
    res_zone = round(lookback_24['High'].max(), 2)
    sup_zone = round(lookback_24['Low'].min(), 2)

    sub_10 = df_h1.iloc[-11:-1]
    minor_res = round(sub_10['High'].max(), 2)
    minor_sup = round(sub_10['Low'].min(), 2)

    ema10_h1, ema21_h1, ema50_h1 = c0['EMA10'], c0['EMA21'], c0['EMA50']
    ema21_h4, ema50_h4 = df_h4['EMA21'].iloc[-1], df_h4['EMA50'].iloc[-1]
    h4_close = df_h4['Close'].iloc[-1]

    h4_uptrend = (h4_close >= ema21_h4 - 2.0) and (ema21_h4 >= ema50_h4 - 3.0)
    h4_downtrend = (h4_close <= ema21_h4 + 2.0) and (ema21_h4 <= ema50_h4 + 3.0)

    prior_lows = df_h1['Low'].iloc[-12:-2]
    has_prior_bottom_test = any(abs(lw - min(c0['Low'], c1['Low'])) <= 5.0 for lw in prior_lows)
    has_prior_top_test = any(abs(hg - max(c0['High'], c1['High'])) <= 5.0 for hg in df_h1['High'].iloc[-12:-2])

    at_buy_confluence = (
        (h4_uptrend and (c0['Low'] <= max(ema10_h1, ema21_h1) + 3.0)) or
        ((c0['Low'] <= sup_zone + 3.5 or c0['Low'] <= minor_sup + 2.5) and (h4_uptrend or has_prior_bottom_test))
    )
    at_sell_confluence = (
        (h4_downtrend and (c0['High'] >= min(ema10_h1, ema21_h1) - 3.0)) or
        ((c0['High'] >= res_zone - 3.5 or c0['High'] >= minor_res - 2.5) and (h4_downtrend or has_prior_top_test))
    )

    room_for_buy = h4_uptrend or ((res_zone - curr_price) >= 10.0)
    room_for_sell = h4_downtrend or ((curr_price - sup_zone) >= 10.0)

    r0 = max(c0['Range'], 0.1)
    body0 = abs(c0['Close'] - c0['Open'])
    u_wick0 = c0['High'] - max(c0['Open'], c0['Close'])
    l_wick0 = min(c0['Open'], c0['Close']) - c0['Low']

    r1 = max(c1['Range'], 0.1)
    body1 = abs(c1['Close'] - c1['Open'])

    bull_pinbar = (l_wick0 >= r0 * 0.45) and (u_wick0 <= r0 * 0.30) and (r0 >= 4.5)
    bear_pinbar = (u_wick0 >= r0 * 0.45) and (l_wick0 <= r0 * 0.30) and (r0 >= 4.5)

    is_inside_bar = (c1['High'] <= c2['High'] + 1.0) and (c1['Low'] >= c2['Low'] - 1.0)
    bull_fakey = is_inside_bar and (c0['Low'] < min(c1['Low'], c2['Low'])) and (c0['Close'] > c1['Low'] + r1 * 0.4) and (c0['Close'] > c0['Open'])
    bear_fakey = is_inside_bar and (c0['High'] > max(c1['High'], c2['High'])) and (c0['Close'] < c1['High'] - r1 * 0.4) and (c0['Close'] < c0['Open'])

    bull_engulfing = (c1['Close'] < c1['Open']) and (c0['Close'] > c0['Open']) and (body0 >= body1 * 1.05) and (c0['Close'] >= c1['High'] - 1.0) and (body0 >= r0 * 0.45)
    bear_engulfing = (c1['Close'] > c1['Open']) and (c0['Close'] < c0['Open']) and (body0 >= body1 * 1.05) and (c0['Close'] <= c1['Low'] + 1.0) and (body0 >= r0 * 0.45)

    bull_twobar = (c1['Close'] < c1['Open']) and (body1 >= 4.0) and (c0['Close'] > c0['Open']) and (body0 >= body1 * 0.80) and (abs(c0['Low'] - c1['Low']) <= 3.0)
    bear_twobar = (c1['Close'] > c1['Open']) and (body1 >= 4.0) and (c0['Close'] < c0['Open']) and (body0 >= body1 * 0.80) and (abs(c0['High'] - c1['High']) <= 3.0)

    bull_trendbar = h4_uptrend and (c0['Close'] > c0['Open']) and (body0 >= r0 * 0.60) and (body0 >= atr * 0.65) and (curr_price > ema10_h1)
    bear_trendbar = h4_downtrend and (c0['Close'] < c0['Open']) and (body0 >= r0 * 0.60) and (body0 >= atr * 0.65) and (curr_price < ema10_h1)

    if is_choppy and not (bull_fakey or bear_fakey):
        return "STAND_ASIDE", None, "Thị trường đang trong Vùng Giằng Co.", res_zone, sup_zone, 0, 0

    if at_buy_confluence and room_for_buy:
        pattern_name = None
        if bull_fakey: pattern_name = f"FAKEY TĂNG (Quét đáy Inside Bar tại {minor_sup})"
        elif bull_pinbar: pattern_name = f"PIN BAR ĐUÔI DÀI TĂNG (Rút chân {l_wick0:.1f} giá)"
        elif bull_engulfing: pattern_name = f"BULLISH ENGULFING (Nhấn Chìm Tăng)"
        elif bull_twobar: pattern_name = f"HAI NẾN ĐẢO CHIỀU TĂNG"
        elif bull_trendbar: pattern_name = f"TREND BAR TĂNG THUẬN SÓNG H4"

        if pattern_name:
            pattern_low = min(c0['Low'], c1['Low'])
            raw_sl = round(pattern_low - 2.0, 2)
            dist_from_close = curr_price - raw_sl

            if dist_from_close > 15.0:
                watch_zone = round(pattern_low + (curr_price - pattern_low) * 0.45, 2)
                opt_sl = round(max(raw_sl, watch_zone - MAX_SL_DIST), 2)
                return "WATCH_BUY", pattern_name, "Canh ngầm vùng hồi", res_zone, sup_zone, watch_zone, opt_sl
            else:
                opt_sl = round(curr_price - np.clip(dist_from_close, MIN_SL_DIST, 15.0), 2)
                return "MARKET_BUY", pattern_name, "Vào BUY ngay!", res_zone, sup_zone, curr_price, opt_sl

    if at_sell_confluence and room_for_sell:
        pattern_name = None
        if bear_fakey: pattern_name = f"FAKEY GIẢM (Quét đỉnh Inside Bar tại {minor_res})"
        elif bear_pinbar: pattern_name = f"PIN BAR ĐUÔI DÀI GIẢM (Rút râu trên {u_wick0:.1f} giá)"
        elif bear_engulfing: pattern_name = f"BEARISH ENGULFING (Nhấn Chìm Giảm)"
        elif bear_twobar: pattern_name = f"HAI NẾN ĐẢO CHIỀU GIẢM"
        elif bear_trendbar: pattern_name = f"TREND BAR GIẢM THUẬN SÓNG H4"

        if pattern_name:
            pattern_high = max(c0['High'], c1['High'])
            raw_sl = round(pattern_high + 2.0, 2)
            dist_from_close = raw_sl - curr_price

            if dist_from_close > 15.0:
                watch_zone = round(pattern_high - (pattern_high - curr_price) * 0.45, 2)
                opt_sl = round(min(raw_sl, watch_zone + MAX_SL_DIST), 2)
                return "WATCH_SELL", pattern_name, "Canh ngầm vùng hồi", res_zone, sup_zone, watch_zone, opt_sl
            else:
                opt_sl = round(curr_price + np.clip(dist_from_close, MIN_SL_DIST, 15.0), 2)
                return "MARKET_SELL", pattern_name, "Vào SELL ngay!", res_zone, sup_zone, curr_price, opt_sl

    return "STAND_ASIDE", None, "Chưa có thiết lập Price Action tại Vùng Hợp Lưu.", res_zone, sup_zone, 0, 0

def main():
    print(f"=== BOT #1 (PRICE ACTION - MARKET ONLY): ({len(GROQ_KEYS_POOL)} KEY GROQ) ===")
    df_h1, df_h4, is_4h_closed, prog, c_close = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = c_close.strftime('%H:%M - %d/%m')
    t_id = f"#PA_{c_close.strftime('%d%m_%H')}H"

    check_active_and_watch_zones(df_h1, df_h4, time_str)
    check_and_send_weekly_summary()

    st = load_state()
    if st.get("last_processed_candle") == t_id:
        return
    st["last_processed_candle"] = t_id
    save_state(st)

    if c_close.hour in [2, 3, 4, 5]:
        return

    h4_last = df_h4.iloc[-1]
    ema21_h4 = round(h4_last['EMA21'], 2)
    ema50_h4 = round(h4_last['EMA50'], 2)

    if h4_last['Close'] > ema21_h4 and ema21_h4 >= ema50_h4:
        h4_trend_str = "XU HƯỚNG TĂNG (BULLISH)"
    elif h4_last['Close'] < ema21_h4 and ema21_h4 <= ema50_h4:
        h4_trend_str = "XU HƯỚNG GIẢM (BEARISH)"
    else:
        h4_trend_str = "CHUYỂN GIAO / TÍCH LŨY"

    mode, pattern_name, entry_note, res_zone, sup_zone, opt_entry, opt_sl = scan_traderviet_price_action(df_h1, df_h4)
    unprotected_trades = [t for t in st.get("active_trades", []) if not t.get("tp1_hit", False)]
    has_risk_trade = len(unprotected_trades) > 0

    if mode.startswith("MARKET_") and not has_risk_trade:
        act = "BUY" if "BUY" in mode else "SELL"
        risk = round(abs(opt_entry - opt_sl), 2)
        tp1_val = round(opt_entry + risk * 1.5, 2) if act == "BUY" else round(opt_entry - risk * 1.5, 2)
        tp2_val = round(opt_entry + risk * 2.6, 2) if act == "BUY" else round(opt_entry - risk * 2.6, 2)

        sl_pips = int(round(risk * 10))
        tp1_pips = int(round(abs(tp1_val - opt_entry) * 10))
        tp2_pips = int(round(abs(tp2_val - opt_entry) * 10))

        chart_file = generate_pa_chart(df_h1, res_zone, sup_zone, opt_entry, f"[BOT #1: MARKET {act}] @ {opt_entry}")
        caption = (
            f"🟢 [BOT #1: VÀO LỆNH {act} MARKET NGAY]\n"
            f"🎫 Mã vé: {t_id} | ⏰ {time_str}\n"
            f"📊 H4: {h4_trend_str}\n"
            f"🕯️ Mô hình: {pattern_name}\n"
            f"📍 Entry Market: {opt_entry}\n"
            f"🛑 SL: {opt_sl} (-{sl_pips} pips)\n"
            f"🎯 TP1: {tp1_val} (+{tp1_pips} pips) -> Dời BE\n"
            f"🚀 TP2: {tp2_val} (+{tp2_pips} pips)"
        )
        st['active_trades'].append({
            "id": t_id, "action": act, "timeframe": "H1_H4_PA",
            "entry": opt_entry, "sl": opt_sl, "tp1": tp1_val, "tp2": tp2_val,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
            "tp1_hit": False, "hours_held": 0
        })
        save_state(st)
        send_telegram_photo(chart_file, caption)
        if os.path.exists(chart_file):
            os.remove(chart_file)

    elif mode.startswith("WATCH_") and not has_risk_trade:
        # CANH NGẦM - KHÔNG GỬI TIN NHẮN CHỜ LIMIT LÊN TELEGRAM
        act = "BUY" if "BUY" in mode else "SELL"
        st['watch_zones'] = [{
            "id": t_id, "action": act, "pattern": pattern_name,
            "zone_price": opt_entry, "sl": opt_sl,
            "res_zone": res_zone, "sup_zone": sup_zone, "wait_hours": 0
        }]
        save_state(st)
        print(f"👀 [{t_id}] Đang canh ngầm vùng {act} tại {opt_entry} (SL {opt_sl}). Khi giá về & giữ cấu trúc sẽ báo Market ngay!")
    else:
        print(f"🚫 [{t_id}] Đứng ngoài: {entry_note}")

if __name__ == "__main__":
    main()

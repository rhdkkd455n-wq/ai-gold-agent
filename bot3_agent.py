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

BOT3_TOKEN = os.environ.get("BOT3_TELEGRAM_TOKEN", "").strip()
MAIN_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_BOT_TOKEN = BOT3_TOKEN if BOT3_TOKEN else MAIN_TOKEN

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

STATE_FILE = "bot3_state.json"
VN_TZ = timezone(timedelta(hours=7))

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

def query_groq_with_key_rotation(prompt):
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
                        {"role": "system", "content": "Bạn là Chuyên gia Giao dịch Định lượng Volume Profile & CVD Order Flow. Phân tích ngắn gọn bằng tiếng Việt."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=700,
                    temperature=0.25,
                    timeout=25.0
                )
                ans = res.choices[0].message.content
                if ans and ans.strip():
                    return ans.strip()
            except Exception as e:
                err_str = str(e)
                if any(code in err_str.lower() for code in ["401", "invalid_api_key", "429", "rate_limit", "organization_restricted", "model_decommissioned", "model_not_found"]):
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
        f"📅 [BÁO CÁO TỔNG KẾT TUẦN - BOT #3 DAY TRADING]\n"
        f"🗓️ Kỳ báo cáo: Tuần {week_num}/{year}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"• Tổng lệnh: {total_trades} | Thắng: {len(win_trades)} ✅ | Thua: {len(loss_trades)} 🛑\n"
        f"• Winrate: {win_rate}% | Ròng: {net_str} PIPS 💰\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{details_str}"
    )
    send_telegram_msg(summary_msg)
    state["last_weekly_report"] = week_id
    state["closed_trades"] = []
    save_state(state)

def get_spot_gold_h1():
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY!")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=120&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"

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

    df['Range'] = df['High'] - df['Low']
    df['Volume'] = (df['Range'] * 1200).astype(int) + 600
    df_h1 = df.iloc[:-1].copy()

    body = df_h1['Close'] - df_h1['Open']
    u_wick = df_h1['High'] - df_h1[['Open', 'Close']].max(axis=1)
    l_wick = df_h1[['Open', 'Close']].min(axis=1) - df_h1['Low']

    eff_score = np.where(df_h1['Range'] > 0, (body + (l_wick - u_wick) * 0.6) / df_h1['Range'], 0.0)
    buy_ratio = np.clip(0.5 + eff_score * 0.45, 0.12, 0.88)

    df_h1['Buy_Vol']  = (df_h1['Volume'] * buy_ratio * 15).astype(int)
    df_h1['Sell_Vol'] = (df_h1['Volume'] * (1 - buy_ratio) * 15).astype(int)
    df_h1['Delta']    = df_h1['Buy_Vol'] - df_h1['Sell_Vol']
    df_h1['CVD_6']    = df_h1['Delta'].rolling(6).sum()

    typ_price = (df_h1['High'] + df_h1['Low'] + df_h1['Close']) / 3.0
    df_h1['VWAP'] = (typ_price * df_h1['Volume']).rolling(24).sum() / df_h1['Volume'].rolling(24).sum()
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    df_h1['EMA50'] = df_h1['Close'].ewm(span=50, adjust=False).mean()
    df_h1['Vol_MA20'] = df_h1['Volume'].rolling(20).mean()

    h_l = df_h1['High'] - df_h1['Low']
    h_c = np.abs(df_h1['High'] - df_h1['Close'].shift())
    l_c = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()

    candle_open = df_h1.index[-1]
    candle_close = candle_open + timedelta(hours=1)
    return df_h1.tail(36), candle_close

def calculate_advanced_volume_profile(df, n_bins=32):
    p_min, p_max = df['Low'].min(), df['High'].max()
    bins = np.linspace(p_min, p_max, n_bins)
    vol_prof = np.zeros(n_bins - 1)
    for _, r in df.iterrows():
        if r['High'] == r['Low'] or r['Volume'] == 0:
            continue
        for b in range(n_bins - 1):
            ov = max(0, min(r['High'], bins[b+1]) - max(r['Low'], bins[b]))
            if ov > 0:
                vol_prof[b] += r['Volume'] * (ov / (r['High'] - r['Low']))

    b_cents = (bins[:-1] + bins[1:]) / 2.0
    poc_idx = int(np.argmax(vol_prof))
    poc = round(b_cents[poc_idx], 2)
    tgt = np.sum(vol_prof) * 0.70
    left = right = poc_idx
    cur_v = vol_prof[poc_idx]
    while cur_v < tgt and (left > 0 or right < len(vol_prof) - 1):
        vl = vol_prof[left - 1] if left > 0 else 0
        vr = vol_prof[right + 1] if right < len(vol_prof) - 1 else 0
        if vl >= vr and left > 0:
            left -= 1; cur_v += vl
        elif right < len(vol_prof) - 1:
            right += 1; cur_v += vr
        else:
            if left > 0: left -= 1; cur_v += vl
            elif right < len(vol_prof) - 1: right += 1; cur_v += vr

    val = round(b_cents[left], 2)
    vah = round(b_cents[right], 2)
    va_slice = vol_prof[max(0, left):min(len(vol_prof), right+1)]
    lvn_price = round(b_cents[left + int(np.argmin(va_slice))], 2) if len(va_slice) > 0 else poc
    return poc, vah, val, lvn_price, b_cents, vol_prof

def generate_profile_chart(df, poc, vah, val, bin_centers, vol_prof, title_suffix=""):
    path = "xauusd_daytrade_chart.png"
    add_plots = [
        mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5),
        mpf.make_addplot(df['EMA50'], color='#00bcd4', width=1.4),
        mpf.make_addplot(df['VWAP'], color='#9c27b0', width=1.3, linestyle='--'),
        mpf.make_addplot(np.full(len(df), poc), color='#e91e63', width=2.0),
        mpf.make_addplot(np.full(len(df), vah), color='#2196f3', width=1.5),
        mpf.make_addplot(np.full(len(df), val), color='#4caf50', width=1.5)
    ]
    fig, axlist = mpf.plot(df, type='candle', style='binance', addplot=add_plots, title=f"\n[BOT #3: ORDER FLOW] {title_suffix}", figsize=(14, 7), returnfig=True)
    ax = axlist[0]
    max_v = np.max(vol_prof) if np.max(vol_prof) > 0 else 1
    for i in range(len(bin_centers)):
        bar_len = (vol_prof[i] / max_v) * 10
        col = '#e91e63' if abs(bin_centers[i] - poc) < 2 else '#607d8b'
        ax.barh(bin_centers[i], bar_len, left=len(df) - bar_len, height=(bin_centers[1]-bin_centers[0])*0.85, color=col, alpha=0.35, edgecolor=col)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def check_trades_and_watch_zones(df_h1, close_time, poc, vah, val, bin_centers, vol_prof):
    state = load_state()
    notified = set(state.get("notified_events", []))
    last_c = df_h1.iloc[-1]
    c_high, c_low, c_close = last_c['High'], last_c['Low'], last_c['Close']
    c_delta = last_c['Delta']
    closed_time_str = datetime.now(VN_TZ).strftime('%d/%m %H:%M')
    is_end_of_day = close_time.hour in [3, 4]

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
                    send_telegram_msg(f"🚀 [BOT #3 - FULL TP2] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time_str})
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [BOT #3 - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Entry ({entry})!")
                    notified.add(ev_key)
        elif act == "SELL":
            if c_high >= sl:
                sl_hit = True
                is_closed = True
            elif c_low <= tp2:
                ev_key = f"{t_id}_TP2"
                if ev_key not in notified:
                    send_telegram_msg(f"🚀 [BOT #3 - FULL TP2] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time_str})
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [BOT #3 - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Entry ({entry})!")
                    notified.add(ev_key)

        if sl_hit:
            if trade.get('tp1_hit', False):
                ev_key = f"{t_id}_BE"
                if ev_key not in notified:
                    send_telegram_msg(f"🛡️ [BOT #3 - CHẠM HÒA VỐN BE] Kèo {act} ({t_id}) đóng tại Entry ({entry}) sau khi đạt TP1 (+{trade['tp1_pips']} pips).")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "TP1 + BE", "pips": trade['tp1_pips'], "closed_at": closed_time_str})
            else:
                ev_key = f"{t_id}_SL"
                if ev_key not in notified:
                    send_telegram_msg(f"🛑 [BOT #3 - CHẠM SL] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "CẮN SL", "pips": -trade['sl_pips'], "closed_at": closed_time_str})
                    pm_prompt = f"Giải phẫu lệnh Day Trade {act} ({t_id}) vừa chạm SL tại {sl} (Entry: {entry}). High={c_high}, Low={c_low}, Close={c_close}, Delta={c_delta:,} HĐ. Viết 3 gạch đầu dòng ngắn gọn."
                    pm_analysis = query_groq_with_key_rotation(pm_prompt)
                    if pm_analysis:
                        send_telegram_msg(f"🔍 [GIẢI PHẪU LỆNH BOT #3: {t_id}]\n\n{pm_analysis}")

        if not is_closed and is_end_of_day:
            ev_key = f"{t_id}_EOD"
            if ev_key not in notified:
                raw_pips = int(round((c_close - entry) * 10)) if act == "BUY" else int(round((entry - c_close) * 10))
                final_pips = max(raw_pips, trade.get('tp1_pips', 0)) if trade.get('tp1_hit', False) else raw_pips
                p_str = f"+{final_pips}" if final_pips >= 0 else f"{final_pips}"
                send_telegram_msg(f"🌙 [BOT #3 - ĐÓNG CUỐI NGÀY] Đóng kèo {act} ({t_id}) tại {c_close} ({p_str} pips).")
                notified.add(ev_key)
                state["closed_trades"].append({"id": t_id, "action": act, "result": "ĐÓNG CUỐI NGÀY", "pips": final_pips, "closed_at": closed_time_str})
            is_closed = True

        if not is_closed:
            remaining_active.append(trade)

    # --- 2. KIỂM TRA VÙNG CANH NGẦM -> GIÁ ĐẾN & GIỮ CẤU TRÚC THÌ BÁO VÀO MARKET NGAY ---
    watch_list = state.get("watch_zones", [])
    remaining_watch = []
    has_unprotected = any(not t.get("tp1_hit", False) for t in remaining_active)
    ema21 = last_c['EMA21']
    ema50 = last_c['EMA50']

    for w in watch_list:
        w_id, w_act = w['id'], w['action']
        zone_price, sl = float(w['zone_price']), float(w['sl'])
        wait_h = w.get("wait_hours", 0) + 1
        w["wait_hours"] = wait_h

        if wait_h > 5 or is_end_of_day:
            continue

        if w_act == "BUY":
            if c_low <= sl or ema21 < ema50 - 3.0:
                continue
            if c_low <= zone_price + 1.5 and c_close > sl + 1.5 and not has_unprotected:
                curr_entry = round(c_close, 2)
                risk = round(np.clip(curr_entry - sl, 8.5, 13.5), 2)
                opt_sl = round(curr_entry - risk, 2)
                tp1_calc = round(curr_entry + risk * 1.15, 2)
                tp2_calc = round(curr_entry + risk * 2.1, 2)
                sl_pips = int(round(risk * 10))
                tp1_pips = int(round(abs(tp1_calc - curr_entry) * 10))
                tp2_pips = int(round(abs(tp2_calc - curr_entry) * 10))

                chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_prof, f"MARKET BUY @ {curr_entry}")
                caption = (
                    f"🟢 [BOT #3: GIÁ ĐÃ VỀ VÙNG - VÀO BUY MARKET NGAY]\n"
                    f"🎫 Mã vé: {w_id} | ⏰ {close_time.strftime('%H:%M - %d/%m')}\n"
                    f"⚡ Xác nhận: Giá test vùng {zone_price} & giữ cấu trúc Tăng\n"
                    f"📍 Entry Market ngay: {curr_entry}\n"
                    f"🛑 SL: {opt_sl} (-{sl_pips} pips)\n"
                    f"🎯 TP1: {tp1_calc} (+{tp1_pips} pips) -> Dời BE\n"
                    f"🚀 TP2: {tp2_calc} (+{tp2_pips} pips)"
                )
                send_telegram_photo(chart_file, caption)
                if os.path.exists(chart_file):
                    os.remove(chart_file)
                remaining_active.append({
                    "id": w_id, "action": "BUY", "timeframe": "H1_CTE_MKT",
                    "entry": curr_entry, "sl": opt_sl, "tp1": tp1_calc, "tp2": tp2_calc,
                    "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
                })
                has_unprotected = True
                continue

        elif w_act == "SELL":
            if c_high >= sl or ema21 > ema50 + 3.0:
                continue
            if c_high >= zone_price - 1.5 and c_close < sl - 1.5 and not has_unprotected:
                curr_entry = round(c_close, 2)
                risk = round(np.clip(sl - curr_entry, 8.5, 13.5), 2)
                opt_sl = round(curr_entry + risk, 2)
                tp1_calc = round(curr_entry - risk * 1.15, 2)
                tp2_calc = round(curr_entry - risk * 2.1, 2)
                sl_pips = int(round(risk * 10))
                tp1_pips = int(round(abs(curr_entry - tp1_calc) * 10))
                tp2_pips = int(round(abs(curr_entry - tp2_calc) * 10))

                chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_prof, f"MARKET SELL @ {curr_entry}")
                caption = (
                    f"🔴 [BOT #3: GIÁ ĐÃ VỀ VÙNG - VÀO SELL MARKET NGAY]\n"
                    f"🎫 Mã vé: {w_id} | ⏰ {close_time.strftime('%H:%M - %d/%m')}\n"
                    f"⚡ Xác nhận: Giá test vùng {zone_price} & giữ cấu trúc Giảm\n"
                    f"📍 Entry Market ngay: {curr_entry}\n"
                    f"🛑 SL: {opt_sl} (-{sl_pips} pips)\n"
                    f"🎯 TP1: {tp1_calc} (+{tp1_pips} pips) -> Dời BE\n"
                    f"🚀 TP2: {tp2_calc} (+{tp2_pips} pips)"
                )
                send_telegram_photo(chart_file, caption)
                if os.path.exists(chart_file):
                    os.remove(chart_file)
                remaining_active.append({
                    "id": w_id, "action": "SELL", "timeframe": "H1_CTE_MKT",
                    "entry": curr_entry, "sl": opt_sl, "tp1": tp1_calc, "tp2": tp2_calc,
                    "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
                })
                has_unprotected = True
                continue

        remaining_watch.append(w)

    state['active_trades'] = remaining_active
    state['watch_zones'] = remaining_watch
    state['notified_events'] = list(notified)
    save_state(state)

def evaluate_smart_orderflow_setup(df_h1, poc, vah, val):
    c0 = df_h1.iloc[-1]
    c1 = df_h1.iloc[-2]

    curr_price = round(c0['Close'], 2)
    ema21 = c0['EMA21']
    ema50 = c0['EMA50']
    vwap = c0['VWAP'] if not np.isnan(c0['VWAP']) else poc
    cvd_6 = c0['CVD_6'] if not np.isnan(c0['CVD_6']) else 0
    atr = round(np.clip(c0['ATR'] if not np.isnan(c0['ATR']) else 6.5, 5.0, 12.0), 1)

    ema_slope = ema21 - df_h1['EMA21'].iloc[-4]
    u_wick0 = c0['High'] - max(c0['Open'], c0['Close'])
    l_wick0 = min(c0['Open'], c0['Close']) - c0['Low']

    is_downtrend = (ema21 < ema50 - 1.5 and curr_price <= ema50 + 2.0) or (curr_price < ema21 and curr_price < vwap and cvd_6 < -80000) or (ema_slope < -4.0)
    is_uptrend   = (ema21 > ema50 + 1.5 and curr_price >= ema50 - 2.0) or (curr_price > ema21 and curr_price > vwap and cvd_6 > 80000) or (ema_slope > 4.0)

    if is_downtrend and not is_uptrend:
        regime = "DOWNTREND"
    elif is_uptrend and not is_downtrend:
        regime = "UPTREND"
    else:
        regime = "BALANCED"

    recent_heavy_dump = any((r['Close'] < r['Open'] and (r['Open'] - r['Close']) >= 7.5 and r['Delta'] < -90000) for _, r in df_h1.iloc[-4:-1].iterrows())
    recent_heavy_pump = any((r['Close'] > r['Open'] and (r['Close'] - r['Open']) >= 7.5 and r['Delta'] > 90000) for _, r in df_h1.iloc[-4:-1].iterrows())
    is_climax_candle = c0['Range'] > max(atr * 1.4, 11.5)
    sl_buf = 2.5

    if regime == "DOWNTREND":
        touched_res = (
            (c0['High'] >= ema21 - 2.0) or
            (c0['High'] >= vwap - 2.0) or
            (abs(c0['High'] - poc) <= 3.5) or
            (abs(c0['High'] - val) <= 3.5 and curr_price < val)
        )
        bull_trap_rejected = (c1['Close'] > c1['Open'] and c0['Close'] < c0['Open'] and c0['Close'] <= c1['Low'] + c1['Range']*0.4)
        bear_candle_confirm = (c0['Close'] < c0['Open'] and c0['Delta'] < 0) or (u_wick0 >= c0['Range'] * 0.40 and c0['Close'] < ema21)

        if touched_res and (bear_candle_confirm or bull_trap_rejected):
            setup_name = "TREND PULLBACK SELL"
            raw_sl = round(max(c0['High'], c1['High']) + sl_buf, 2)
            dist = raw_sl - curr_price
            if is_climax_candle or dist > 13.0:
                watch_p = round(c0['High'] - c0['Range'] * 0.45, 2)
                opt_sl = round(min(raw_sl, watch_p + 13.0), 2)
                return "WATCH_SELL", regime, setup_name, watch_p, opt_sl
            else:
                opt_sl = round(curr_price + np.clip(dist, 8.5, 13.5), 2)
                return "MARKET_SELL", regime, setup_name, curr_price, opt_sl

        val_breakdown = (c1['Close'] >= val - 1.0) and (curr_price < val - 1.5) and (curr_price >= val - 8.0) and (c0['Close'] < c0['Open']) and (c0['Delta'] < -30000)
        if val_breakdown and not is_climax_candle:
            setup_name = "INITIATIVE VAL BREAKDOWN SELL"
            opt_sl = round(curr_price + np.clip(max(c0['High'] - curr_price + sl_buf, 8.5), 8.5, 13.5), 2)
            return "MARKET_SELL", regime, setup_name, curr_price, opt_sl

    elif regime == "UPTREND":
        touched_sup = (
            (c0['Low'] <= ema21 + 2.0) or
            (c0['Low'] <= vwap + 2.0) or
            (abs(c0['Low'] - poc) <= 3.5) or
            (abs(c0['Low'] - vah) <= 3.5 and curr_price > vah)
        )
        bear_trap_rejected = (c1['Close'] < c1['Open'] and c0['Close'] > c0['Open'] and c0['Close'] >= c1['High'] - c1['Range']*0.4)
        bull_candle_confirm = (c0['Close'] > c0['Open'] and c0['Delta'] > 0) or (l_wick0 >= c0['Range'] * 0.40 and c0['Close'] > ema21)

        if touched_sup and (bull_candle_confirm or bear_trap_rejected) and not recent_heavy_dump:
            setup_name = "TREND PULLBACK BUY"
            raw_sl = round(min(c0['Low'], c1['Low']) - sl_buf, 2)
            dist = curr_price - raw_sl
            if is_climax_candle or dist > 13.0:
                watch_p = round(c0['Low'] + c0['Range'] * 0.45, 2)
                opt_sl = round(max(raw_sl, watch_p - 13.0), 2)
                return "WATCH_BUY", regime, setup_name, watch_p, opt_sl
            else:
                opt_sl = round(curr_price - np.clip(dist, 8.5, 13.5), 2)
                return "MARKET_BUY", regime, setup_name, curr_price, opt_sl

        vah_breakout = (c1['Close'] <= vah + 1.0) and (curr_price > vah + 1.5) and (curr_price <= vah + 8.0) and (c0['Close'] > c0['Open']) and (c0['Delta'] > 30000)
        if vah_breakout and not is_climax_candle and not recent_heavy_dump:
            setup_name = "INITIATIVE VAH BREAKOUT BUY"
            opt_sl = round(curr_price - np.clip(max(curr_price - c0['Low'] + sl_buf, 8.5), 8.5, 13.5), 2)
            return "MARKET_BUY", regime, setup_name, curr_price, opt_sl

    else:
        va_width = max(vah - val, 10.0)
        in_middle_poc = abs(curr_price - poc) < va_width * 0.30
        if not in_middle_poc:
            if (c0['Low'] <= val + 2.5) and (curr_price > val + 0.5) and (c1['Low'] >= val - 2.0) and not recent_heavy_dump:
                if (c0['Close'] > c0['Open'] and l_wick0 >= c0['Range'] * 0.35 and cvd_6 > -60000):
                    setup_name = "BALANCED VAL REVERSAL BUY"
                    opt_sl = round(curr_price - np.clip(curr_price - min(c0['Low'], val) + sl_buf, 8.5, 13.0), 2)
                    return "MARKET_BUY", regime, setup_name, curr_price, opt_sl

            if (c0['High'] >= vah - 2.5) and (curr_price < vah - 0.5) and (c1['High'] <= vah + 2.0) and not recent_heavy_pump:
                if (c0['Close'] < c0['Open'] and u_wick0 >= c0['Range'] * 0.35 and cvd_6 < 60000):
                    setup_name = "BALANCED VAH REVERSAL SELL"
                    opt_sl = round(curr_price + np.clip(max(c0['High'], vah) - curr_price + sl_buf, 8.5, 13.0), 2)
                    return "MARKET_SELL", regime, setup_name, curr_price, opt_sl

    return "STAND_ASIDE", regime, "Chưa có điểm vào tối ưu.", 0, 0

def main():
    print(f"=== BOT #3 (ORDER FLOW - MARKET ONLY): {len(GROQ_KEYS_POOL)} KEY GROQ ===")
    df_h1, close_time = get_spot_gold_h1()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    t_id = f"#OF_H1_{close_time.strftime('%d%m_%H')}H"

    poc, vah, val, lvn_price, bin_centers, vol_prof = calculate_advanced_volume_profile(df_h1)
    check_trades_and_watch_zones(df_h1, close_time, poc, vah, val, bin_centers, vol_prof)
    check_and_send_weekly_summary()

    st = load_state()
    if st.get("last_processed_candle") == t_id:
        return
    st["last_processed_candle"] = t_id
    save_state(st)

    if close_time.hour in [2, 3, 4, 5, 6]:
        return

    c_last = df_h1.iloc[-1]
    delta_val = int(c_last['Delta'])
    delta_str = f"+{delta_val:,} HĐ" if delta_val >= 0 else f"{delta_val:,} HĐ"
    cvd6_val = int(c_last['CVD_6']) if not np.isnan(c_last['CVD_6']) else 0
    cvd6_str = f"+{cvd6_val:,} HĐ" if cvd6_val >= 0 else f"{cvd6_val:,} HĐ"
    vwap_val = round(c_last['VWAP'], 2) if not np.isnan(c_last['VWAP']) else poc
    ema21_val = round(c_last['EMA21'], 2)
    ema50_val = round(c_last['EMA50'], 2)

    mode, regime, setup_type, opt_entry, opt_sl = evaluate_smart_orderflow_setup(df_h1, poc, vah, val)
    unprotected_trades = [t for t in st.get("active_trades", []) if not t.get("tp1_hit", False)]
    has_risk_trade = len(unprotected_trades) > 0

    if mode.startswith("MARKET_") and not has_risk_trade:
        act_str = "BUY" if "BUY" in mode else "SELL"
        risk_pts = round(abs(opt_entry - opt_sl), 2)
        tp1_calc = round(opt_entry + risk_pts * 1.15, 2) if act_str == "BUY" else round(opt_entry - risk_pts * 1.15, 2)
        tp2_calc = round(opt_entry + risk_pts * 2.1, 2) if act_str == "BUY" else round(opt_entry - risk_pts * 2.1, 2)

        sl_pips = int(round(risk_pts * 10))
        tp1_pips = int(round(abs(tp1_calc - opt_entry) * 10))
        tp2_pips = int(round(abs(tp2_calc - opt_entry) * 10))

        chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_prof, f"MARKET {act_str} @ {opt_entry}")
        photo_caption = (
            f"🎯 [BOT #3: VÀO LỆNH {act_str} MARKET NGAY]\n"
            f"🎫 Mã vé: {t_id} | ⏰ {time_str}\n"
            f"🧭 Xu hướng: {regime} | Setup: {setup_type}\n"
            f"📊 EMA21: {ema21_val} | EMA50: {ema50_val} | VWAP: {vwap_val}\n"
            f"📐 VAH={vah} | POC={poc} | VAL={val}\n"
            f"🌊 CVD (6 nến): {cvd6_str} | Delta: {delta_str}\n\n"
            f"📍 Entry Market: {opt_entry}\n"
            f"🛑 SL: {opt_sl} (-{sl_pips} pips)\n"
            f"🎯 TP1: {tp1_calc} (+{tp1_pips} pips) -> Dời BE\n"
            f"🚀 TP2: {tp2_calc} (+{tp2_pips} pips)"
        )
        st['active_trades'].append({
            "id": t_id, "action": act_str, "timeframe": "H1_CTE",
            "entry": opt_entry, "sl": opt_sl, "tp1": tp1_calc, "tp2": tp2_calc,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
        send_telegram_photo(chart_file, photo_caption)
        if os.path.exists(chart_file):
            os.remove(chart_file)

    elif mode.startswith("WATCH_") and not has_risk_trade:
        # CANH NGẦM - KHÔNG GỬI TIN NHẮN CHỜ LIMIT LÊN TELEGRAM
        act_str = "BUY" if "BUY" in mode else "SELL"
        st['watch_zones'] = [{
            "id": t_id, "action": act_str,
            "zone_price": opt_entry, "sl": opt_sl, "wait_hours": 0
        }]
        save_state(st)
        print(f"👀 [{t_id}] Đang canh ngầm vùng {act_str} tại {opt_entry} (SL {opt_sl}). Khi giá về & giữ cấu trúc sẽ báo Market ngay!")
    else:
        print(f"🚫 [{t_id}] ({regime}) Đứng ngoài: {setup_type}")

if __name__ == "__main__":
    main()

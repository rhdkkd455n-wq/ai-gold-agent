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
        "pending_setups": [],
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

def query_groq_with_key_rotation(prompt):
    if not GROQ_KEYS_POOL:
        return "⚠️ Chưa cấu hình GROQ_API_KEY trong GitHub Secrets."

    models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
    for idx, key in enumerate(GROQ_KEYS_POOL):
        masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "Key"
        cl = Groq(api_key=key)
        for m in models:
            try:
                res = cl.chat.completions.create(
                    model=m,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Bạn là Giám đốc Giao dịch Định lượng chuyên phương pháp Trend-Aligned Volume Profile & CVD Order Flow "
                                "(Tuyệt đối tuân thủ xu hướng lớn EMA21/50 & CVD, không bắt dao rơi, không mua đuổi nến Climax). "
                                "Phân tích ngắn gọn, thực chiến bằng tiếng Việt."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.25,
                    timeout=25.0
                )
                ans = res.choices[0].message.content
                if ans and ans.strip():
                    print(f"✅ AI Bot #3 thành công với Key #{idx+1} ({masked_key}) - Model: {m}")
                    return ans.strip()
            except Exception as e:
                err_str = str(e)
                print(f"⚠️ Key #{idx+1} lỗi trên {m}: {err_str}")
                if any(code in err_str.lower() for code in ["401", "invalid_api_key", "429", "rate_limit", "organization_restricted", "model_decommissioned", "model_not_found"]):
                    break
                else:
                    time.sleep(1.5)
    return "⚠️ Các API Key Groq đang bận. Bám sát kỷ luật xu hướng & các mốc POC/VAH/VAL."

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
        f"📅 [BÁO CÁO TỔNG KẾT TUẦN - BOT #3 TREND & VOLUME PROFILE]\n"
        f"🗓️ Kỳ báo cáo: Tuần {week_num}/{year}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 THỐNG KÊ HIỆU SUẤT DAY TRADE:\n"
        f"• Tổng số kèo trong tuần: {total_trades} lệnh\n"
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

def check_trades_and_pending(df_h1, close_time):
    state = load_state()
    notified = set(state.get("notified_events", []))
    last_c = df_h1.iloc[-1]
    c_high, c_low, c_close = last_c['High'], last_c['Low'], last_c['Close']
    c_delta = last_c['Delta']
    closed_time_str = datetime.now(VN_TZ).strftime('%d/%m %H:%M')
    is_end_of_day = close_time.hour in [3, 4]

    # --- 1. KIỂM TRA LỆNH ĐANG CHẠY ---
    active_trades = state.get("active_trades", [])
    remaining_active = []
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
                    send_telegram_msg(f"🚀 [DAY TRADE - FULL TP2] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time_str})
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [CHẠM TP1 - DỜI HÒA VỐN]\nKèo {act} ({t_id}) đã khớp TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Entry ({entry}) bảo toàn 100% vốn!")
                    notified.add(ev_key)
        elif act == "SELL":
            if c_high >= sl:
                sl_hit = True
                is_closed = True
            elif c_low <= tp2:
                ev_key = f"{t_id}_TP2"
                if ev_key not in notified:
                    send_telegram_msg(f"🚀 [DAY TRADE - FULL TP2] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time_str})
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [CHẠM TP1 - DỜI HÒA VỐN]\nKèo {act} ({t_id}) đã khớp TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Entry ({entry}) bảo toàn 100% vốn!")
                    notified.add(ev_key)
                
        if sl_hit:
            if trade.get('tp1_hit', False):
                ev_key = f"{t_id}_BE"
                if ev_key not in notified:
                    send_telegram_msg(f"🛡️ [CHẠM HÒA VỐN BE] Kèo Day Trade {act} ({t_id}) đóng phần còn lại tại Entry ({entry}) sau khi đã bỏ túi TP1 (+{trade['tp1_pips']} pips).")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "TP1 + BE", "pips": trade['tp1_pips'], "closed_at": closed_time_str})
            else:
                ev_key = f"{t_id}_SL"
                if ev_key not in notified:
                    send_telegram_msg(f"🛑 [CẮN SL DAY TRADE] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "CẮN SL", "pips": -trade['sl_pips'], "closed_at": closed_time_str})
                    pm_prompt = f"Giải phẫu lệnh Day Trade {act} ({t_id}) vừa chạm SL tại {sl} (Entry: {entry}). High={c_high}, Low={c_low}, Close={c_close}, Delta={c_delta:,} HĐ. Viết 3 gạch đầu dòng ngắn gọn."
                    pm_analysis = query_groq_with_key_rotation(pm_prompt)
                    send_telegram_msg(f"🔍 [GIẢI PHẪU LỆNH: {t_id}]\n\n{pm_analysis}")

        if not is_closed and is_end_of_day:
            ev_key = f"{t_id}_EOD"
            if ev_key not in notified:
                raw_pips = int(round((c_close - entry) * 10)) if act == "BUY" else int(round((entry - c_close) * 10))
                final_pips = max(raw_pips, trade['tp1_pips']) if trade.get('tp1_hit', False) else raw_pips
                p_str = f"+{final_pips}" if final_pips >= 0 else f"{final_pips}"
                send_telegram_msg(f"🌙 [CHỐT SỔ CUỐI NGÀY - KHÔNG ÔM QUA ĐÊM]\nĐóng lệnh Day Trade {act} ({t_id}) tại giá {c_close} (Entry: {entry}).\n• Kết quả: {p_str} pips.")
                notified.add(ev_key)
                state["closed_trades"].append({"id": t_id, "action": act, "result": "ĐÓNG CUỐI NGÀY", "pips": final_pips, "closed_at": closed_time_str})
            is_closed = True
                
        if not is_closed:
            remaining_active.append(trade)

    # --- 2. KIỂM TRA LỆNH CHỜ TEST LẠI VÙNG (PENDING LIMIT RETEST) ---
    pending_list = state.get("pending_setups", [])
    remaining_pending = []
    has_unprotected = any(not t.get("tp1_hit", False) for t in remaining_active)

    for p in pending_list:
        p_id, p_act = p['id'], p['action']
        limit_entry, sl, tp1, tp2 = p['entry'], p['sl'], p['tp1'], p['tp2']
        wait_h = p.get("wait_hours", 0) + 1
        p["wait_hours"] = wait_h

        if wait_h > 4 or is_end_of_day:
            continue

        if p_act == "BUY":
            if c_low <= sl or (c_high >= tp1 and c_low > limit_entry + 1.5):
                continue
            if c_low <= limit_entry + 1.2 and c_close > sl + 1.5 and not has_unprotected:
                act_entry = round(min(c_close, limit_entry + 0.8), 2)
                sl_pips = int(round(abs(act_entry - sl) * 10))
                tp1_pips = int(round(abs(tp1 - act_entry) * 10))
                tp2_pips = int(round(abs(tp2 - act_entry) * 10))
                send_telegram_msg(
                    f"✅ [BOT #3: KHỚP LỆNH BUY SAU KHI TEST LẠI CẢN - {p_id}]\n"
                    f"📍 Entry thực tế: {act_entry}\n"
                    f"🛑 SL ngắn: {sl} (-{sl_pips} pips)\n"
                    f"🎯 TP1: {tp1} (+{tp1_pips} pips) -> Dời BE\n"
                    f"🚀 TP2: {tp2} (+{tp2_pips} pips)"
                )
                remaining_active.append({
                    "id": p_id, "action": "BUY", "timeframe": "H1_CTE",
                    "entry": act_entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                    "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
                })
                has_unprotected = True
                continue
        elif p_act == "SELL":
            if c_high >= sl or (c_low <= tp1 and c_high < limit_entry - 1.5):
                continue
            if c_high >= limit_entry - 1.2 and c_close < sl - 1.5 and not has_unprotected:
                act_entry = round(max(c_close, limit_entry - 0.8), 2)
                sl_pips = int(round(abs(sl - act_entry) * 10))
                tp1_pips = int(round(abs(act_entry - tp1) * 10))
                tp2_pips = int(round(abs(act_entry - tp2) * 10))
                send_telegram_msg(
                    f"✅ [BOT #3: KHỚP LỆNH SELL SAU KHI TEST LẠI CẢN - {p_id}]\n"
                    f"📍 Entry thực tế: {act_entry}\n"
                    f"🛑 SL ngắn: {sl} (-{sl_pips} pips)\n"
                    f"🎯 TP1: {tp1} (+{tp1_pips} pips) -> Dời BE\n"
                    f"🚀 TP2: {tp2} (+{tp2_pips} pips)"
                )
                remaining_active.append({
                    "id": p_id, "action": "SELL", "timeframe": "H1_CTE",
                    "entry": act_entry, "sl": sl, "tp1": tp1, "tp2": tp2,
                    "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
                })
                has_unprotected = True
                continue

        remaining_pending.append(p)

    state['active_trades'] = remaining_active
    state['pending_setups'] = remaining_pending
    state['notified_events'] = list(notified)
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
    
    # TÍNH ORDER FLOW CHÍNH XÁC HƠN: Xét cả hướng thân nến và lực râu nến từ chối (Passive Absorption)
    body = df_h1['Close'] - df_h1['Open']
    u_wick = df_h1['High'] - df_h1[['Open', 'Close']].max(axis=1)
    l_wick = df_h1[['Open', 'Close']].min(axis=1) - df_h1['Low']
    
    # Hiệu chỉnh lực mua/bán thực tế dựa trên kết quả đấu giá (Body + Wick Rejection)
    eff_score = np.where(df_h1['Range'] > 0, (body + (l_wick - u_wick) * 0.6) / df_h1['Range'], 0.0)
    buy_ratio = np.clip(0.5 + eff_score * 0.45, 0.12, 0.88)
    
    df_h1['Buy_Vol']  = (df_h1['Volume'] * buy_ratio * 15).astype(int)
    df_h1['Sell_Vol'] = (df_h1['Volume'] * (1 - buy_ratio) * 15).astype(int)
    df_h1['Delta']    = df_h1['Buy_Vol'] - df_h1['Sell_Vol']
    
    # TỔNG DELTA TÍCH LŨY 6 NẾN (CVD_6) ĐỂ ĐỌC DÒNG TIỀN CHỦ ĐẠO (CHỐNG BẪY 1 NẾN CAO)
    df_h1['CVD_6'] = df_h1['Delta'].rolling(6).sum()
    
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
    fig, axlist = mpf.plot(df, type='candle', style='binance', addplot=add_plots, title=f"\n[BOT #3: TREND & ORDER FLOW] {title_suffix}", figsize=(14, 7), returnfig=True)
    ax = axlist[0]
    max_v = np.max(vol_prof) if np.max(vol_prof) > 0 else 1
    for i in range(len(bin_centers)):
        bar_len = (vol_prof[i] / max_v) * 10
        col = '#e91e63' if abs(bin_centers[i] - poc) < 2 else '#607d8b'
        ax.barh(bin_centers[i], bar_len, left=len(df) - bar_len, height=(bin_centers[1]-bin_centers[0])*0.85, color=col, alpha=0.35, edgecolor=col)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def build_orderflow_dossier(df_h1):
    dossier = []
    for t, r in df_h1.tail(5).iterrows():
        tot = r['Buy_Vol'] + r['Sell_Vol']
        b_pct = int((r['Buy_Vol'] / tot) * 100) if tot > 0 else 50
        s_pct = 100 - b_pct
        d_str = f"+{r['Delta']:,}" if r['Delta'] >= 0 else f"{r['Delta']:,}"
        dossier.append(f"• Nến {t.strftime('%H:%M')}: Mua={b_pct}%, Bán={s_pct}%, Delta={d_str} HĐ, Close={r['Close']:.2f}")
    return "\n".join(dossier)

def evaluate_smart_orderflow_setup(df_h1, poc, vah, val):
    """
    THUẬT TOÁN MỚI CHỐNG DÍNH SL NGƯỢC XU HƯỚNG & CHỐNG BẪY 1 NẾN DATA CAO:
    1. Xác định Chế độ Xu hướng (Trend Regime) dựa trên EMA21, EMA50, VWAP và CVD_6 (Tổng Delta 6 nến):
       - STRONG_DOWNTREND: Giá dưới EMA21 & EMA50, hoặc EMA21 < EMA50 kèm CVD_6 âm nặng. -> CẤM TUYỆT ĐỐI LỆNH BUY!
       - STRONG_UPTREND: Giá trên EMA21 & EMA50, hoặc EMA21 > EMA50 kèm CVD_6 dương mạnh. -> CẤM TUYỆT ĐỐI LỆNH SELL!
       - BALANCED_RANGE: Khi EMA21 và EMA50 đi ngang đan xen nhau và CVD_6 cân bằng. -> Mới cho phép đánh Mean Reversion tại 2 biên VAH/VAL!
    2. Chống mua đỉnh / bán đáy nến có Data cao (Anti-Climax Trap):
       - Không bao giờ BUY ở đỉnh 1 cây nến xanh dài khi nến trước đó là nến xả mạnh.
       - Chỉ vào lệnh khi: Giá HỒI VỀ TEST CẢN (Pullback to POC / VWAP / EMA21 / VAH / VAL) và xuất hiện nến từ chối thuận xu hướng chính!
    """
    c0 = df_h1.iloc[-1]
    c1 = df_h1.iloc[-2]
    c2 = df_h1.iloc[-3]

    curr_price = round(c0['Close'], 2)
    ema21 = c0['EMA21']
    ema50 = c0['EMA50']
    vwap = c0['VWAP'] if not np.isnan(c0['VWAP']) else poc
    cvd_6 = c0['CVD_6'] if not np.isnan(c0['CVD_6']) else 0
    atr = round(np.clip(c0['ATR'] if not np.isnan(c0['ATR']) else 6.5, 5.0, 12.0), 1)

    ema_slope = ema21 - df_h1['EMA21'].iloc[-4]
    u_wick0 = c0['High'] - max(c0['Open'], c0['Close'])
    l_wick0 = min(c0['Open'], c0['Close']) - c0['Low']
    body0 = abs(c0['Close'] - c0['Open'])

    # --- BƯỚC 1: PHÂN LOẠI CHẾ ĐỘ XU HƯỚNG (TREND REGIME) ---
    is_downtrend = (ema21 < ema50 - 1.5 and curr_price <= ema50 + 2.0) or (curr_price < ema21 and curr_price < vwap and cvd_6 < -80000) or (ema_slope < -4.0)
    is_uptrend   = (ema21 > ema50 + 1.5 and curr_price >= ema50 - 2.0) or (curr_price > ema21 and curr_price > vwap and cvd_6 > 80000) or (ema_slope > 4.0)

    if is_downtrend and not is_uptrend:
        regime = "DOWNTREND (XU HƯỚNG GIẢM CHỦ ĐẠO - CẤM BUY BẮT ĐÁY)"
    elif is_uptrend and not is_downtrend:
        regime = "UPTREND (XU HƯỚNG TĂNG CHỦ ĐẠO - CẤM SELL BẮT ĐỈNH)"
    else:
        regime = "BALANCED (THỊ TRƯỜNG CÂN BẰNG / SIDEWAY)"

    # --- BƯỚC 2: BỘ LỌC CHỐNG ĐU ĐỈNH NẾN DATA CAO (ANTI-CLIMAX & ANTI-BULL/BEAR TRAP) ---
    # Nếu 3 nến gần nhất có 1 cây nến xả cực mạnh (Delta < -110,000 và thân dài > 8 giá), cấm nhảy vào BUY chỉ vì 1 cây nến hồi xanh!
    recent_heavy_dump = any((r['Close'] < r['Open'] and (r['Open'] - r['Close']) >= 7.5 and r['Delta'] < -90000) for _, r in df_h1.iloc[-4:-1].iterrows())
    recent_heavy_pump = any((r['Close'] > r['Open'] and (r['Close'] - r['Open']) >= 7.5 and r['Delta'] > 90000) for _, r in df_h1.iloc[-4:-1].iterrows())

    # Không bao giờ vào Market đuổi nếu bản thân cây nến c0 vừa dãn quá dài (> 12 giá)
    is_climax_candle = c0['Range'] > max(atr * 1.4, 11.5)

    # --- BƯỚC 3: CÁC THIẾT LẬP VÀO LỆNH THUẬN DÒNG TIỀN (HIGH WINRATE SETUPS) ---

    # A. KỊCH BẢN TRONG XU HƯỚNG GIẢM (CHỈ CANH SELL):
    if "DOWNTREND" in regime:
        # Setup Sell 1: Pullback Exhaustion (Giá hồi lên chạm EMA21 / VWAP / POC / VAH hoặc sau 1 cây nến hồi xanh bị hấp thụ và đóng nến đỏ/rút râu)
        touched_res = (
            (c0['High'] >= ema21 - 2.0) or
            (c0['High'] >= vwap - 2.0) or
            (abs(c0['High'] - poc) <= 3.5) or
            (abs(c0['High'] - val) <= 3.5 and curr_price < val) # Breakdown VAL rồi hồi lên retest VAL từ bên dưới!
        )
        # Bắt bài bẫy Bull Trap: Nến c1 là nến hồi xanh (nhỏ lẻ tưởng đảo chiều nhảy vào mua), nhưng nến c0 lập tức bị cá mập đạp đỏ hoặc rút râu trên dài!
        bull_trap_rejected = (c1['Close'] > c1['Open'] and c0['Close'] < c0['Open'] and c0['Close'] <= c1['Low'] + c1['Range']*0.4)
        bear_candle_confirm = (c0['Close'] < c0['Open'] and c0['Delta'] < 0) or (u_wick0 >= c0['Range'] * 0.40 and c0['Close'] < ema21)

        if touched_res and (bear_candle_confirm or bull_trap_rejected):
            setup_name = "TREND PULLBACK SELL (Hồi lên cản EMA21/POC/VAL trong xu hướng Giảm bị từ chối)"
            raw_sl = round(max(c0['High'], c1['High']) + 2.2, 2)
            dist = raw_sl - curr_price
            if is_climax_candle or dist > 11.5:
                retest_p = round(c0['High'] - c0['Range'] * 0.45, 2)
                opt_sl = round(min(raw_sl, retest_p + 10.5), 2)
                return "PENDING_SELL", regime, setup_name, retest_p, opt_sl
            else:
                opt_sl = round(curr_price + np.clip(dist, 6.5, 11.0), 2)
                return "MARKET_SELL", regime, setup_name, curr_price, opt_sl

        # Setup Sell 2: VAL Breakdown & Continuation (Giá phá thủng đáy VAL thuận xu hướng giảm và đóng nến dưới VAL)
        val_breakdown = (c1['Close'] >= val - 1.0) and (curr_price < val - 1.5) and (curr_price >= val - 8.0) and (c0['Close'] < c0['Open']) and (c0['Delta'] < -30000)
        if val_breakdown and not is_climax_candle:
            setup_name = "INITIATIVE VAL BREAKDOWN SELL (Phá thủng đáy VAL thuận xu hướng Giảm)"
            opt_sl = round(curr_price + np.clip(max(c0['High'] - curr_price + 1.8, 7.0), 6.5, 11.0), 2)
            return "MARKET_SELL", regime, setup_name, curr_price, opt_sl

    # B. KỊCH BẢN TRONG XU HƯỚNG TĂNG (CHỈ CANH BUY):
    elif "UPTREND" in regime:
        touched_sup = (
            (c0['Low'] <= ema21 + 2.0) or
            (c0['Low'] <= vwap + 2.0) or
            (abs(c0['Low'] - poc) <= 3.5) or
            (abs(c0['Low'] - vah) <= 3.5 and curr_price > vah) # Breakout VAH rồi nhúng xuống retest VAH từ bên trên!
        )
        bear_trap_rejected = (c1['Close'] < c1['Open'] and c0['Close'] > c0['Open'] and c0['Close'] >= c1['High'] - c1['Range']*0.4)
        bull_candle_confirm = (c0['Close'] > c0['Open'] and c0['Delta'] > 0) or (l_wick0 >= c0['Range'] * 0.40 and c0['Close'] > ema21)

        if touched_sup and (bull_candle_confirm or bear_trap_rejected) and not recent_heavy_dump:
            setup_name = "TREND PULLBACK BUY (Nhúng về hỗ trợ EMA21/POC/VAH trong xu hướng Tăng rút chân)"
            raw_sl = round(min(c0['Low'], c1['Low']) - 2.2, 2)
            dist = curr_price - raw_sl
            if is_climax_candle or dist > 11.5:
                retest_p = round(c0['Low'] + c0['Range'] * 0.45, 2)
                opt_sl = round(max(raw_sl, retest_p - 10.5), 2)
                return "PENDING_BUY", regime, setup_name, retest_p, opt_sl
            else:
                opt_sl = round(curr_price - np.clip(dist, 6.5, 11.0), 2)
                return "MARKET_BUY", regime, setup_name, curr_price, opt_sl

        vah_breakout = (c1['Close'] <= vah + 1.0) and (curr_price > vah + 1.5) and (curr_price <= vah + 8.0) and (c0['Close'] > c0['Open']) and (c0['Delta'] > 30000)
        if vah_breakout and not is_climax_candle and not recent_heavy_dump:
            setup_name = "INITIATIVE VAH BREAKOUT BUY (Phá vượt đỉnh VAH thuận xu hướng Tăng)"
            opt_sl = round(curr_price - np.clip(max(curr_price - c0['Low'] + 1.8, 7.0), 6.5, 11.0), 2)
            return "MARKET_BUY", regime, setup_name, curr_price, opt_sl

    # C. KỊCH BẢN THỊ TRƯỜNG CÂN BẰNG / SIDEWAY (CHỈ ĐÁNH ĐẢO CHIỀU TẠI BIÊN KHI ĐÃ CÓ XÁC NHẬN 2 NẾN & KHÔNG CÓ XẢ MẠNH):
    else:
        va_width = max(vah - val, 10.0)
        in_middle_poc = abs(curr_price - poc) < va_width * 0.30
        if not in_middle_poc:
            # Mua tại VAL chỉ khi KHÔNG có cú xả mạnh trước đó và giá đã giữ vững trên VAL 2 nến liên tiếp
            if (c0['Low'] <= val + 2.5) and (curr_price > val + 0.5) and (c1['Low'] >= val - 2.0) and not recent_heavy_dump:
                if (c0['Close'] > c0['Open'] and l_wick0 >= c0['Range'] * 0.35 and cvd_6 > -60000):
                    setup_name = "BALANCED VAL REVERSAL BUY (Thị trường Sideway - Hấp thụ tại biên dưới VAL)"
                    opt_sl = round(curr_price - np.clip(curr_price - min(c0['Low'], val) + 2.5, 6.5, 10.5), 2)
                    return "MARKET_BUY", regime, setup_name, curr_price, opt_sl

            # Bán tại VAH chỉ khi KHÔNG có cú bơm mạnh trước đó và giá bị từ chối dưới VAH
            if (c0['High'] >= vah - 2.5) and (curr_price < vah - 0.5) and (c1['High'] <= vah + 2.0) and not recent_heavy_pump:
                if (c0['Close'] < c0['Open'] and u_wick0 >= c0['Range'] * 0.35 and cvd_6 < 60000):
                    setup_name = "BALANCED VAH REVERSAL SELL (Thị trường Sideway - Hấp thụ tại biên trên VAH)"
                    opt_sl = round(curr_price + np.clip(max(c0['High'], vah) - curr_price + 2.5, 6.5, 10.5), 2)
                    return "MARKET_SELL", regime, setup_name, curr_price, opt_sl

    return "STAND_ASIDE", regime, "Chưa có điểm hồi Pullback thuận xu hướng hoặc đang chống lại dòng tiền CVD_6.", 0, 0

def main():
    print(f"=== BOT #3 (TREND-ALIGNED ORDER FLOW & VOLUME PROFILE): {len(GROQ_KEYS_POOL)} KEY GROQ ===")
    df_h1, close_time = get_spot_gold_h1()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    t_id = f"#OF_H1_{close_time.strftime('%d%m_%H')}H"
    
    check_trades_and_pending(df_h1, close_time)
    check_and_send_weekly_summary()
    
    st = load_state()
    if st.get("last_processed_candle") == t_id:
        print(f"⏭️ Nến {t_id} đã được xử lý. Bỏ qua!")
        return
    st["last_processed_candle"] = t_id
    save_state(st)

    if close_time.hour in [2, 3, 4, 5, 6]:
        print("🌙 Giờ nghỉ chuyển giao phiên: Tạm ngưng mở lệnh mới.")
        return

    poc, vah, val, lvn_price, bin_centers, vol_prof = calculate_advanced_volume_profile(df_h1)
    c_last = df_h1.iloc[-1]
    
    buy_v, sell_v = int(c_last['Buy_Vol']), int(c_last['Sell_Vol'])
    tot_v = buy_v + sell_v if (buy_v + sell_v) > 0 else 1
    buy_pct = int((buy_v / tot_v) * 100)
    sell_pct = 100 - buy_pct
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

    if mode != "STAND_ASIDE" and not has_risk_trade:
        act_str = "BUY" if "BUY" in mode else "SELL"
        risk_pts = round(abs(opt_entry - opt_sl), 2)
        
        if act_str == "BUY":
            tp1_calc = round(opt_entry + risk_pts * 1.3, 2)
            tp2_calc = round(opt_entry + risk_pts * 2.2, 2)
        else:
            tp1_calc = round(opt_entry - risk_pts * 1.3, 2)
            tp2_calc = round(opt_entry - risk_pts * 2.2, 2)

        sl_pips = int(round(risk_pts * 10))
        tp1_pips = int(round(abs(tp1_calc - opt_entry) * 10))
        tp2_pips = int(round(abs(tp2_calc - opt_entry) * 10))

        chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_prof, f"{act_str} | {regime.split(' (')[0]}")
        dossier_str = build_orderflow_dossier(df_h1)

        header_info = (
            f"🎯 [BOT #3: TREND-ALIGNED ORDER FLOW & VOLUME PROFILE]\n"
            f"🎫 Mã vé: {t_id} | ⏰ {time_str}\n"
            f"🧭 Chế độ Xu hướng: {regime}\n"
            f"📊 EMA21: {ema21_val} | EMA50: {ema50_val} | VWAP: {vwap_val}\n"
            f"📐 Volume Profile: VAH={vah} | POC={poc} | VAL={val}\n"
            f"🌊 Dòng tiền CVD (6 nến): {cvd6_str} | Nến hiện tại: {delta_str}\n"
            f"⚡ Setup: {setup_type}\n\n"
        )

        if mode.startswith("MARKET_"):
            prompt = (
                f"Phân tích kèo {act_str} Vàng (XAU/USD) thuận xu hướng theo Volume Profile & CVD Order Flow:\n"
                f"- Chế độ xu hướng: {regime}\n"
                f"- Setup: {setup_type}\n"
                f"- Tọa độ: VAH={vah} | POC={poc} | VAL={val} | EMA21={ema21_val} | EMA50={ema50_val}\n"
                f"- Tổng CVD 6 nến: {cvd6_str} | Delta nến tín hiệu: {delta_str}\n"
                f"- Entry: {opt_entry} | SL: {opt_sl} (-{sl_pips} pips) | TP1: {tp1_calc} (+{tp1_pips} pips) | TP2: {tp2_calc} (+{tp2_pips} pips)\n"
                f"- 5 nến gần nhất:\n{dossier_str}\n"
                f"Viết nhận định 3 mục ngắn gọn: 1. Sự đồng thuận giữa Xu hướng EMA và dòng tiền CVD_6. 2. Tại sao điểm vào này tránh được bẫy ngược xu hướng. 3. Quản lý lệnh."
            )
            ai_analysis = query_groq_with_key_rotation(prompt)

            photo_caption = (
                f"{header_info}"
                f"🚨 TÍN HIỆU THUẬN XU HƯỚNG: {act_str} VÀNG\n"
                f"📍 Entry tối ưu: {opt_entry}\n"
                f"🛑 Cắt lỗ ngắn (SL): {opt_sl} (-{sl_pips} pips)\n"
                f"🎯 TP1 (R:R 1:1.3 - Dời Hòa Vốn): {tp1_calc} (+{tp1_pips} pips)\n"
                f"🚀 TP2 (R:R 1:2.2 - Mục tiêu sóng): {tp2_calc} (+{tp2_pips} pips)"
            )
            st['active_trades'].append({
                "id": t_id, "action": act_str, "timeframe": "H1_CTE",
                "entry": opt_entry, "sl": opt_sl, "tp1": tp1_calc, "tp2": tp2_calc,
                "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
            })
            save_state(st)
            send_telegram_photo(chart_file, photo_caption)
            time.sleep(1)
            send_telegram_msg(f"💡 [PHÂN TÍCH DÒNG TIỀN CVD & XU HƯỚNG - {t_id}]\n\n{ai_analysis}")
        else:
            # Chế độ chờ giá hồi về test lại 45% cây nến khi nến tín hiệu quá dài
            photo_caption = (
                f"{header_info}"
                f"⏳ TÍN HIỆU CHỜ RETEST (KHÔNG ĐU ĐỈNH/ĐÁY NẾN DÀI):\n"
                f"📍 Giá đóng nến: {curr_price}\n"
                f"🎯 Vùng Chờ Khớp Lệnh ({act_str} LIMIT): {opt_entry}\n"
                f"🛑 Cắt lỗ ngắn (SL): {opt_sl} (-{sl_pips} pips)\n"
                f"💰 TP1 (R:R 1:1.3): {tp1_calc} (+{tp1_pips} pips) -> Dời BE\n"
                f"🚀 TP2 (R:R 1:2.2): {tp2_calc} (+{tp2_pips} pips)\n"
                f"👉 Bot sẽ tự động báo khớp lệnh khi giá hồi về {opt_entry}!"
            )
            st['pending_setups'] = [{
                "id": t_id, "action": act_str, "entry": opt_entry,
                "sl": opt_sl, "tp1": tp1_calc, "tp2": tp2_calc,
                "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips,
                "wait_hours": 0
            }]
            save_state(st)
            send_telegram_photo(chart_file, photo_caption)

        if os.path.exists(chart_file):
            os.remove(chart_file)
    else:
        print(f"🚫 [{t_id}] ({regime}) Đứng ngoài: {setup_type}")

if __name__ == "__main__":
    main()

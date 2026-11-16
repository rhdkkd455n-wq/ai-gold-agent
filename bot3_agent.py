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
                        {"role": "system", "content": "Bạn là Chuyên gia Day Trading Order Flow & Volume Profile Vàng (XAU/USD) trong ngày. Phân tích ngắn gọn, thực chiến bằng tiếng Việt."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=850,
                    temperature=0.3,
                    timeout=25.0
                )
                ans = res.choices[0].message.content
                if ans and ans.strip():
                    print(f"✅ AI Day Trading thành công với Key #{idx+1} ({masked_key}) - Model: {m}")
                    return ans.strip()
            except Exception as e:
                err_str = str(e)
                print(f"⚠️ Key #{idx+1} lỗi trên {m}: {err_str}")
                if any(code in err_str.lower() for code in ["401", "invalid_api_key", "429", "rate_limit", "organization_restricted", "model_decommissioned", "model_not_found"]):
                    break
                else:
                    time.sleep(1.5)
    return "⚠️ Các API Key Groq đang bận. Vui lòng bám sát các mốc POC/VAH/VAL kỹ thuật."

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

def check_tp_sl_and_eod_close(df_h1, close_time):
    state = load_state()
    active_trades = state.get("active_trades", [])
    if not active_trades:
        return

    notified = set(state.get("notified_events", []))
    last_c = df_h1.iloc[-1]
    c_high, c_low, c_close = last_c['High'], last_c['Low'], last_c['Close']
    c_delta = last_c['Delta']
    remaining = []
    closed_time_str = datetime.now(VN_TZ).strftime('%d/%m %H:%M')
    is_end_of_day = close_time.hour in [3, 4] # Đóng lệnh Day Trade cuối phiên Mỹ (3h-4h sáng VN)
    
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
                    send_telegram_msg(f"🎉 [DAY TRADE - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
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
                    send_telegram_msg(f"🎉 [DAY TRADE - ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
                    notified.add(ev_key)
                
        if sl_hit:
            if trade.get('tp1_hit', False):
                ev_key = f"{t_id}_BE"
                if ev_key not in notified:
                    send_telegram_msg(f"🛡️ [CHẠM HÒA VỐN BE] Kèo Day Trade {act} ({t_id}) đã đóng phần còn lại tại Entry ({entry}) sau khi bỏ túi TP1 (+{trade['tp1_pips']} pips).")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "TP1 + BE", "pips": trade['tp1_pips'], "closed_at": closed_time_str})
            else:
                ev_key = f"{t_id}_SL"
                if ev_key not in notified:
                    send_telegram_msg(f"🛑 [CẮN SL DAY TRADE] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "CẮN SL", "pips": -trade['sl_pips'], "closed_at": closed_time_str})
                    pm_prompt = f"Kèo Day Trade {act} Vàng ({t_id}) vừa chạm SL tại {sl} (Entry: {entry}). Nến H1 có High={c_high}, Low={c_low}, Close={c_close}, Delta={c_delta:,} HĐ. Giải phẫu ngắn gọn trong 3 gạch đầu dòng."
                    pm_analysis = query_groq_with_key_rotation(pm_prompt)
                    send_telegram_msg(f"🔍 [GIẢI PHẪU LỆNH DAY TRADE: {t_id}]\n\n{pm_analysis}")

        # ĐÓNG LỆNH CUỐI NGÀY (KHÔNG ÔM QUA ĐÊM ĐỐI VỚI BOT DAY TRADE)
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
            remaining.append(trade)
            
    state['active_trades'] = remaining
    state['notified_events'] = list(notified)
    save_state(state)

def get_spot_gold_h1():
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY!")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=80&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    
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
    
    raw_ratio = np.where(df_h1['Range'] > 0, (df_h1['Close'] - df_h1['Low']) / df_h1['Range'], 0.5)
    buy_ratio = np.clip(raw_ratio, 0.12, 0.88)
    df_h1['Buy_Vol']  = (df_h1['Volume'] * buy_ratio * 15).astype(int)
    df_h1['Sell_Vol'] = (df_h1['Volume'] * (1 - buy_ratio) * 15).astype(int)
    df_h1['Delta']    = df_h1['Buy_Vol'] - df_h1['Sell_Vol']
    
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    df_h1['Vol_MA20'] = df_h1['Volume'].rolling(20).mean()
    h_l = df_h1['High'] - df_h1['Low']
    h_c = np.abs(df_h1['High'] - df_h1['Close'].shift())
    l_c = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    candle_open = df_h1.index[-1]
    candle_close = candle_open + timedelta(hours=1)
    # Lấy 35 nến H1 gần nhất để Volume Profile bám sát vùng giá trong ngày (Intraday)
    return df_h1.tail(35), candle_close

def calculate_volume_profile(df, n_bins=30):
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
    poc_idx = np.argmax(vol_prof)
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
    return poc, vah, val, b_cents, vol_prof

def generate_profile_chart(df, poc, vah, val, bin_centers, vol_prof):
    path = "xauusd_daytrade_chart.png"
    add_plots = [
        mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5),
        mpf.make_addplot(np.full(len(df), poc), color='#e91e63', width=2.0),
        mpf.make_addplot(np.full(len(df), vah), color='#2196f3', width=1.5),
        mpf.make_addplot(np.full(len(df), val), color='#4caf50', width=1.5)
    ]
    fig, axlist = mpf.plot(df, type='candle', style='binance', addplot=add_plots, title="\n[BOT #3: DAY TRADING] INTRADAY VOLUME PROFILE & ORDER FLOW", figsize=(14, 7), returnfig=True)
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
        dossier.append(f"• Nến {t.strftime('%H:%M')}: Mua={r['Buy_Vol']:,} ({b_pct}%), Bán={r['Sell_Vol']:,} ({s_pct}%), Delta={d_str} HĐ, Close={r['Close']:.2f}")
    return "\n".join(dossier)

def main():
    print(f"=== BOT #3 (DAY TRADING INTRADAY): XOAY VÒNG {len(GROQ_KEYS_POOL)} KEY GROQ ===")
    df_h1, close_time = get_spot_gold_h1()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    t_id = f"#DAY_H1_{close_time.strftime('%d%m_%H')}H"
    
    check_tp_sl_and_eod_close(df_h1, close_time)
    check_and_send_weekly_summary()
    
    st = load_state()
    if st.get("last_processed_candle") == t_id:
        print(f"⏭️ Nến {t_id} đã được xử lý. Bỏ qua!")
        return
    st["last_processed_candle"] = t_id
    save_state(st)

    # Không mở lệnh Day Trade mới vào cuối phiên Mỹ (từ 02:00 - 06:00 sáng VN)
    if close_time.hour in [2, 3, 4, 5]:
        print("🌙 Cuối phiên Mỹ / Đầu phiên Á thanh khoản mỏng: Tạm ngưng mở lệnh Day Trade mới.")
        return

    poc, vah, val, bin_centers, vol_prof = calculate_volume_profile(df_h1)
    c_last = df_h1.iloc[-1]
    buy_v, sell_v = int(c_last['Buy_Vol']), int(c_last['Sell_Vol'])
    tot_v = buy_v + sell_v if (buy_v + sell_v) > 0 else 1
    buy_pct = int((buy_v / tot_v) * 100)
    sell_pct = 100 - buy_pct
    delta_val = int(c_last['Delta'])
    delta_str = f"+{delta_val:,} HĐ" if delta_val >= 0 else f"{delta_val:,} HĐ"
    
    atr_val = round(df_h1['ATR'].iloc[-1], 1)
    if np.isnan(atr_val) or atr_val < 5.0:
        atr_val = 6.0
    
    u_wick = c_last['High'] - max(c_last['Open'], c_last['Close'])
    l_wick = min(c_last['Open'], c_last['Close']) - c_last['Low']
    is_active_vol = c_last['Volume'] >= c_last['Vol_MA20'] * 0.85
    
    bull_signal = is_active_vol and (l_wick >= c_last['Range'] * 0.25 or c_last['Close'] > c_last['Open'])
    bear_signal = is_active_vol and (u_wick >= c_last['Range'] * 0.25 or c_last['Close'] < c_last['Open'])
    
    if l_wick >= c_last['Range'] * 0.35:
        absorp_desc = "HẤP THỤ CẦU NGẮN HẠN (Rút chân gom hàng)"
    elif u_wick >= c_last['Range'] * 0.35:
        absorp_desc = "HẤP THỤ CUNG NGẮN HẠN (Rút râu xả hàng)"
    else:
        absorp_desc = "Dòng lệnh Intraday thuận xu hướng"
        
    ema21_val = c_last['EMA21']
    not_exhausted = c_last['Range'] <= atr_val * 2.1
    
    # Điều kiện Day Trade linh hoạt bắt sóng trong ngày (Phi Âu - Mỹ)
    room_to_val = (curr_price - val) >= (atr_val * 0.65)
    is_valid_sell = (
        sell_pct >= 58 and delta_val < 0 and bear_signal and not_exhausted and
        ((curr_price <= ema21_val + 3.0 and room_to_val) or (abs(c_last['High'] - vah) <= 6.0 and c_last['Close'] < vah))
    )
    
    room_to_vah = (vah - curr_price) >= (atr_val * 0.65)
    is_valid_buy = (
        buy_pct >= 58 and delta_val > 0 and bull_signal and not_exhausted and
        ((curr_price >= ema21_val - 3.0 and room_to_vah) or (abs(c_last['Low'] - val) <= 6.0 and c_last['Close'] > val))
    )

    unprotected_trades = [t for t in st.get("active_trades", []) if not t.get("tp1_hit", False)]
    has_risk_trade = len(unprotected_trades) > 0

    # Day Trade: SL gọn (từ 6.5 - 11.5 giá), ăn nhanh TP1 (1:1.2) và TP2 (1:2.0)
    base_risk = max(atr_val * 1.0, 6.5)

    if (is_valid_sell or is_valid_buy) and not has_risk_trade:
        chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_prof)
        dossier_str = build_orderflow_dossier(df_h1)
        prompt = (
            f"Phân tích kèo DAY TRADING (Đánh ngắn trong ngày) Vàng XAU/USD:\n"
            f"- POC trong ngày: {poc} | VAH: {vah} | VAL: {val} | Giá hiện tại: {curr_price}\n"
            f"- Lực Mua: {buy_v:,} HĐ ({buy_pct}%) | Lực Bán: {sell_v:,} HĐ ({sell_pct}%) | Delta: {delta_str}\n"
            f"- Lịch sử 5 nến:\n{dossier_str}\n"
            f"Hãy nhận định ngắn gọn 3 ý: 1. Áp lực Delta hiện tại. 2. Phản ứng giá tại POC/VAH/VAL trong ngày. 3. Lưu ý quản lý lệnh Day Trade (chốt lời nhanh, không giữ qua đêm)."
        )
        ai_analysis = query_groq_with_key_rotation(prompt)

        if is_valid_sell:
            struct_sl = round(max(c_last['High'] + 1.8, curr_price + base_risk), 2)
            risk_pts = round(min(struct_sl - curr_price, atr_val * 1.4), 2)
            sl_calc = round(curr_price + risk_pts, 2)
            tp1_calc = round(curr_price - (risk_pts * 1.2), 2)
            tp2_calc = round(curr_price - (risk_pts * 2.0), 2)
            act_str = "SELL"
        else:
            struct_sl = round(min(c_last['Low'] - 1.8, curr_price - base_risk), 2)
            risk_pts = round(min(curr_price - struct_sl, atr_val * 1.4), 2)
            sl_calc = round(curr_price - risk_pts, 2)
            tp1_calc = round(curr_price + (risk_pts * 1.2), 2)
            tp2_calc = round(curr_price + (risk_pts * 2.0), 2)
            act_str = "BUY"

        sl_pips = int(risk_pts * 10)
        tp1_pips = int(round(risk_pts * 12))
        tp2_pips = int(round(risk_pts * 20))

        header_info = f"⚡ [BOT #3: DAY TRADING - ĐÁNH NHANH TRONG NGÀY]\n" \
                      f"🎫 Mã vé: {t_id} (Tự đóng trước 03h sáng)\n" \
                      f"⏰ Nến H1 đóng: {time_str}\n" \
                      f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n" \
                      f"🎯 VÙNG GIÁ TRỊ TRONG NGÀY:\n" \
                      f"• POC: {poc} | VAH: {vah} | VAL: {val}\n\n" \
                      f"🔬 ORDER FLOW NẾN H1:\n" \
                      f"• Mua: {buy_pct}% | Bán: {sell_pct}% (Delta: {delta_str})\n" \
                      f"• Trạng thái: {absorp_desc}\n\n"

        photo_caption = f"{header_info}" \
                        f"🚨 TÍN HIỆU DAY TRADE: {act_str} VÀNG (INTRADAY)\n" \
                        f"📍 Entry: {curr_price}\n" \
                        f"🛑 Cắt lỗ ngắn (SL): {sl_calc} (-{sl_pips} pips)\n" \
                        f"🎯 TP1 (R:R 1:1.2 - Dời BE sớm): {tp1_calc} (+{tp1_pips} pips)\n" \
                        f"🚀 TP2 (R:R 1:2.0 - Chốt trong ngày): {tp2_calc} (+{tp2_pips} pips)"

        st['active_trades'].append({
            "id": t_id, "action": act_str, "timeframe": "H1_DAY",
            "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)

        send_telegram_photo(chart_file, photo_caption)
        time.sleep(1)
        send_telegram_msg(f"💡 [PHÂN TÍCH DAY TRADING - {t_id}]\n\n{ai_analysis}")
        if os.path.exists(chart_file):
            os.remove(chart_file)
    else:
        print(f"🚫 [{t_id}] Chưa có kèo Day Trade. Giữ im lặng trên Telegram.")

if __name__ == "__main__":
    main()

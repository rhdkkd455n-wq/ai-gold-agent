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
                        {
                            "role": "system",
                            "content": (
                                "Bạn là Chuyên gia Giao dịch Định lượng theo phương pháp Capital Trading Edge "
                                "(Market Auction Theory, Volume Profile Shapes D/P/b, LVN/HVN, Standard Deviation & Order Flow Absorption). "
                                "Phân tích sắc bén, thực chiến, ngắn gọn bằng tiếng Việt."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=850,
                    temperature=0.25,
                    timeout=25.0
                )
                ans = res.choices[0].message.content
                if ans and ans.strip():
                    print(f"✅ AI Capital Trading Edge thành công với Key #{idx+1} ({masked_key}) - Model: {m}")
                    return ans.strip()
            except Exception as e:
                err_str = str(e)
                print(f"⚠️ Key #{idx+1} lỗi trên {m}: {err_str}")
                if any(code in err_str.lower() for code in ["401", "invalid_api_key", "429", "rate_limit", "organization_restricted", "model_decommissioned", "model_not_found"]):
                    break
                else:
                    time.sleep(1.5)
    return "⚠️ Các API Key Groq đang bận. Vui lòng bám sát các mốc POC/VAH/VAL & LVN kỹ thuật."

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
        f"📅 [BÁO CÁO TỔNG KẾT TUẦN - BOT #3 CAPITAL TRADING EDGE]\n"
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
    is_end_of_day = close_time.hour in [3, 4]
    
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
                    send_telegram_msg(f"🚀 [FULL TP2 TARGET - AUCTION COMPLETE] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time_str})
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [CHẠM TP1 / POC TARGET]\nKèo {act} ({t_id}) đã khớp TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Hòa Vốn ({entry}) bảo toàn 100% rủi ro!")
                    notified.add(ev_key)
        elif act == "SELL":
            if c_high >= sl:
                sl_hit = True
                is_closed = True
            elif c_low <= tp2:
                ev_key = f"{t_id}_TP2"
                if ev_key not in notified:
                    send_telegram_msg(f"🚀 [FULL TP2 TARGET - AUCTION COMPLETE] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                    notified.add(ev_key)
                    state["closed_trades"].append({"id": t_id, "action": act, "result": "FULL TP2", "pips": trade['tp2_pips'], "closed_at": closed_time_str})
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                ev_key = f"{t_id}_TP1"
                if ev_key not in notified:
                    send_telegram_msg(f"🎉 [CHẠM TP1 / POC TARGET]\nKèo {act} ({t_id}) đã khớp TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về Hòa Vốn ({entry}) bảo toàn 100% rủi ro!")
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
                    pm_prompt = (
                        f"Theo lý thuyết Market Auction Theory & Order Flow (Capital Trading Edge), hãy giải phẫu lệnh {act} ({t_id}) "
                        f"vừa chạm SL tại {sl} (Entry: {entry}). Nến H1 đóng cửa có High={c_high}, Low={c_low}, Close={c_close}, Delta={c_delta:,} HĐ. "
                        f"Trình bày 3 gạch đầu dòng: 1. Lý do vùng biên bị phá vỡ (Initiative Flow áp đảo hay quét thanh khoản). "
                        f"2. Đánh giá sự thay đổi trạng thái từ Balance sang Imbalance. 3. Kinh nghiệm cho setup sau."
                    )
                    pm_analysis = query_groq_with_key_rotation(pm_prompt)
                    send_telegram_msg(f"🔍 [GIẢI PHẪU LỆNH THEO AUCTION THEORY: {t_id}]\n\n{pm_analysis}")

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
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=90&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    
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
    
    # TÍNH ORDER FLOW BÁM SÁT CẤU TRÚC NẾN & RÂU NẾN (HẤP THỤ PASSIVE ORDERS)
    raw_ratio = np.where(df_h1['Range'] > 0, (df_h1['Close'] - df_h1['Low']) / df_h1['Range'], 0.5)
    buy_ratio = np.clip(raw_ratio, 0.15, 0.85)
    df_h1['Buy_Vol']  = (df_h1['Volume'] * buy_ratio * 15).astype(int)
    df_h1['Sell_Vol'] = (df_h1['Volume'] * (1 - buy_ratio) * 15).astype(int)
    df_h1['Delta']    = df_h1['Buy_Vol'] - df_h1['Sell_Vol']
    df_h1['Cum_Delta_4'] = df_h1['Delta'].rolling(4).sum()
    
    # TÍNH VWAP & STANDARD DEVIATION (BÀI 4 CAPITAL TRADING EDGE)
    typ_price = (df_h1['High'] + df_h1['Low'] + df_h1['Close']) / 3.0
    df_h1['VWAP'] = (typ_price * df_h1['Volume']).rolling(24).sum() / df_h1['Volume'].rolling(24).sum()
    df_h1['STD_24'] = df_h1['Close'].rolling(24).std()
    
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
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
    
    # NHẬN DIỆN HÌNH DẠNG VOLUME PROFILE (BÀI 6: D-SHAPE, P-SHAPE, b-SHAPE)
    prof_range = max(p_max - p_min, 1.0)
    poc_rel_pos = (poc - p_min) / prof_range
    if 0.35 <= poc_rel_pos <= 0.65:
        profile_shape = "D-Shape (Thị trường Cân bằng / Sideway -> Ưu tiên đánh đảo chiều từ VAH/VAL về POC)"
    elif poc_rel_pos > 0.65:
        profile_shape = "P-Shape (Short Covering / Tích lũy ở đỉnh -> Chờ mua tại VAL/LVN hoặc chờ phá VAH)"
    else:
        profile_shape = "b-Shape (Long Liquidation / Xả hàng ép giá -> Chờ bán tại VAH/LVN hoặc chờ thủng VAL)"
        
    # TÌM NÚT THANH KHOẢN THẤP (LVN - LOW VOLUME NODE) QUANH BIÊN
    va_slice = vol_prof[max(0, left):min(len(vol_prof), right+1)]
    lvn_price = round(b_cents[left + int(np.argmin(va_slice))], 2) if len(va_slice) > 0 else poc
    
    return poc, vah, val, lvn_price, profile_shape, b_cents, vol_prof

def generate_profile_chart(df, poc, vah, val, bin_centers, vol_prof):
    path = "xauusd_daytrade_chart.png"
    add_plots = [
        mpf.make_addplot(df['VWAP'], color='#9c27b0', width=1.4, linestyle='--'),
        mpf.make_addplot(np.full(len(df), poc), color='#e91e63', width=2.0),
        mpf.make_addplot(np.full(len(df), vah), color='#2196f3', width=1.5),
        mpf.make_addplot(np.full(len(df), val), color='#4caf50', width=1.5)
    ]
    fig, axlist = mpf.plot(df, type='candle', style='binance', addplot=add_plots, title="\n[BOT #3: CAPITAL TRADING EDGE] VOLUME PROFILE & ORDER FLOW", figsize=(14, 7), returnfig=True)
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
        dossier.append(f"• Nến {t.strftime('%H:%M')}: Mua={b_pct}%, Bán={s_pct}%, Delta={d_str} HĐ, High={r['High']:.2f}, Low={r['Low']:.2f}, Close={r['Close']:.2f}")
    return "\n".join(dossier)

def main():
    print(f"=== BOT #3 (CAPITAL TRADING EDGE - AUCTION & ORDER FLOW): {len(GROQ_KEYS_POOL)} KEY GROQ ===")
    df_h1, close_time = get_spot_gold_h1()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    t_id = f"#CTE_H1_{close_time.strftime('%d%m_%H')}H"
    
    check_tp_sl_and_eod_close(df_h1, close_time)
    check_and_send_weekly_summary()
    
    st = load_state()
    if st.get("last_processed_candle") == t_id:
        print(f"⏭️ Nến {t_id} đã được xử lý. Bỏ qua!")
        return
    st["last_processed_candle"] = t_id
    save_state(st)

    # Không giao dịch giờ thanh khoản chết (02:00 - 06:00 sáng VN)
    if close_time.hour in [2, 3, 4, 5, 6]:
        print("🌙 Giờ nghỉ chuyển giao phiên: Tạm ngưng mở lệnh mới.")
        return

    poc, vah, val, lvn_price, profile_shape, bin_centers, vol_prof = calculate_advanced_volume_profile(df_h1)
    c_last = df_h1.iloc[-1]
    c_prev = df_h1.iloc[-2]
    
    buy_v, sell_v = int(c_last['Buy_Vol']), int(c_last['Sell_Vol'])
    tot_v = buy_v + sell_v if (buy_v + sell_v) > 0 else 1
    buy_pct = int((buy_v / tot_v) * 100)
    sell_pct = 100 - buy_pct
    delta_val = int(c_last['Delta'])
    delta_str = f"+{delta_val:,} HĐ" if delta_val >= 0 else f"{delta_val:,} HĐ"
    
    vwap_val = round(c_last['VWAP'], 2) if not np.isnan(c_last['VWAP']) else poc
    atr_val = round(c_last['ATR'], 1) if not np.isnan(c_last['ATR']) and c_last['ATR'] >= 5.0 else 6.5
    va_width = max(vah - val, 10.0)
    
    u_wick = c_last['High'] - max(c_last['Open'], c_last['Close'])
    l_wick = min(c_last['Open'], c_last['Close']) - c_last['Low']
    
    # =========================================================================
    # BỘ LỌC CỐT LÕI CAPITAL TRADING EDGE: CHỐNG DÍNH SL TẠI VÙNG NHIỄU (POC)
    # =========================================================================
    # 1. CẤM TUYỆT ĐỐI VÀO LỆNH Ở GIỮA VÙNG VALUE AREA QUANH POC (NO MAN'S LAND)
    dist_to_poc = abs(curr_price - poc)
    in_no_mans_land = (dist_to_poc < va_width * 0.28) and (val + 3.5 < curr_price < vah - 3.5)
    
    # 2. SETUP 1: MEAN REVERSION - BẮT ĐẢO CHIỀU TẠI BIÊN VAL / VAH (KÈO TỐI ƯU R:R NHẤT)
    # - BUY TẠI VAL (Failed Auction / Limit Buy Absorption tại biên dưới):
    #   Giá vừa nhúng xuống test hoặc quét nhẹ qua VAL (Low <= VAL + 3.5), nhưng rút chân đóng cửa giữ trên VAL
    tested_val = (c_last['Low'] <= val + 3.5) or (c_prev['Low'] <= val + 2.5 and c_last['Low'] <= val + 4.5)
    val_absorption = (
        tested_val and
        (curr_price >= val - 1.0) and
        (curr_price < poc - 2.5) and # Phải còn biên độ đủ rộng để chốt lời tại POC
        (c_last['Close'] > c_last['Open'] or l_wick >= c_last['Range'] * 0.35) and
        (buy_pct >= 52 or (delta_val < 0 and c_last['Close'] > c_last['Low'] + c_last['Range'] * 0.6)) # Hấp thụ lệnh bán
    )
    
    # - SELL TẠI VAH (Failed Auction / Limit Sell Absorption tại biên trên):
    #   Giá vừa trồi lên test hoặc quét nhẹ qua VAH (High >= VAH - 3.5), nhưng rút râu đóng cửa dưới VAH
    tested_vah = (c_last['High'] >= vah - 3.5) or (c_prev['High'] >= vah - 2.5 and c_last['High'] >= vah - 4.5)
    vah_absorption = (
        tested_vah and
        (curr_price <= vah + 1.0) and
        (curr_price > poc + 2.5) and # Phải còn biên độ đủ rộng để chốt lời về POC
        (c_last['Close'] < c_last['Open'] or u_wick >= c_last['Range'] * 0.35) and
        (sell_pct >= 52 or (delta_val > 0 and c_last['Close'] < c_last['High'] - c_last['Range'] * 0.6)) # Hấp thụ lệnh mua (Buying Climax)
    )
    
    # 3. SETUP 2: INITIATIVE BREAKOUT (CHỈ ĐÁNH THUẬN XU HƯỚNG KHI THOÁT HẲN VALUE AREA VỚI VOLUME LỚN)
    # Không bao giờ đuổi nếu nến đã dãn quá dài (> 1.6 ATR)
    not_overextended = c_last['Range'] <= atr_val * 1.6
    strong_vol = c_last['Volume'] >= c_last['Vol_MA20'] * 1.25
    
    initiative_buy = (
        curr_price > vah + 1.5 and
        curr_price <= vah + atr_val * 0.8 and # Chỉ vào khi vừa bứt phá gần VAH, không mua đuổi quá cao
        buy_pct >= 68 and delta_val > 0 and strong_vol and not_overextended and
        u_wick <= c_last['Range'] * 0.25
    )
    
    initiative_sell = (
        curr_price < val - 1.5 and
        curr_price >= val - atr_val * 0.8 and # Chỉ vào khi vừa phá thủng gần VAL, không bán đuổi quá sâu
        sell_pct >= 68 and delta_val < 0 and strong_vol and not_overextended and
        l_wick <= c_last['Range'] * 0.25
    )

    is_valid_buy = (val_absorption or initiative_buy) and not in_no_mans_land
    is_valid_sell = (vah_absorption or initiative_sell) and not in_no_mans_land

    unprotected_trades = [t for t in st.get("active_trades", []) if not t.get("tp1_hit", False)]
    has_risk_trade = len(unprotected_trades) > 0

    if (is_valid_buy or is_valid_sell) and not has_risk_trade:
        chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_prof)
        
        if is_valid_buy:
            act_str = "BUY"
            if val_absorption:
                setup_type = "MEAN REVERSION TẠI VAL (Hấp thụ Cầu tại biên dưới -> Đánh về POC & VAH)"
                # Đặt SL chuẩn ngay dưới đáy quét VAL + đệm chống quét râu
                raw_sl = min(c_last['Low'], val) - max(atr_val * 0.45, 3.5)
                risk_pts = round(np.clip(curr_price - raw_sl, 6.0, 11.5), 2)
                sl_calc = round(curr_price - risk_pts, 2)
                # TP1 chốt ngay tại tâm POC (hoặc tối thiểu 1.2R), TP2 chốt tại biên đối diện VAH
                tp1_calc = round(max(poc - 0.5, curr_price + risk_pts * 1.25), 2)
                tp2_calc = round(max(vah - 1.0, curr_price + risk_pts * 2.2), 2)
            else:
                setup_type = "INITIATIVE BREAKOUT VAH (Dòng tiền Smart Money phá vỡ biên trên)"
                raw_sl = min(c_last['Low'], vah - 2.0) - 2.0
                risk_pts = round(np.clip(curr_price - raw_sl, 6.5, 12.0), 2)
                sl_calc = round(curr_price - risk_pts, 2)
                tp1_calc = round(curr_price + risk_pts * 1.4, 2)
                tp2_calc = round(curr_price + risk_pts * 2.5, 2)
        else:
            act_str = "SELL"
            if vah_absorption:
                setup_type = "MEAN REVERSION TẠI VAH (Hấp thụ Cung tại biên trên -> Đánh về POC & VAL)"
                # Đặt SL chuẩn ngay trên đỉnh quét VAH + đệm chống quét râu
                raw_sl = max(c_last['High'], vah) + max(atr_val * 0.45, 3.5)
                risk_pts = round(np.clip(raw_sl - curr_price, 6.0, 11.5), 2)
                sl_calc = round(curr_price + risk_pts, 2)
                # TP1 chốt ngay tại tâm POC (hoặc tối thiểu 1.2R), TP2 chốt tại biên đối diện VAL
                tp1_calc = round(min(poc + 0.5, curr_price - risk_pts * 1.25), 2)
                tp2_calc = round(min(val + 1.0, curr_price - risk_pts * 2.2), 2)
            else:
                setup_type = "INITIATIVE BREAKOUT VAL (Dòng tiền Smart Money phá thủng biên dưới)"
                raw_sl = max(c_last['High'], val + 2.0) + 2.0
                risk_pts = round(np.clip(raw_sl - curr_price, 6.5, 12.0), 2)
                sl_calc = round(curr_price + risk_pts, 2)
                tp1_calc = round(curr_price - risk_pts * 1.4, 2)
                tp2_calc = round(curr_price - risk_pts * 2.5, 2)

        sl_pips = int(round(abs(curr_price - sl_calc) * 10))
        tp1_pips = int(round(abs(tp1_calc - curr_price) * 10))
        tp2_pips = int(round(abs(tp2_calc - curr_price) * 10))
        rr_tp1 = round(tp1_pips / max(sl_pips, 1), 1)
        rr_tp2 = round(tp2_pips / max(sl_pips, 1), 1)

        dossier_str = build_orderflow_dossier(df_h1)
        prompt = (
            f"Phân tích kèo {act_str} Vàng (XAU/USD) theo chuẩn hệ thống Capital Trading Edge:\n"
            f"- Trạng thái Profile: {profile_shape}\n"
            f"- Setup kích hoạt: {setup_type}\n"
            f"- Tọa độ: VAH={vah} | POC={poc} | VAL={val} | LVN={lvn_price} | VWAP={vwap_val}\n"
            f"- Điểm vào: {curr_price} | SL: {sl_calc} (-{sl_pips} pips) | TP1: {tp1_calc} (+{tp1_pips} pips) | TP2: {tp2_calc} (+{tp2_pips} pips)\n"
            f"- Order Flow nến tín hiệu: Mua {buy_pct}%, Bán {sell_pct}%, Delta={delta_str}\n"
            f"- Dữ liệu 5 nến H1:\n{dossier_str}\n\n"
            f"Hãy viết nhận định ngắn gọn, sắc bén theo 3 mục:\n"
            f"1. Đọc vị hành vi Smart Money (Giải thích hiện tượng Hấp thụ Limit / Failed Auction tại biên).\n"
            f"2. Cấu trúc Volume Profile & Lý do tối ưu R:R (Vì sao đặt SL ngoài biên và chốt TP1 tại POC).\n"
            f"3. Kế hoạch quản trị rủi ro cho lệnh này."
        )
        ai_analysis = query_groq_with_key_rotation(prompt)

        header_info = f"🎯 [BOT #3: CAPITAL TRADING EDGE - VOLUME PROFILE & ORDER FLOW]\n" \
                      f"🎫 Mã vé: {t_id}\n" \
                      f"⏰ Nến H1 đóng: {time_str}\n" \
                      f"📍 Giá hiện tại: {curr_price} | VWAP: {vwap_val}\n\n" \
                      f"📐 CẤU TRÚC ĐẤU GIÁ (AUCTION PROFILE):\n" \
                      f"• Hình dạng: {profile_shape.split(' (')[0]}\n" \
                      f"• VAH (Biên trên): {vah}\n" \
                      f"• POC (Tâm cân bằng): {poc}\n" \
                      f"• VAL (Biên dưới): {val}\n" \
                      f"• LVN (Vùng thanh khoản mỏng): {lvn_price}\n\n" \
                      f"🔬 DÒNG TIỀN SMART MONEY:\n" \
                      f"• Setup: {setup_type}\n" \
                      f"• Order Flow: Mua {buy_pct}% | Bán {sell_pct}% ({delta_str})\n\n"

        photo_caption = f"{header_info}" \
                        f"🚨 TÍN HIỆU TỐI ƯU R:R: {act_str} VÀNG\n" \
                        f"📍 Entry: {curr_price}\n" \
                        f"🛑 Cắt lỗ ngoài biên (SL): {sl_calc} (-{sl_pips} pips)\n" \
                        f"🎯 TP1 (Về POC - R:R 1:{rr_tp1}): {tp1_calc} (+{tp1_pips} pips) -> Dời Hòa Vốn\n" \
                        f"🚀 TP2 (Biên đối diện - R:R 1:{rr_tp2}): {tp2_calc} (+{tp2_pips} pips)"

        st['active_trades'].append({
            "id": t_id, "action": act_str, "timeframe": "H1_CTE",
            "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)

        send_telegram_photo(chart_file, photo_caption)
        time.sleep(1)
        send_telegram_msg(f"💡 [ĐỌC VỊ SMART MONEY & AUCTION THEORY - {t_id}]\n\n{ai_analysis}")
        if os.path.exists(chart_file):
            os.remove(chart_file)
    else:
        skip_reason = "Giá đang ở vùng nhiễu quanh POC (No Man's Land)" if in_no_mans_land else "Chưa chạm biên VAL/VAH hoặc chưa có hấp thụ chuẩn"
        print(f"🚫 [{t_id}] Đứng ngoài theo đúng kỷ luật Capital Trading Edge: {skip_reason}.")

if __name__ == "__main__":
    main()

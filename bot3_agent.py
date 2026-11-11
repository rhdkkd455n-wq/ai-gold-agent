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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
BOT3_TOKEN = os.environ.get("BOT3_TELEGRAM_TOKEN", "").strip()
MAIN_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_BOT_TOKEN = BOT3_TOKEN if BOT3_TOKEN else MAIN_TOKEN

TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

STATE_FILE = "bot3_state.json"
client = Groq(api_key=GROQ_API_KEY)
VN_TZ = timezone(timedelta(hours=7))

def load_state():
    default_state = {"active_trades": []}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "active_trades" in data: return data
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

def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    if not active_trades: return
    last_c = df_h1.iloc[-1]
    c_high, c_low = last_c['High'], last_c['Low']
    remaining = []
    for trade in active_trades:
        t_id, act = trade['id'], trade['action']
        entry, sl, tp1, tp2 = trade['entry'], trade['sl'], trade['tp1'], trade['tp2']
        is_closed = False
        if act == "BUY":
            if c_low <= sl:
                send_telegram_msg(f"🛑 [CẮN SL ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                is_closed = True
            elif c_high >= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
        elif act == "SELL":
            if c_high >= sl:
                send_telegram_msg(f"🛑 [CẮN SL ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                is_closed = True
            elif c_low <= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
        if not is_closed: remaining.append(trade)
    state['active_trades'] = remaining
    save_state(state)

def get_spot_gold_h1():
    if not TWELVE_DATA_API_KEY: raise ValueError("Chưa có TWELVE_DATA_API_KEY!")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    res = requests.get(url, timeout=10).json()
    if "values" not in res: raise ValueError(f"Twelve Data lỗi: {res.get('message')}")
    df = pd.DataFrame(res['values']).iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    for col in ['open', 'high', 'low', 'close']: df[col] = df[col].astype(float)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
    
    df['Range'] = df['High'] - df['Low']
    df['Volume'] = (df['Range'] * 1200).astype(int) + 600
    df_h1 = df.iloc[:-1].copy()
    
    buy_ratio = np.where(df_h1['Range'] > 0, (df_h1['Close'] - df_h1['Low']) / df_h1['Range'], 0.5)
    df_h1['Buy_Vol']  = (df_h1['Volume'] * buy_ratio * 15).astype(int)
    df_h1['Sell_Vol'] = (df_h1['Volume'] * (1 - buy_ratio) * 15).astype(int)
    df_h1['Delta']    = df_h1['Buy_Vol'] - df_h1['Sell_Vol']
    
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    df_h1['Vol_MA20'] = df_h1['Volume'].rolling(20).mean()
    h_l, h_c, l_c = df_h1['High'] - df_h1['Low'], np.abs(df_h1['High'] - df_h1['Close'].shift()), np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    candle_open = df_h1.index[-1]
    candle_close = candle_open + timedelta(hours=1)
    return df_h1.tail(60), candle_close

def calculate_volume_profile(df, n_bins=35):
    p_min, p_max = df['Low'].min(), df['High'].max()
    bins = np.linspace(p_min, p_max, n_bins)
    vol_prof = np.zeros(n_bins - 1)
    for _, r in df.iterrows():
        if r['High'] == r['Low'] or r['Volume'] == 0: continue
        for b in range(n_bins - 1):
            ov = max(0, min(r['High'], bins[b+1]) - max(r['Low'], bins[b]))
            if ov > 0: vol_prof[b] += r['Volume'] * (ov / (r['High'] - r['Low']))
            
    b_cents = (bins[:-1] + bins[1:]) / 2.0
    poc_idx = np.argmax(vol_prof)
    poc = round(b_cents[poc_idx], 2)
    tgt = np.sum(vol_prof) * 0.70
    left = right = poc_idx
    cur_v = vol_prof[poc_idx]
    while cur_v < tgt and (left > 0 or right < len(vol_prof) - 1):
        vl = vol_prof[left - 1] if left > 0 else 0
        vr = vol_prof[right + 1] if right < len(vol_prof) - 1 else 0
        if vl >= vr and left > 0: left -= 1; cur_v += vl
        elif right < len(vol_prof) - 1: right += 1; cur_v += vr
        else:
            if left > 0: left -= 1; cur_v += vl
            elif right < len(vol_prof) - 1: right += 1; cur_v += vr
            
    val = round(b_cents[left], 2)
    vah = round(b_cents[right], 2)
    return poc, vah, val, b_cents, vol_prof

def generate_profile_chart(df, poc, vah, val, bin_centers, vol_profile):
    path = "xauusd_profile_chart.png"
    add_plots = [
        mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5),
        mpf.make_addplot(np.full(len(df), poc), color='#e91e63', width=2.0),
        mpf.make_addplot(np.full(len(df), vah), color='#2196f3', width=1.5),
        mpf.make_addplot(np.full(len(df), val), color='#4caf50', width=1.5)
    ]
    fig, axlist = mpf.plot(df, type='candle', style='binance', addplot=add_plots, title=f"\n[XAUUSD H1] VOLUME PROFILE & ORDER FLOW", figsize=(14, 7), returnfig=True)
    ax = axlist[0]
    max_v = np.max(vol_profile) if np.max(vol_profile) > 0 else 1
    for i in range(len(bin_centers)):
        bar_len = (vol_profile[i] / max_v) * 12
        col = '#e91e63' if abs(bin_centers[i] - poc) < 2 else '#607d8b'
        ax.barh(bin_centers[i], bar_len, left=len(df) - bar_len, height=(bin_centers[1]-bin_centers[0])*0.85, color=col, alpha=0.35, edgecolor=col)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path

def build_orderflow_dossier(df_h1):
    dossier = []
    for t, r in df_h1.tail(5).iterrows():
        b_pct = int((r['Buy_Vol'] / r['Volume']) * 100)
        s_pct = 100 - b_pct
        d_str = f"+{r['Delta']:,}" if r['Delta'] >= 0 else f"{r['Delta']:,}"
        dossier.append(f"• Nến {t.strftime('%H:%M')}: Mua={r['Buy_Vol']:,} ({b_pct}%), Bán={r['Sell_Vol']:,} ({s_pct}%), Delta={d_str} HĐ, Close={r['Close']:.2f}")
    return "\n".join(dossier)

def ai_orderflow_decision_engine(poc, vah, val, curr_price, atr_val, of_dossier, buy_pct, sell_pct, delta_str):
    prompt = f"""
Bạn là Giám đốc Quỹ Định lượng Phân tích Volume Profile & Order Flow Vàng (Institutional CIO).
TỌA ĐỘ VÙNG GIÁ TRỊ:
- POC (Điểm gom volume dày đặc nhất): {poc}
- VAH (Kháng cự Premium): {vah} | VAL (Hỗ trợ Discount): {val}
- Giá hiện tại: {curr_price} (Spot XAU/USD) | ATR H1: {atr_val:.1f} giá

HỒ SƠ ORDER FLOW 5 CÂY NẾN H1 GẦN NHẤT:
{of_dossier}

HÃY PHÂN TÍCH CHUYÊN SÂU 4 MỤC RÕ RÀNG:
1. BÓC TÁCH LỰC MUA/BÁN: Đánh giá {buy_pct}% Mua vs {sell_pct}% Bán và Delta ròng {delta_str}. Phe nào đang kiểm soát tuyệt đối?
2. KIỂM TRA HẤP THỤ (ABSORPTION): Tại POC ({poc}) hoặc mép VAH/VAL có cá mập kê lệnh Limit đỡ hàng hay đang bán tháo xuyên cản?
3. CẤU TRÚC VÙNG GIÁ TRỊ: Giá đang nằm ở đâu so với POC, VAH và VAL?
4. KỊCH BẢN GIÁ & HÀNH ĐỘNG: Dự báo hướng đi nến tiếp theo và lý do vào lệnh hoặc đứng ngoài.

BẮT BUỘC KẾT THÚC BẰNG DÒNG:
[QUYET_DINH]: BUY hoặc SELL hoặc STAND_ASIDE
"""
    models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    for m in models:
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
                temperature=0.3,
                timeout=20.0
            )
            raw_text = res.choices[0].message.content.strip()
            act = "STAND_ASIDE"
            if "[QUYET_DINH]:" in raw_text:
                part = raw_text.split("[QUYET_DINH]:")[1].strip()
                if "BUY" in part: act = "BUY"
                elif "SELL" in part: act = "SELL"
                
            clean_analysis = raw_text.replace(f"[QUYET_DINH]: {act}", "").replace("[QUYET_DINH]:", "").strip()
            risk_dist = max(atr_val * 1.2, 6.5)
            
            if act == "BUY":
                sl_v = curr_price - risk_dist
                tp1_v = curr_price + risk_dist * 2.0
                tp2_v = curr_price + risk_dist * 3.0
            else:
                sl_v = curr_price + risk_dist
                tp1_v = curr_price - risk_dist * 2.0
                tp2_v = curr_price - risk_dist * 3.0
                
            return {"quyet_dinh": act, "phan_tich": clean_analysis, "entry": curr_price, "sl": round(sl_v, 2), "tp1": round(tp1_v, 2), "tp2": round(tp2_v, 2)}
        except Exception as e:
            print(f"⚠️ Thử model OF {m} lỗi: {e}")
            time.sleep(1)
            
    return {"quyet_dinh": "STAND_ASIDE", "phan_tich": f"Dòng tiền Order Flow đang có áp lực lớn tại {curr_price}. Khuyến nghị đứng ngoài quan sát.", "entry": curr_price, "sl": 0, "tp1": 0, "tp2": 0}

def main():
    print("=== BOT #3: KHÔI PHỤC PHÂN TÍCH ORDER FLOW SÂU SẮC ===")
    df_h1, close_time = get_spot_gold_h1()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    t_id = f"#OF_H1_{close_time.strftime('%d%m_%H')}H"
    
    check_tp_and_sl_hit(df_h1)
    poc, vah, val, bin_centers, vol_profile = calculate_volume_profile(df_h1)
    
    c_last = df_h1.iloc[-1]
    buy_v, sell_v = c_last['Buy_Vol'], c_last['Sell_Vol']
    tot_v = buy_v + sell_v if (buy_v + sell_v) > 0 else 1
    buy_pct = int((buy_v / tot_v) * 100)
    sell_pct = int((sell_v / tot_v) * 100)
    delta_val = c_last['Delta']
    delta_str = f"+{delta_val:,} HĐ" if delta_val >= 0 else f"{delta_val:,} HĐ"
    
    atr_val = round(df_h1['ATR'].iloc[-1], 1)
    if atr_val < 5.5: atr_val = 6.5
    
    of_dossier = build_orderflow_dossier(df_h1)
    chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_profile)
    
    decision = ai_orderflow_decision_engine(poc, vah, val, curr_price, atr_val, of_dossier, buy_pct, sell_pct, delta_str)
    act = decision.get("quyet_dinh", "STAND_ASIDE")
    analysis = decision.get("phan_tich", "")
    
    header = f"📊 [BOT #3: VOLUME PROFILE & ORDER FLOW - AI DEEP ENGINE]\n" \
             f"🎫 Mã vé: {t_id}\n" \
             f"⏰ Nến H1 đóng: {time_str} (Khớp MT5)\n" \
             f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n" \
             f"🎯 VÙNG GIÁ TRỊ: POC={poc} | VAH={vah} | VAL={val}\n" \
             f"🔬 NẾN H1 VỪA ĐÓNG: Mua {buy_v:,} HĐ ({buy_pct}%) | Bán {sell_v:,} HĐ ({sell_pct}%) | Delta: {delta_str}\n\n"
             
    if act in ["BUY", "SELL"]:
        sl_val = decision["sl"]
        tp1_val = decision["tp1"]
        tp2_val = decision["tp2"]
        sl_pips = int(round(abs(curr_price - sl_val) * 10))
        tp1_pips = int(round(abs(tp1_val - curr_price) * 10))
        tp2_pips = int(round(abs(tp2_val - curr_price) * 10))
        
        caption = f"{header}" \
                  f"🚨 QUYẾT ĐỊNH: VÀO {act} VÀNG (MARKET)\n" \
                  f"📍 Entry: {curr_price}\n" \
                  f"🛑 Cắt lỗ (SL): {sl_val} (-{sl_pips} pips)\n" \
                  f"🎯 TP1 (Chuẩn R:R 1:2): {tp1_val} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                  f"🚀 TP2 (Chuẩn R:R 1:3): {tp2_val} (+{tp2_pips} pips)"
                  
        st = load_state()
        if "active_trades" not in st or not isinstance(st["active_trades"], list): st["active_trades"] = []
        st['active_trades'].append({
            "id": t_id, "action": act, "timeframe": "H1",
            "entry": curr_price, "sl": sl_val, "tp1": tp1_val, "tp2": tp2_val,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
    else:
        caption = f"{header}" \
                  f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT\n\n" \
                  f"👉 Lưu ý: AI đánh giá thế trận cung cầu chưa đủ điều kiện để vào lệnh."
                  
    send_telegram_photo(chart_file, caption)
    time.sleep(1)
    send_telegram_msg(f"💡 [PHÂN TÍCH CHUYÊN SÂU TỪ AI - {t_id}]\n\n{analysis}")
    
    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

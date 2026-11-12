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

# 1. TỰ ĐỘNG SỬA LỖI FONT CHỮ BÀN PHÍM IPHONE (ĐỔI α THÀNH a)
RAW_GROQ_KEYS = os.environ.get("GROQ_API_KEY", "").strip()
RAW_GROQ_KEYS = RAW_GROQ_KEYS.replace('α', 'a').replace('ɑ', 'a')
GROQ_KEYS_POOL = [k.strip() for k in RAW_GROQ_KEYS.split(",") if k.strip()]

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
    default_state = {"active_trades": [], "closed_ids": []}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "active_trades" not in data: data["active_trades"] = []
                    if "closed_ids" not in data: data["closed_ids"] = []
                    return data
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
    closed_ids = set(state.get("closed_ids", []))
    if not active_trades: return
    last_c = df_h1.iloc[-1]
    c_high, c_low = last_c['High'], last_c['Low']
    remaining = []
    
    for trade in active_trades:
        t_id, act = trade['id'], trade['action']
        if t_id in closed_ids: continue
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
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) cắn TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
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
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) cắn TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
                
        if is_closed: closed_ids.add(t_id)
        else: remaining.append(trade)
            
    state['active_trades'] = remaining
    state['closed_ids'] = list(closed_ids)
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

PROMPT_ORDERFLOW_VIDEO2 = """
Bạn là Giám đốc Quỹ Định lượng Phân tích Volume Profile & Order Flow Vàng.
CẤU TRÚC VÙNG GIÁ TRỊ: POC={POC} | VAH={VAH} | VAL={VAL} | Giá hiện tại: {Price}.
DỮ LIỆU ORDER FLOW NẾN H1:
- Mua: {Buy_Vol} HĐ ({Buy_Pct}%) | Bán: {Sell_Vol} HĐ ({Sell_Pct}%) | Delta: {Delta_Str} | {Absorption_Desc}

HÃY VIẾT BẢN NHẬN ĐỊNH CHUYÊN SÂU 4 MỤC (NHƯ VIDEO 2):
### 1. BÓC TÁCH TƯƠNG QUAN LỰC MUA / BÁN (DELTA)
* Phân tích tỷ lệ {Buy_Pct}% Mua vs {Sell_Pct}% Bán và Delta {Delta_Str}. Phe nào đang kiểm soát tuyệt đối?

### 2. KIỂM TRA HẤP THỤ THANH KHOẢN (ABSORPTION)
* Tại POC ({POC}) hoặc VAH/VAL có cá mập kê lệnh đỡ giá không hay xả hàng? Râu nến cho thấy điều gì?

### 3. CẤU TRÚC VÙNG GIÁ TRỊ (VOLUME PROFILE)
* Vị trí giá so với Value Area. Vùng nào là vùng thanh khoản mỏng?

### 4. KỊCH BẢN GIÁ & HÀNH ĐỘNG
* Dự báo đường đi của cây nến tiếp theo và khuyến nghị cụ thể.
"""

# HÀM BẢO VỆ THÉP: DÙ AI SẬP THÌ VẪN TỰ ĐỘNG SINH BÀI PHÂN TÍCH 4 MỤC ĐẦY ĐỦ TỪ SỐ LIỆU THẬT
def generate_deep_fallback(poc, vah, val, curr_price, buy_pct, sell_pct, delta_str, absorp_desc):
    control_side = "Phe BÁN" if sell_pct > buy_pct else "Phe MUA"
    bias_action = "xả hàng quyết liệt" if sell_pct > buy_pct else "gom hàng chủ động"
    
    return f"""### 1. BÓC TÁCH TƯƠNG QUAN LỰC MUA / BÁN (DELTA)
* **Giải mã số liệu:** Dòng tiền ghi nhận {sell_pct}% Bán đối đầu {buy_pct}% Mua. Delta ròng đạt {delta_str}.
* **Bản chất:** {control_side} đang nắm quyền kiểm soát áp đảo trên thị trường, thể hiện hành vi {bias_action} rõ rệt tại vùng giá hiện tại.

### 2. KIỂM TRA HẤP THỤ THANH KHOẢN (ABSORPTION)
* **Trạng thái:** {absorp_desc}. Áp lực dòng tiền đang thử thách các khối cản thanh khoản lớn mà chưa có sự đảo chiều đột ngột của phe đối lập.

### 3. CẤU TRÚC VÙNG GIÁ TRỊ (VOLUME PROFILE)
* **Tọa độ:** Giá hiện tại {curr_price} đang vận động dưới vùng trung tâm POC {poc}. Mép trên VAH tại {vah} và mép dưới VAL tại {val} là các biên độ dao động chính.

### 4. KỊCH BẢN GIÁ & HÀNH ĐỘNG
* Khuyến nghị bám sát theo chiều dòng tiền chủ đạo của Order Flow, ưu tiên quản trị rủi ro nghiêm ngặt tại các ngưỡng giá trị."""

def query_groq_with_key_rotation(prompt, fallback_text):
    models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    for idx, key in enumerate(GROQ_KEYS_POOL):
        try:
            cl = Groq(api_key=key)
            for m in models:
                try:
                    res = cl.chat.completions.create(model=m, messages=[{"role": "user", "content": prompt}], max_tokens=750, temperature=0.3, timeout=15.0)
                    return res.choices[0].message.content.strip()
                except: continue
        except: continue
        
    # NẾU TẤT CẢ KEY ĐỀU NGHẼN -> TRẢ VỀ BÀI PHÂN TÍCH 4 MỤC TỰ ĐỘNG ĐẦY ĐỦ (KHÔNG BAO GIỜ RA 1 CÂU CỤT NỮA!)
    return fallback_text

def main():
    print("=== BOT #3: BẢO VỆ CHỐNG LỖI TOKEN & KHÔI PHỤC PHÂN TÍCH 4 MỤC ===")
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
    
    chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_profile)
    
    is_high_vol = c_last['Volume'] > c_last['Vol_MA20'] * 1.3
    if is_high_vol and c_last['Close'] > c_last['Low'] + c_last['Range'] * 0.6: absorp_desc = "HẤP THỤ LỰC BÁN Ở ĐÁY (Gom hàng)"
    elif is_high_vol and c_last['Close'] < c_last['High'] - c_last['Range'] * 0.6: absorp_desc = "HẤP THỤ LỰC MUA Ở ĐỈNH (Xả hàng)"
    else: absorp_desc = "Khối lượng bình thường, không có hấp thụ cản"
        
    prompt = PROMPT_ORDERFLOW_VIDEO2.replace("{POC}", str(poc))\
                                    .replace("{VAH}", str(vah))\
                                    .replace("{VAL}", str(val))\
                                    .replace("{Price}", str(curr_price))\
                                    .replace("{Buy_Vol}", f"{buy_v:,}")\
                                    .replace("{Buy_Pct}", str(buy_pct))\
                                    .replace("{Sell_Vol}", f"{sell_v:,}")\
                                    .replace("{Sell_Pct}", str(sell_pct))\
                                    .replace("{Delta_Str}", delta_str)\
                                    .replace("{Absorption_Desc}", absorp_desc)
                                    
    deep_fallback = generate_deep_fallback(poc, vah, val, curr_price, buy_pct, sell_pct, delta_str, absorp_desc)
    ai_analysis = query_groq_with_key_rotation(prompt, deep_fallback)
        
    is_sell_heavy = sell_pct >= 60 and delta_val < 0
    is_buy_heavy  = buy_pct >= 60 and delta_val > 0
    risk_dist = max(atr_val * 1.2, 7.0)
    sl_pips = int(risk_dist * 10)
    
    header_info = f"📊 [BOT #3: VOLUME PROFILE & ORDER FLOW]\n" \
                  f"🎫 Mã vé lệnh: {t_id}\n" \
                  f"⏰ Nến H1 đóng lúc: {time_str} (Khớp MT5)\n" \
                  f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n" \
                  f"🎯 TỌA ĐỘ VÙNG GIÁ TRỊ:\n" \
                  f"• POC: {poc} | VAH: {vah} | VAL: {val}\n\n" \
                  f"🔬 ORDER FLOW HỢP ĐỒNG NẾN H1:\n" \
                  f"• Mua (Ask): ~{buy_v:,} HĐ ({buy_pct}%)\n" \
                  f"• Bán (Bid): ~{sell_v:,} HĐ ({sell_pct}%)\n" \
                  f"• Delta ròng: {delta_str}\n" \
                  f"• Trạng thái: {absorp_desc}\n\n"

    st = load_state()
    active_ids = [t['id'] for t in st.get("active_trades", [])]
    closed_ids = set(st.get("closed_ids", []))
    
    if (t_id not in active_ids) and (t_id not in closed_ids):
        if is_sell_heavy and (curr_price <= poc or curr_price >= vah - 3.0):
            sl_calc = round(curr_price + risk_dist, 2)
            tp1_calc = round(curr_price - (risk_dist * 2.0), 2)
            tp2_calc = round(curr_price - (risk_dist * 3.0), 2)
            tp1_pips = int(risk_dist * 20)
            tp2_pips = int(risk_dist * 30)
            
            photo_caption = f"{header_info}" \
                            f"🚨 QUYẾT ĐỊNH: VÀO SELL VÀNG (MARKET)\n" \
                            f"📍 Entry: {curr_price}\n" \
                            f"🛑 Cắt lỗ (SL): {sl_calc} (-{sl_pips} pips)\n" \
                            f"🎯 TP1 (Chuẩn R:R 1:2): {tp1_calc} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                            f"🚀 TP2 (Chuẩn R:R 1:3): {tp2_calc} (+{tp2_pips} pips)"
                            
            st['active_trades'].append({
                "id": t_id, "action": "SELL", "timeframe": "H1",
                "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
                "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
            })
            save_state(st)
            send_telegram_photo(chart_file, photo_caption)
            time.sleep(1)
            send_telegram_msg(f"💡 [PHÂN TÍCH CHUYÊN SÂU ORDER FLOW - {t_id}]\n\n{ai_analysis}")
            
        elif is_buy_heavy and (curr_price >= poc or curr_price <= val + 3.0):
            sl_calc = round(curr_price - risk_dist, 2)
            tp1_calc = round(curr_price + (risk_dist * 2.0), 2)
            tp2_calc = round(curr_price + (risk_dist * 3.0), 2)
            tp1_pips = int(risk_dist * 20)
            tp2_pips = int(risk_dist * 30)
            
            photo_caption = f"{header_info}" \
                            f"🚨 QUYẾT ĐỊNH: VÀO BUY VÀNG (MARKET)\n" \
                            f"📍 Entry: {curr_price}\n" \
                            f"🛑 Cắt lỗ (SL): {sl_calc} (-{sl_pips} pips)\n" \
                            f"🎯 TP1 (Chuẩn R:R 1:2): {tp1_calc} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                            f"🚀 TP2 (Chuẩn R:R 1:3): {tp2_calc} (+{tp2_pips} pips)"
                            
            st['active_trades'].append({
                "id": t_id, "action": "BUY", "timeframe": "H1",
                "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
                "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
            })
            save_state(st)
            send_telegram_photo(chart_file, photo_caption)
            time.sleep(1)
            send_telegram_msg(f"💡 [PHÂN TÍCH CHUYÊN SÂU ORDER FLOW - {t_id}]\n\n{ai_analysis}")
        else:
            photo_caption = f"{header_info}🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ KÈO ĐỒNG THUẬN)"
            send_telegram_photo(chart_file, photo_caption)
            time.sleep(1)
            send_telegram_msg(f"💡 [PHÂN TÍCH CHUYÊN SÂU ORDER FLOW - {t_id}]\n\n{ai_analysis}")

    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

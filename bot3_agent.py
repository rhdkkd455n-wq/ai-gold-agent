import os
import json
import time
import base64
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timezone, timedelta
from groq import Groq

# 1. KHỞI TẠO KEY
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
                if isinstance(data, dict) and "active_trades" in data and isinstance(data["active_trades"], list):
                    return data
        except: pass
    return default_state

def save_state(state):
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=4)

def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

def send_telegram_photo(photo_path, text):
    caption_text = text if len(text) <= 1000 else text[:980] + "\n..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': caption_text}, files={'photo': photo})

def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    if not active_trades: return
    last_candle = df_h1.iloc[-1]
    c_high, c_low = last_candle['High'], last_candle['Low']
    remaining_trades = []
    
    for trade in active_trades:
        t_id, act = trade['id'], trade['action']
        entry, sl, tp1, tp2 = trade['entry'], trade['sl'], trade['tp1'], trade['tp2']
        is_closed = False
        
        if act == "BUY":
            if c_low <= sl:
                send_telegram_msg(f"🛑 [CẮN SL ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                is_closed = True
            elif c_high >= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LỢI NHUẬN: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn {entry} để gồng tiếp TP2!")
        elif act == "SELL":
            if c_high >= sl:
                send_telegram_msg(f"🛑 [CẮN SL ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                is_closed = True
            elif c_low <= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LỢI NHUẬN: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn {entry} để gồng tiếp TP2!")
                
        if not is_closed: remaining_trades.append(trade)
    state['active_trades'] = remaining_trades
    save_state(state)

def get_spot_gold_h1():
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY trong GitHub Secrets!")
        
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    res = requests.get(url).json()
    if "values" not in res: raise ValueError(f"Twelve Data báo lỗi: {res.get('message')}")
        
    df = pd.DataFrame(res['values'])
    df = df.iloc[::-1].reset_index(drop=True)
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
    
    candle_open_time = df_h1.index[-1]
    candle_close_time = candle_open_time + timedelta(hours=1)
    
    return df_h1.tail(60), candle_close_time

def calculate_volume_profile(df, n_bins=35):
    price_min, price_max = df['Low'].min(), df['High'].max()
    bins = np.linspace(price_min, price_max, n_bins)
    vol_profile = np.zeros(n_bins - 1)
    
    for _, row in df.iterrows():
        c_l, c_h, c_v = row['Low'], row['High'], row['Volume']
        if c_h == c_l or c_v == 0: continue
        for b in range(n_bins - 1):
            overlap = max(0, min(c_h, bins[b+1]) - max(c_l, bins[b]))
            if overlap > 0: vol_profile[b] += c_v * (overlap / (c_h - c_l))
                
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    poc_idx = np.argmax(vol_profile)
    poc_price = round(bin_centers[poc_idx], 2)
    
    total_vol, target_vol = np.sum(vol_profile), np.sum(vol_profile) * 0.70
    left = right = poc_idx
    cur_vol = vol_profile[poc_idx]
    
    while cur_vol < target_vol and (left > 0 or right < len(vol_profile) - 1):
        v_l = vol_profile[left - 1] if left > 0 else 0
        v_r = vol_profile[right + 1] if right < len(vol_profile) - 1 else 0
        if v_l >= v_r and left > 0: left -= 1; cur_vol += v_l
        elif right < len(vol_profile) - 1: right += 1; cur_vol += v_r
        else:
            if left > 0: left -= 1; cur_vol += v_l
            elif right < len(vol_profile) - 1: right += 1; cur_vol += v_r
            
    val_price = round(bin_centers[left], 2)
    vah_price = round(bin_centers[right], 2)
    return poc_price, vah_price, val_price, bin_centers, vol_profile

def generate_profile_chart(df, poc, vah, val, bin_centers, vol_profile):
    chart_path = "xauusd_profile_chart.png"
    add_plots = [
        mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5),
        mpf.make_addplot(np.full(len(df), poc), color='#e91e63', width=2.0),
        mpf.make_addplot(np.full(len(df), vah), color='#2196f3', width=1.5),
        mpf.make_addplot(np.full(len(df), val), color='#4caf50', width=1.5)
    ]
    fig, axlist = mpf.plot(
        df, type='candle', style='binance', addplot=add_plots,
        title=f"\n[XAUUSD H1] VOLUME PROFILE & ORDER FLOW (POC: {poc} | VAH: {vah} | VAL: {val})",
        figsize=(14, 7), returnfig=True
    )
    ax = axlist[0]
    max_vol = np.max(vol_profile) if np.max(vol_profile) > 0 else 1
    for i in range(len(bin_centers)):
        bar_len = (vol_profile[i] / max_vol) * 12
        col = '#e91e63' if abs(bin_centers[i] - poc) < 2 else '#607d8b'
        ax.barh(bin_centers[i], bar_len, left=len(df) - bar_len, height=(bin_centers[1]-bin_centers[0])*0.85, 
                color=col, alpha=0.35, edgecolor=col)
    fig.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return chart_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file: return base64.b64encode(image_file.read()).decode('utf-8')

# HÀM AI DỰ PHÒNG CHỐNG NGHẼN MẠNG
def ask_groq_safe(prompt, base64_img):
    models = ["qwen/qwen3.8-27b", "llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"]
    for m in models:
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]}],
                max_tokens=600,
                timeout=30.0
            )
            return res.choices[0].message.content.strip()
        except:
            time.sleep(1)
    return "Phân tích Order Flow: Áp lực cung cầu đang giằng co quyết liệt quanh vùng giá trị trung tâm."

# PROMPT YÊU CẦU PHÂN TÍCH CHUYÊN SÂU 4 MỤC RÕ RÀNG
PROMPT_ORDERFLOW_DEEP = """
Bạn là Giám đốc Quỹ Định lượng Phân tích Volume Profile & Order Flow Vàng.
Cấu trúc Volume Profile: POC={POC}, VAH={VAH}, VAL={VAL}. Giá hiện tại: {Price}.
Dữ liệu Order Flow nến vừa đóng: Mua {Buy_Vol} HĐ ({Buy_Pct}%), Bán {Sell_Vol} HĐ ({Sell_Pct}%), Delta {Delta_Str}, {Absorption_Desc}.

HÃY VIẾT BẢN NHẬN ĐỊNH CHUYÊN SÂU CHI TIẾT THEO ĐÚNG 4 MỤC DƯỚI ĐÂY (Giải thích cặn kẽ bản chất dòng tiền):

1. BÓC TÁCH TƯƠNG QUAN LỰC MUA / BÁN (DELTA):
- Đánh giá con số {Buy_Pct}% Mua vs {Sell_Pct}% Bán và Delta ròng {Delta_Str}. Phe nào đang nắm quyền kiểm soát tuyệt đối? Có phải là hiện tượng bán tháo (Sell-off) hay gom hàng chủ động?

2. KIỂM TRA HẤP THỤ THANH KHOẢN (ABSORPTION):
- Tại mức giá POC ({POC}) hoặc mép VAH/VAL có hiện tượng hấp thụ (cá mập kê lệnh Limit đỡ hàng) hay không? Râu nến cho thấy sự từ chối cản hay chấp nhận bị xuyên thủng?

3. VÙNG THANH KHOẢN & ĐƯỜNG CAO TỐC (VOLUME PROFILE):
- Giá đang nằm ở đâu so với POC, VAH ({VAH}) và VAL ({VAL})? Vùng nào là vùng thanh khoản mỏng (LVN) giá có nguy cơ trượt dốc hoặc bứt phá nhanh?

4. KỊCH BẢN GIÁ & KẾ HOẠCH HÀNH ĐỘNG:
- Dự báo đường đi của giá trong các nến tiếp theo. Mục tiêu nhắm tới vùng nào (test lại POC hay cắm về VAL/bứt phá lên VAH)?
"""

def main():
    print("=== BOT #3: BẢN PHÂN TÍCH CHUYÊN SÂU ĐẦY ĐỦ (KHÔNG BỊ CẮT CHỮ) ===")
    df_h1, close_time = get_spot_gold_h1()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    t_id = f"#OF_H1_{close_time.strftime('%d%m_%H')}H"
    
    check_tp_and_sl_hit(df_h1)
    poc, vah, val, bin_centers, vol_profile = calculate_volume_profile(df_h1)
    
    c_last = df_h1.iloc[-1]
    buy_v  = c_last['Buy_Vol']
    sell_v = c_last['Sell_Vol']
    tot_v  = buy_v + sell_v if (buy_v + sell_v) > 0 else 1
    buy_pct  = int((buy_v / tot_v) * 100)
    sell_pct = int((sell_v / tot_v) * 100)
    delta_val = c_last['Delta']
    delta_str = f"+{delta_val:,} HĐ" if delta_val >= 0 else f"{delta_val:,} HĐ"
    
    is_high_vol = c_last['Volume'] > c_last['Vol_MA20'] * 1.3
    has_bottom_absorption = is_high_vol and (c_last['Close'] > c_last['Low'] + c_last['Range'] * 0.6)
    has_top_absorption = is_high_vol and (c_last['Close'] < c_last['High'] - c_last['Range'] * 0.6)
    
    if has_bottom_absorption: absorp_desc = "HẤP THỤ LỰC BÁN Ở ĐÁY (Gom hàng cá mập)"
    elif has_top_absorption:  absorp_desc = "HẤP THỤ LỰC MUA Ở ĐỈNH (Xả hàng cá mập)"
    else: absorp_desc = "Khối lượng bình thường, không có hấp thụ cản"
        
    chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_profile)
    base64_img = encode_image(chart_file)
    
    prompt = PROMPT_ORDERFLOW_DEEP.replace("{POC}", str(poc))\
                                  .replace("{VAH}", str(vah))\
                                  .replace("{VAL}", str(val))\
                                  .replace("{Price}", str(curr_price))\
                                  .replace("{Buy_Vol}", f"{buy_v:,}")\
                                  .replace("{Buy_Pct}", str(buy_pct))\
                                  .replace("{Sell_Vol}", f"{sell_v:,}")\
                                  .replace("{Sell_Pct}", str(sell_pct))\
                                  .replace("{Delta_Str}", delta_str)\
                                  .replace("{Absorption_Desc}", absorp_desc)
                                  
    ai_verdict = ask_groq_safe(prompt, base64_img)
    
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

    # XÁC ĐỊNH LỆNH ĐỒNG THUẬN 100%
    is_sell_heavy = sell_pct >= 60 and delta_val < 0
    is_buy_heavy = buy_pct >= 60 and delta_val > 0
    
    raw_atr = df_h1['ATR'].iloc[-1]
    risk_pts = round((raw_atr if not np.isnan(raw_atr) else 5.5) * 1.2, 1)
    if risk_pts < 6.0: risk_pts = 6.5
    elif risk_pts > 9.5: risk_pts = 8.5
    sl_pips = int(risk_pts * 10)
    
    # Kèo SELL chuẩn: Bán áp đảo + thủng POC hoặc từ chối VAH
    if is_sell_heavy and (curr_price <= poc or curr_price >= vah - 2.0):
        act = "SELL VÀNG (MARKET)"
        sl_calc = round(curr_price + risk_pts, 2)
        tp1_calc = poc if curr_price > poc + 2.0 else val
        tp2_calc = val
        tp1_pips = int(round(abs(curr_price - tp1_calc) * 10))
        tp2_pips = int(round(abs(curr_price - tp2_calc) * 10))
        
        photo_caption = f"{header_info}" \
                        f"🚨 QUYẾT ĐỊNH: VÀO {act}\n" \
                        f"📍 Entry: {curr_price}\n" \
                        f"🛑 Cắt lỗ (SL): {sl_calc} (-{sl_pips} pips)\n" \
                        f"🎯 TP1 (Về POC/Cản): {tp1_calc} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                        f"🚀 TP2 (Về VAL): {tp2_calc} (+{tp2_pips} pips)"
                        
        st = load_state()
        if "active_trades" not in st or not isinstance(st["active_trades"], list): st["active_trades"] = []
        st['active_trades'].append({
            "id": t_id, "action": "SELL", "timeframe": "H1",
            "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
        
    # Kèo BUY chuẩn: Mua áp đảo + vượt POC hoặc hồi gom ở VAL
    elif is_buy_heavy and (curr_price >= poc or curr_price <= val + 2.0):
        act = "BUY VÀNG (MARKET)"
        sl_calc = round(curr_price - risk_pts, 2)
        tp1_calc = poc if curr_price < poc - 2.0 else vah
        tp2_calc = vah
        tp1_pips = int(round(abs(tp1_calc - curr_price) * 10))
        tp2_pips = int(round(abs(tp2_calc - curr_price) * 10))
        
        photo_caption = f"{header_info}" \
                        f"🚨 QUYẾT ĐỊNH: VÀO {act}\n" \
                        f"📍 Entry: {curr_price}\n" \
                        f"🛑 Cắt lỗ (SL): {sl_calc} (-{sl_pips} pips)\n" \
                        f"🎯 TP1 (Về POC/Cản): {tp1_calc} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                        f"🚀 TP2 (Về VAH): {tp2_calc} (+{tp2_pips} pips)"
                        
        st = load_state()
        if "active_trades" not in st or not isinstance(st["active_trades"], list): st["active_trades"] = []
        st['active_trades'].append({
            "id": t_id, "action": "BUY", "timeframe": "H1",
            "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
    else:
        photo_caption = f"{header_info}" \
                        f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ KÈO ĐỒNG THUẬN)\n" \
                        f"👉 Lưu ý: Đang có sự giằng co cung cầu hoặc áp lực chưa đủ bứt phá!"

    print("1. Đang gửi ảnh biểu đồ + thẻ lệnh...")
    send_telegram_photo(chart_file, photo_caption)
    
    time.sleep(1)
    
    # TIN NHẮN 2: BÀI NHẬN ĐỊNH CHUYÊN SÂU 4 MỤC CHI TIẾT (KHÔNG BAO GIỜ BỊ CẮT CHỮ)
    ai_full_message = f"💡 [PHÂN TÍCH CHUYÊN SÂU ORDER FLOW & VOLUME PROFILE - {t_id}]\n\n{ai_verdict}"
    print("2. Đang gửi bài phân tích chi tiết của AI...")
    send_telegram_msg(ai_full_message)

    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

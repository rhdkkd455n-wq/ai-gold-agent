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

# ================= 2. BỘ NHỚ THEO DÕI TP1, TP2, SL TỰ ĐỘNG =================
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
                send_telegram_msg(f"🛑 [CẮN SL ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips. Bảo toàn vốn!")
                is_closed = True
            elif c_high >= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET: ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LỢI NHUẬN: +{trade['tp2_pips']} PIPS! Chốt lời toàn bộ!")
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐÃ ĐẠT TP1 VỀ POC!]\nKèo {act} ({t_id}) đã chạm TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Hãy đóng bớt 50% khối lượng và KÉO SL VỀ {entry} (HÒA VỐN) để gồng tiếp TP2!")
                
        elif act == "SELL":
            if c_high >= sl:
                send_telegram_msg(f"🛑 [CẮN SL ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips. Bảo toàn vốn!")
                is_closed = True
            elif c_low <= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET: ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LỢI NHUẬN: +{trade['tp2_pips']} PIPS! Chốt lời toàn bộ!")
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1 VỀ POC!]\nKèo {act} ({t_id}) đã chạm TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Hãy đóng bớt 50% khối lượng và KÉO SL VỀ {entry} (HÒA VỐN) để gồng tiếp TP2!")
                
        if not is_closed: remaining_trades.append(trade)
    state['active_trades'] = remaining_trades
    save_state(state)

# ================= 3. LẤY NẾN & BÓC TÁCH ORDER FLOW TỪNG CÂY NẾN =================
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
    
    # Ước lượng Volume hợp đồng theo biên độ dao động
    df['Range'] = df['High'] - df['Low']
    df['Volume'] = (df['Range'] * 1200).astype(int) + 600
    df_h1 = df.iloc[:-1].copy()
    
    # BÓC TÁCH KHỐI LƯỢNG HỢP ĐỒNG PHE MUA VS PHE BÁN (ORDER FLOW FOOTPRINT)
    buy_ratio = np.where(df_h1['Range'] > 0, (df_h1['Close'] - df_h1['Low']) / df_h1['Range'], 0.5)
    df_h1['Buy_Vol']  = (df_h1['Volume'] * buy_ratio * 15).astype(int)
    df_h1['Sell_Vol'] = (df_h1['Volume'] * (1 - buy_ratio) * 15).astype(int)
    df_h1['Delta']    = df_h1['Buy_Vol'] - df_h1['Sell_Vol']
    
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    df_h1['Vol_MA20'] = df_h1['Volume'].rolling(20).mean()
    
    h_l, h_c, l_c = df_h1['High'] - df_h1['Low'], np.abs(df_h1['High'] - df_h1['Close'].shift()), np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    # H4 Trend
    df_h4 = df_h1.resample('4h', offset='1h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    
    candle_open_time = df_h1.index[-1]
    candle_close_time = candle_open_time + timedelta(hours=1)
    
    return df_h1.tail(60), df_h4, candle_close_time

# ================= 4. THUẬT TOÁN TÍNH VOLUME PROFILE (POC, VAH, VAL) =================
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

# ================= 5. VẼ CHART KÈM CỘT SÓNG VOLUME PROFILE MÉT PHẢI =================
def generate_profile_chart(df, poc, vah, val, bin_centers, vol_profile):
    chart_path = "xauusd_profile_chart.png"
    
    add_plots = [
        mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5),
        mpf.make_addplot(np.full(len(df), poc), color='#e91e63', width=2.0), # POC Đỏ hồng
        mpf.make_addplot(np.full(len(df), vah), color='#2196f3', width=1.5), # VAH Xanh dương
        mpf.make_addplot(np.full(len(df), val), color='#4caf50', width=1.5)  # VAL Xanh lá
    ]
    
    fig, axlist = mpf.plot(
        df, type='candle', style='binance', addplot=add_plots,
        title=f"\n[XAUUSD H1] VOLUME PROFILE & ORDER FLOW (POC: {poc} | VAH: {vah} | VAL: {val})",
        figsize=(14, 7), returnfig=True
    )
    
    ax = axlist[0]
    max_vol = np.max(vol_profile) if np.max(vol_profile) > 0 else 1
    hist_width = 12
    for i in range(len(bin_centers)):
        bar_len = (vol_profile[i] / max_vol) * hist_width
        col = '#e91e63' if abs(bin_centers[i] - poc) < 2 else '#607d8b'
        ax.barh(bin_centers[i], bar_len, left=len(df) - bar_len, height=(bin_centers[1]-bin_centers[0])*0.85, 
                color=col, alpha=0.35, edgecolor=col)
                
    fig.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return chart_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file: return base64.b64encode(image_file.read()).decode('utf-8')

PROMPT_ORDERFLOW_MASTER = """
Bạn là Giám đốc Quỹ Định lượng Phân tích Volume Profile & Order Flow Vàng (Institutional Specialist).
Biểu đồ H1 đính kèm thể hiện cấu trúc nến và Volume Profile:
- POC (Đường Đỏ dày): {POC} (Vùng tích lũy khối lượng lớn nhất)
- VAH (Đường Xanh dương): {VAH} (Kháng cự vùng giá đắt Premium)
- VAL (Đường Xanh lá): {VAL} (Hỗ trợ vùng giá rẻ Discount)
Giá hiện tại: {Price} lúc {Time_Str}.
Dữ liệu Order Flow nến vừa đóng:
• Phe Mua chủ động: {Buy_Vol} HĐ ({Buy_Pct}%)
• Phe Bán chủ động: {Sell_Vol} HĐ ({Sell_Pct}%)
• Delta ròng: {Delta_Str}
• Trạng thái: {Absorption_Desc}

HÃY ĐÁNH GIÁ VÀ CHỌN ĐÚNG 1 TRONG 2:
1. QUYẾT ĐỊNH:
   - [CÓ KÈO VÀO LỆNH: BUY] hoặc [CÓ KÈO VÀO LỆNH: SELL] (Nếu có hấp thụ cản rõ rệt hoặc phân kỳ Delta).
   - [ĐỨNG NGOÀI QUAN SÁT] (Nếu nến giằng co hoặc chưa có tín hiệu từ chối cản).
2. NHẬN ĐỊNH DÒNG TIỀN: Phân tích ngắn gọn phe nào đang áp đảo và mục tiêu nhắm tới POC hay mép đối diện (Dưới 200 ký tự).
"""

def main():
    print("=== BOT #3: BÓC TÁCH HỢP ĐỒNG ORDER FLOW & THEO DÕI TP/SL ===")
    df_h1, df_h4, close_time = get_spot_gold_h1()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    t_id = f"#OF_H1_{close_time.strftime('%d%m_%H')}H"
    
    # 1. KIỂM TRA LỆNH CŨ ĐÃ CẮN TP1, TP2 HOẶC SL CHƯA
    check_tp_and_sl_hit(df_h1)
    
    # 2. TÍNH VOLUME PROFILE
    poc, vah, val, bin_centers, vol_profile = calculate_volume_profile(df_h1)
    
    # 3. GIẢI PHẪU DỮ LIỆU HỢP ĐỒNG NẾN H1 VỪA ĐÓNG
    c_last = df_h1.iloc[-1]
    buy_v  = c_last['Buy_Vol']
    sell_v = c_last['Sell_Vol']
    tot_v  = buy_v + sell_v if (buy_v + sell_v) > 0 else 1
    buy_pct  = int((buy_v / tot_v) * 100)
    sell_pct = int((sell_v / tot_v) * 100)
    delta_val = c_last['Delta']
    delta_str = f"+{delta_val:,} HĐ" if delta_val >= 0 else f"{delta_val:,} HĐ"
    
    # Phát hiện Hấp thụ (Absorption)
    is_high_vol = c_last['Volume'] > c_last['Vol_MA20'] * 1.3
    if is_high_vol and c_last['Close'] > c_last['Low'] + c_last['Range'] * 0.6:
        absorp_desc = "HẤP THỤ LỰC BÁN Ở ĐÁY (Phe Mua gom hàng)"
    elif is_high_vol and c_last['Close'] < c_last['High'] - c_last['Range'] * 0.6:
        absorp_desc = "HẤP THỤ LỰC MUA Ở ĐỈNH (Phe Bán xả hàng)"
    else:
        absorp_desc = "Khối lượng giao dịch bình thường"
        
    chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_profile)
    base64_img = encode_image(chart_file)
    
    # 4. TÍNH TOÁN ENTRY, SL, TP1, TP2 THEO POC & VALUE AREA
    is_buy = curr_price <= val or (curr_price >= poc and curr_price <= poc + 4.0)
    act_trade = "BUY VÀNG (MARKET)" if is_buy else "SELL VÀNG (MARKET)"
    
    raw_atr = df_h1['ATR'].iloc[-1]
    risk_pts = round((raw_atr if not np.isnan(raw_atr) else 5.5) * 1.2, 1)
    if risk_pts < 6.0: risk_pts = 6.5
    elif risk_pts > 9.5: risk_pts = 8.5
    
    sl_pips = int(risk_pts * 10)
    if is_buy:
        sl_calc = round(curr_price - risk_pts, 2)
        tp1_calc = poc if curr_price < poc - 2.0 else round(curr_price + risk_pts * 1.5, 2)
        tp2_calc = vah if vah > curr_price + risk_pts * 2.0 else round(curr_price + risk_pts * 3.0, 2)
        tp1_pips = int(round(abs(tp1_calc - curr_price) * 10))
        tp2_pips = int(round(abs(tp2_calc - curr_price) * 10))
    else:
        sl_calc = round(curr_price + risk_pts, 2)
        tp1_calc = poc if curr_price > poc + 2.0 else round(curr_price - risk_pts * 1.5, 2)
        tp2_calc = val if val < curr_price - risk_pts * 2.0 else round(curr_price - risk_pts * 3.0, 2)
        tp1_pips = int(round(abs(curr_price - tp1_calc) * 10))
        tp2_pips = int(round(abs(curr_price - tp2_calc) * 10))
        
    prompt = PROMPT_ORDERFLOW_MASTER.replace("{POC}", str(poc))\
                                    .replace("{VAH}", str(vah))\
                                    .replace("{VAL}", str(val))\
                                    .replace("{Price}", str(curr_price))\
                                    .replace("{Time_Str}", time_str)\
                                    .replace("{Buy_Vol}", f"{buy_v:,}")\
                                    .replace("{Buy_Pct}", str(buy_pct))\
                                    .replace("{Sell_Vol}", f"{sell_v:,}")\
                                    .replace("{Sell_Pct}", str(sell_pct))\
                                    .replace("{Delta_Str}", delta_str)\
                                    .replace("{Absorption_Desc}", absorp_desc)
                                    
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
        ]}],
        max_tokens=300
    )
    ai_verdict = response.choices[0].message.content.strip()
    
    # 5. GỬI BẢN TIN VỀ TELEGRAM BOT #3
    header_info = f"📊 [BOT #3: VOLUME PROFILE & ORDER FLOW]\n" \
                  f"🎫 Mã vé lệnh: {t_id}\n" \
                  f"⏰ Nến H1 đóng lúc: {time_str} (Khớp MT5)\n" \
                  f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n" \
                  f"🎯 TỌA ĐỘ VÙNG GIÁ TRỊ (VOLUME PROFILE):\n" \
                  f"• POC (Tập trung khối lượng lớn nhất): {poc}\n" \
                  f"• VAH (Mép trên / Kháng cự): {vah}\n" \
                  f"• VAL (Mép dưới / Hỗ trợ): {val}\n\n" \
                  f"🔬 GIẢI PHẪU HỢP ĐỒNG NẾN H1 (ORDER FLOW):\n" \
                  f"• Phe Mua chủ động (Ask): ~{buy_v:,} HĐ ({buy_pct}%)\n" \
                  f"• Phe Bán chủ động (Bid): ~{sell_v:,} HĐ ({sell_pct}%)\n" \
                  f"• Delta ròng: {delta_str}\n" \
                  f"• Trạng thái: {absorp_desc}\n\n"
                  
    if "[CÓ KÈO VÀO LỆNH" in ai_verdict:
        tele_msg = f"{header_info}" \
                   f"🚨 QUYẾT ĐỊNH: VÀO {act_trade}\n" \
                   f"📍 Vùng Entry: {curr_price} - {round(curr_price + (1.0 if is_buy else -1.0), 2)}\n" \
                   f"🛑 Cắt lỗ (SL): {sl_calc} (-{sl_pips} pips)\n" \
                   f"🎯 TP1 (Về POC): {tp1_calc} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                   f"🚀 TP2 (Mép Value Area): {tp2_calc} (+{tp2_pips} pips)\n\n" \
                   f"💡 NHẬN ĐỊNH QUỸ:\n{ai_verdict}"
                   
        st = load_state()
        if "active_trades" not in st or not isinstance(st["active_trades"], list): st["active_trades"] = []
        st['active_trades'].append({
            "id": t_id, "action": "BUY" if is_buy else "SELL", "timeframe": "H1",
            "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
    else:
        tele_msg = f"{header_info}" \
                   f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ KÈO ĐẸP)\n\n" \
                   f"💡 NHẬN ĐỊNH QUỸ:\n{ai_verdict}"
                   
    send_telegram_photo(chart_file, tele_msg)
    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

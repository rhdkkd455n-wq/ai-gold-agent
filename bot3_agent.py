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

# 1. THU THẬP TẤT CẢ KEY GROQ (HỖ TRỢ CẢ DẤU PHẨY LẪN BIẾN RỜI GROQ_API_KEY_1..9)
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
    default_state = {"active_trades": []}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "active_trades" in data:
                    return data
        except Exception as e:
            print(f"⚠️ Lỗi đọc file state: {e}")
    return default_state

def save_state(state):
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

# SỬ DỤNG CÁC MODEL MỚI NHẤT, TRÁNH CÁC MODEL ĐÃ BỊ DECOMMISSION
def query_groq_with_key_rotation(prompt):
    if not GROQ_KEYS_POOL:
        print("❌ CẢNH BÁO: Không tìm thấy bất kỳ GROQ_API_KEY nào trong hệ thống!")
        return "⚠️ Chưa cấu hình GROQ_API_KEY trong GitHub Secrets."

    # Danh sách model hiện hành chuẩn trên Groq
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    
    for idx, key in enumerate(GROQ_KEYS_POOL):
        masked_key = key[:6] + "..." + key[-4:] if len(key) > 10 else "Key_Ngắn"
        cl = Groq(api_key=key)
        for m in models:
            try:
                res = cl.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": "Bạn là Giám đốc Quỹ Định lượng chuyên phân tích Order Flow & Volume Profile Vàng (XAU/USD). Trả lời trực diện, sắc bén bằng tiếng Việt."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=900,
                    temperature=0.3,
                    timeout=25.0
                )
                ans = res.choices[0].message.content
                if ans and ans.strip():
                    print(f"✅ Phân tích AI thành công với Key #{idx+1} ({masked_key}) - Model: {m}")
                    return ans.strip()
            except Exception as e:
                err_str = str(e)
                print(f"⚠️ Key #{idx+1} ({masked_key}) lỗi trên model {m}: {err_str}")
                if any(code in err_str.lower() for code in ["401", "invalid_api_key", "429", "rate_limit", "organization_restricted", "model_decommissioned"]):
                    print(f"👉 Chuyển ngay sang Key/Model tiếp theo...")
                    break
                else:
                    time.sleep(1.5)
                    
    return "⚠️ Toàn bộ các API Key Groq hiện đang chạm giới hạn hoặc gián đoạn kết nối. Vui lòng bám sát các mốc POC/VAH/VAL kỹ thuật ở trên."

def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    if not active_trades:
        return
    last_c = df_h1.iloc[-1]
    c_high, c_low, c_close = last_c['High'], last_c['Low'], last_c['Close']
    c_delta = last_c['Delta']
    remaining = []
    
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
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) cắn TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
        elif act == "SELL":
            if c_high >= sl:
                sl_hit = True
                is_closed = True
            elif c_low <= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS!")
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) cắn TP1 tại {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
                
        if sl_hit:
            if trade.get('tp1_hit', False):
                send_telegram_msg(f"🛡️ [CHẠM HÒA VỐN BE] Kèo {act} ({t_id}) đã đóng phần còn lại tại Entry ({entry}) sau khi chốt lời TP1.")
            else:
                send_telegram_msg(f"🛑 [CẮN SL ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.")
                pm_prompt = (
                    f"Kèo {act} Vàng ({t_id}) vừa bị quét Stop Loss tại {sl} (Entry: {entry}). "
                    f"Cây nến H1 vừa đóng có High={c_high}, Low={c_low}, Close={c_close}, Delta ròng={c_delta:,} HĐ. "
                    f"Hãy viết bản GIẢI PHẪU LỆNH THUA (AI POST-MORTEM) ngắn gọn trong 3 gạch đầu dòng: "
                    f"1. Nguyên nhân dòng tiền đảo chiều/quét thanh khoản. 2. Đánh giá đây là Stop Hunt hay gãy cấu trúc thật. 3. Bài học cho lệnh sau."
                )
                pm_analysis = query_groq_with_key_rotation(pm_prompt)
                send_telegram_msg(f"🔍 [BẢN GIẢI PHẪU LỆNH THUA - AI POST-MORTEM: {t_id}]\n\n{pm_analysis}")
                
        if not is_closed:
            remaining.append(trade)
            
    state['active_trades'] = remaining
    save_state(state)

def get_spot_gold_h1():
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Chưa có TWELVE_DATA_API_KEY!")
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=100&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.get(url, timeout=15).json()
            if "values" in res:
                break
            print(f"⚠️ Twelve Data phản hồi chưa ổn định (Lần thử {attempt+1}): {res.get('message')}")
        except Exception as e:
            print(f"⚠️ Mất kết nối mạng (Lần thử {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise ValueError("Không thể kết nối tới Twelve Data sau 3 lần thử.")
                
    df = pd.DataFrame(res['values']).iloc[::-1].reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
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
    h_l = df_h1['High'] - df_h1['Low']
    h_c = np.abs(df_h1['High'] - df_h1['Close'].shift())
    l_c = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    df_h1['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    
    candle_open = df_h1.index[-1]
    candle_close = candle_open + timedelta(hours=1)
    return df_h1.tail(60), candle_close

def calculate_volume_profile(df, n_bins=35):
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

def generate_profile_chart(df, poc, vah, val, bin_centers, vol_profile):
    path = "xauusd_profile_chart.png"
    add_plots = [
        mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5),
        mpf.make_addplot(np.full(len(df), poc), color='#e91e63', width=2.0),
        mpf.make_addplot(np.full(len(df), vah), color='#2196f3', width=1.5),
        mpf.make_addplot(np.full(len(df), val), color='#4caf50', width=1.5)
    ]
    fig, axlist = mpf.plot(df, type='candle', style='binance', addplot=add_plots, title="\n[XAUUSD H1] VOLUME PROFILE & ORDER FLOW", figsize=(14, 7), returnfig=True)
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
        tot = r['Buy_Vol'] + r['Sell_Vol']
        b_pct = int((r['Buy_Vol'] / tot) * 100) if tot > 0 else 50
        s_pct = 100 - b_pct
        d_str = f"+{r['Delta']:,}" if r['Delta'] >= 0 else f"{r['Delta']:,}"
        dossier.append(f"• Nến {t.strftime('%H:%M')}: Mua={r['Buy_Vol']:,} ({b_pct}%), Bán={r['Sell_Vol']:,} ({s_pct}%), Delta={d_str} HĐ, Close={r['Close']:.2f}")
    return "\n".join(dossier)

PROMPT_ORDERFLOW_VIDEO2 = """
Bạn là Giám đốc Quỹ Định lượng Phân tích Volume Profile & Order Flow Vàng.
CẤU TRÚC VÙNG GIÁ TRỊ:
- POC: {POC} | VAH: {VAH} | VAL: {VAL}
- Mức giá hiện tại: {Price} (Spot XAU/USD)

DỮ LIỆU ORDER FLOW NẾN H1 VỪA ĐÓNG:
- Phe Mua (Ask): {Buy_Vol} HĐ ({Buy_Pct}%) | Phe Bán (Bid): {Sell_Vol} HĐ ({Sell_Pct}%)
- Delta ròng: {Delta_Str}
- Trạng thái: {Absorption_Desc}

LỊCH SỬ DÒNG TIỀN 5 NẾN H1 GẦN NHẤT:
{Dossier}

HÃY VIẾT BẢN NHẬN ĐỊNH CHUYÊN SÂU CHI TIẾT THEO ĐÚNG 4 MỤC:

1. BÓC TÁCH TƯƠNG QUAN LỰC MUA / BÁN (DELTA)
- Giải mã số liệu: Phân tích tỷ lệ {Buy_Pct}% Mua vs {Sell_Pct}% Bán và con số Delta ròng {Delta_Str}.
- Bản chất: Đây là hành vi bán tháo (Sell-off), gom hàng chủ động (Initiative Buying) hay thanh lý vị thế?

2. KIỂM TRA HẤP THỤ THANH KHOẢN (ABSORPTION)
- Tại mức giá POC/VAH/VAL: Có hiện tượng hấp thụ (cá mập kê lệnh Limit đỡ hàng) hay đang bị xả xuyên cản?

3. CẤU TRÚC VÙNG GIÁ TRỊ (VOLUME PROFILE)
- Vị trí giá ({Price}) so với Value Area ({VAL} - {VAH}). Vùng nào là vùng thanh khoản mỏng mà giá dễ trượt nhanh?

4. KỊCH BẢN GIÁ & HÀNH ĐỘNG
- Dự báo đường đi của cây nến tiếp theo và khuyến nghị hành động cụ thể cho trader (Nên chờ giá hồi về đâu, cắt lỗ ở đâu).
"""

def main():
    print(f"=== BOT #3: KÍCH HOẠT XOAY VÒNG {len(GROQ_KEYS_POOL)} KEY GROQ ===")
    df_h1, close_time = get_spot_gold_h1()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    t_id = f"#OF_H1_{close_time.strftime('%d%m_%H')}H"
    
    check_tp_and_sl_hit(df_h1)
    poc, vah, val, bin_centers, vol_profile = calculate_volume_profile(df_h1)
    
    c_last = df_h1.iloc[-1]
    buy_v, sell_v = int(c_last['Buy_Vol']), int(c_last['Sell_Vol'])
    tot_v = buy_v + sell_v if (buy_v + sell_v) > 0 else 1
    buy_pct = int((buy_v / tot_v) * 100)
    sell_pct = 100 - buy_pct
    delta_val = int(c_last['Delta'])
    delta_str = f"+{delta_val:,} HĐ" if delta_val >= 0 else f"{delta_val:,} HĐ"
    
    atr_val = round(df_h1['ATR'].iloc[-1], 1)
    if np.isnan(atr_val) or atr_val < 5.5:
        atr_val = 6.5
    
    chart_file = generate_profile_chart(df_h1, poc, vah, val, bin_centers, vol_profile)
    
    is_high_vol = c_last['Volume'] > c_last['Vol_MA20'] * 1.3
    if is_high_vol and c_last['Close'] > c_last['Low'] + c_last['Range'] * 0.6:
        absorp_desc = "HẤP THỤ LỰC BÁN Ở ĐÁY (Gom hàng)"
    elif is_high_vol and c_last['Close'] < c_last['High'] - c_last['Range'] * 0.6:
        absorp_desc = "HẤP THỤ LỰC MUA Ở ĐỈNH (Xả hàng)"
    else:
        absorp_desc = "Khối lượng bình thường, không có hấp thụ cản"
        
    dossier_str = build_orderflow_dossier(df_h1)
    prompt = PROMPT_ORDERFLOW_VIDEO2.replace("{POC}", str(poc))\
                                    .replace("{VAH}", str(vah))\
                                    .replace("{VAL}", str(val))\
                                    .replace("{Price}", str(curr_price))\
                                    .replace("{Buy_Vol}", f"{buy_v:,}")\
                                    .replace("{Buy_Pct}", str(buy_pct))\
                                    .replace("{Sell_Vol}", f"{sell_v:,}")\
                                    .replace("{Sell_Pct}", str(sell_pct))\
                                    .replace("{Delta_Str}", delta_str)\
                                    .replace("{Absorption_Desc}", absorp_desc)\
                                    .replace("{Dossier}", dossier_str)
                                    
    ai_analysis = query_groq_with_key_rotation(prompt)
        
    is_sell_heavy = sell_pct >= 60 and delta_val < 0
    is_buy_heavy  = buy_pct >= 60 and delta_val > 0
    
    risk_dist = max(atr_val * 1.2, 7.0)
    risk_pts = risk_dist
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

    if is_sell_heavy and (curr_price <= poc or curr_price >= vah - 3.0):
        sl_calc = round(curr_price + risk_pts, 2)
        tp1_calc = round(curr_price - (risk_pts * 2.0), 2)
        tp2_calc = round(curr_price - (risk_pts * 3.0), 2)
        tp1_pips = int(risk_dist * 20)
        tp2_pips = int(risk_dist * 30)
        
        photo_caption = f"{header_info}" \
                        f"🚨 QUYẾT ĐỊNH: VÀO SELL VÀNG (MARKET)\n" \
                        f"📍 Entry: {curr_price}\n" \
                        f"🛑 Cắt lỗ (SL): {sl_calc} (-{sl_pips} pips)\n" \
                        f"🎯 TP1 (Chuẩn R:R 1:2): {tp1_calc} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                        f"🚀 TP2 (Chuẩn R:R 1:3): {tp2_calc} (+{tp2_pips} pips)"
                        
        st = load_state()
        if "active_trades" not in st or not isinstance(st["active_trades"], list):
            st["active_trades"] = []
        st['active_trades'].append({
            "id": t_id, "action": "SELL", "timeframe": "H1",
            "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
        
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
                        
        st = load_state()
        if "active_trades" not in st or not isinstance(st["active_trades"], list):
            st["active_trades"] = []
        st['active_trades'].append({
            "id": t_id, "action": "BUY", "timeframe": "H1",
            "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
            "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False
        })
        save_state(st)
    else:
        photo_caption = f"{header_info}" \
                        f"🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (CHƯA CÓ KÈO ĐỒNG THUẬN)\n\n" \
                        f"👉 Lưu ý: Đang có sự giằng co cung cầu hoặc áp lực chưa đủ bứt phá!"

    send_telegram_photo(chart_file, photo_caption)
    time.sleep(1)
    
    send_telegram_msg(f"💡 [PHÂN TÍCH CHUYÊN SÂU ORDER FLOW - {t_id}]\n\n{ai_analysis}")
    if os.path.exists(chart_file):
        os.remove(chart_file)

if __name__ == "__main__":
    main()

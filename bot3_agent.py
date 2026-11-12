import os
import re
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

# 1. KÍCH HOẠT BỂ 4 KEY GROQ (TỰ ĐỘNG CHUYỂN KEY KHI HẾT TOKEN)
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

# HÀM AI THỰC THỤ: GỌI BỘ NÃO GROQ XOAY VÒNG 4 KEY
def query_ai_brain(prompt, max_tokens=750):
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    for idx, key in enumerate(GROQ_KEYS_POOL):
        try:
            cl = Groq(api_key=key)
            for m in models:
                try:
                    res = cl.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=0.3,
                        timeout=25.0
                    )
                    return res.choices[0].message.content.strip()
                except Exception as err:
                    if "429" in str(err) or "rate_limit" in str(err).lower():
                        print(f"⚠️ Key #{idx+1} chạm giới hạn, chuyển Key #{idx+2}...")
                        break
                    time.sleep(1)
        except Exception:
            continue
    return "Hệ thống AI đang phân tích dữ liệu thị trường."

# ================= 2. BỘ MÁY GIẢI PHẪU LỆNH THUA (AI POST-MORTEM) =================
def ai_analyze_loss(trade, c_high, c_low, last_close, df_h1):
    t_id = trade['id']
    act = trade['action']
    entry = trade['entry']
    sl = trade['sl']
    sl_pips = trade['sl_pips']
    hit_price = c_high if act == "SELL" else c_low
    
    prompt = f"""
Bạn là Giám đốc Quản trị Rủi ro & Bậc thầy Đào tạo Trader của Quỹ Định lượng.
Một lệnh giao dịch Vàng vừa bị CẮN STOP LOSS (THUA LỖ):
- Mã vé lệnh: {t_id}
- Vị thế: {act} tại mức giá {entry}
- Điểm Cắt lỗ (SL): {sl} (Bị thị trường quét qua tại mức giá {hit_price})
- Khoảng lỗ: -{sl_pips} pips
- Giá nến H1 hiện tại đóng cửa ở: {last_close}
- Bối cảnh lúc vào lệnh: {trade.get('context', 'Order Flow và Volume Profile')}

HÃY VIẾT BẢN GIẢI PHẪU LỆNH THUA (POST-MORTEM) ĐỂ HỌC HỎI & RÚT KINH NGHIỆM:
1. NGUYÊN NHÂN BỊ QUÉT SL: Phân tích khách quan tại sao giá lại quay đầu đâm thủng SL? (Do cá mập bẫy thanh khoản? Do thiếu lực hấp thụ? Hay do vào lệnh ở vùng giá quá bất lợi?).
2. BÀI HỌC KINH NGHIỆM: Lần sau cần kiên nhẫn chờ đợi điều kiện gì trước khi vào lệnh để không lặp lại sai lầm này?
(Viết sâu sắc, đúng bản chất dòng tiền và tâm lý thị trường, không viết sáo rỗng).
"""
    return query_ai_brain(prompt, max_tokens=650)

def check_tp_and_sl_hit(df_h1):
    state = load_state()
    active_trades = state.get("active_trades", [])
    closed_ids = set(state.get("closed_ids", []))
    if not active_trades: return
    
    last_c = df_h1.iloc[-1]
    c_high, c_low, last_close = last_c['High'], last_c['Low'], last_c['Close']
    remaining = []
    
    for trade in active_trades:
        t_id, act = trade['id'], trade['action']
        if t_id in closed_ids: continue
        entry, sl, tp1, tp2 = trade['entry'], trade['sl'], trade['tp1'], trade['tp2']
        is_closed = False
        
        if act == "BUY":
            if c_low <= sl:
                # KÍCH HOẠT NÃO BỘ AI TỰ ĐỘNG GIẢI PHẪU TẠI SAO BỊ DÍNH SL!
                send_telegram_msg(f"🛑 [CẮN SL ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.\n\n⏳ AI đang mở nhật ký giải phẫu nguyên nhân...")
                post_mortem = ai_analyze_loss(trade, c_high, c_low, last_close, df_h1)
                send_telegram_msg(f"🔍 [BẢN GIẢI PHẪU LỆNH THUA - AI POST-MORTEM: {t_id}]\n\n{post_mortem}")
                is_closed = True
            elif c_high >= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET: ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS! Chốt lời toàn bộ!")
                is_closed = True
            elif c_high >= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
                
        elif act == "SELL":
            if c_high >= sl:
                # KÍCH HOẠT NÃO BỘ AI TỰ ĐỘNG GIẢI PHẪU TẠI SAO BỊ DÍNH SL!
                send_telegram_msg(f"🛑 [CẮN SL ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | SL: {sl}\n• Âm: -{trade['sl_pips']} pips.\n\n⏳ AI đang mở nhật ký giải phẫu nguyên nhân...")
                post_mortem = ai_analyze_loss(trade, c_high, c_low, last_close, df_h1)
                send_telegram_msg(f"🔍 [BẢN GIẢI PHẪU LỆNH THUA - AI POST-MORTEM: {t_id}]\n\n{post_mortem}")
                is_closed = True
            elif c_low <= tp2:
                send_telegram_msg(f"🚀 [FULL TP2 TARGET: ORDER FLOW] Kèo {act} ({t_id})\n• Entry: {entry} | TP2: {tp2}\n• LÃI: +{trade['tp2_pips']} PIPS! Chốt lời toàn bộ!")
                is_closed = True
            elif c_low <= tp1 and not trade.get('tp1_hit', False):
                trade['tp1_hit'] = True
                trade['sl'] = entry
                send_telegram_msg(f"🎉 [TING TING: ĐẠT TP1]\nKèo {act} ({t_id}) chạm TP1: {tp1} (+{trade['tp1_pips']} pips)!\n👉 Đã dời SL về hòa vốn: {entry}")
                
        if is_closed: closed_ids.add(t_id)
        else: remaining.append(trade)
            
    state['active_trades'] = remaining
    state['closed_ids'] = list(closed_ids)
    save_state(state)

# ================= 3. LẤY DỮ LIỆU NẾN & VOLUME PROFILE =================
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

# PROMPT TỰ TƯ DUY VÀ TỰ RA QUYẾT ĐỊNH (KHÔNG DÙNG CÂU MẪU)
PROMPT_ORDERFLOW_MASTER = """
Bạn là Giám đốc Quỹ Định lượng Phân tích Volume Profile & Order Flow Vàng (Institutional Specialist).
CẤU TRÚC VÙNG GIÁ TRỊ (AUCTION MARKET THEORY):
- POC (Điểm kiểm soát volume lớn nhất): {POC}
- VAH (Kháng cự Premium): {VAH} | VAL (Hỗ trợ Discount): {VAL}
- Mức giá hiện tại: {Price} (Spot XAU/USD) | Biến động ATR: {ATR} giá

HỒ SƠ ORDER FLOW 5 CÂY NẾN H1 GẦN NHẤT:
{OF_Dossier}

HÃY VẬN DỤNG TRÍ TUỆ ĐỂ TỰ SUY LUẬN & TỰ RA QUYẾT ĐỊNH:
1. Đọc vị thế trận: Tương quan {Buy_Pct}% Mua vs {Sell_Pct}% Bán và Delta ròng {Delta_Str} chứng tỏ điều gì? Phe nào đang hoàn toàn kiểm soát thế trận?
2. Kiểm tra cản: Tại POC ({POC}) hoặc mép VAH/VAL có cá mập kê lệnh hấp thụ không? Nến có đang trượt trên vùng thanh khoản mỏng không?
3. TỰ BẠN RA QUYẾT ĐỊNH: Có vào lệnh không? Lệnh gì? Điểm Entry ở đâu, SL và TP nhắm tới đâu?

BẮT BUỘC TRẢ VỀ THEO ĐÚNG CÁC THẺ SAU:
[QUYET_DINH]: BUY hoặc SELL hoặc STAND_ASIDE
[ENTRY]: {Price}
[SL]: muc_gia_sl
[TP1]: muc_gia_tp1 (tối thiểu R:R 1:1.5 - 1:2)
[TP2]: muc_gia_tp2 (tối thiểu R:R 1:2.5 - 1:3)
[PHAN_TICH]:
(Viết bài phân tích chuyên sâu 3-4 đoạn giải thích cặn kẽ thế trận dòng tiền, bẫy giá và dự báo nến tiếp theo. Tuyệt đối không chào hỏi sáo rỗng).
"""

def main():
    print("=== BOT #3: KÍCH HOẠT NÃO BỘ AI & TỰ GIẢI PHẪU LỆNH THUA ===")
    df_h1, close_time = get_spot_gold_h1()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    time_str = close_time.strftime('%H:%M - %d/%m')
    t_id = f"#OF_H1_{close_time.strftime('%d%m_%H')}H"
    
    # 1. KIỂM TRA LỆNH CŨ (NẾU DÍNH SL -> AI SẼ TỰ ĐỘNG GIẢI PHẪU NGUYÊN NHÂN NGAY!)
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
    
    is_high_vol = c_last['Volume'] > c_last['Vol_MA20'] * 1.3
    if is_high_vol and c_last['Close'] > c_last['Low'] + c_last['Range'] * 0.6: absorp_desc = "Hấp thụ lực bán ở đáy (Gom hàng)"
    elif is_high_vol and c_last['Close'] < c_last['High'] - c_last['Range'] * 0.6: absorp_desc = "Hấp thụ lực mua ở đỉnh (Xả hàng)"
    else: absorp_desc = "Khối lượng bình thường, không có hấp thụ cản"
        
    prompt = PROMPT_ORDERFLOW_MASTER.replace("{POC}", str(poc))\
                                    .replace("{VAH}", str(vah))\
                                    .replace("{VAL}", str(val))\
                                    .replace("{Price}", str(curr_price))\
                                    .replace("{ATR}", str(atr_val))\
                                    .replace("{Buy_Pct}", str(buy_pct))\
                                    .replace("{Sell_Pct}", str(sell_pct))\
                                    .replace("{Delta_Str}", delta_str)\
                                    .replace("{OF_Dossier}", of_dossier)
                                    
    # 2. GỌI BỘ NÃO AI SUY LUẬN SÂU SẮC (KHÔNG DÙNG CÂU MẪU)
    raw_ai_text = query_ai_brain(prompt, max_tokens=750)
    
    act = "STAND_ASIDE"
    if "[QUYET_DINH]:" in raw_ai_text:
        part = raw_ai_text.split("[QUYET_DINH]:")[1].split("\n")[0].strip()
        if "BUY" in part: act = "BUY"
        elif "SELL" in part: act = "SELL"
        
    analysis = raw_ai_text.split("[PHAN_TICH]:")[1].strip() if "[PHAN_TICH]:" in raw_ai_text else raw_ai_text
    
    header_info = f"📊 [BOT #3: VOLUME PROFILE & ORDER FLOW]\n" \
                  f"🎫 Mã vé: {t_id}\n" \
                  f"⏰ Nến H1 đóng: {time_str} (Khớp MT5)\n" \
                  f"📍 Giá hiện tại: {curr_price} (Spot XAU/USD)\n\n" \
                  f"🎯 VÙNG GIÁ TRỊ: POC={poc} | VAH={vah} | VAL={val}\n" \
                  f"🔬 ORDER FLOW HỢP ĐỒNG NẾN H1:\n" \
                  f"• Mua (Ask): ~{buy_v:,} HĐ ({buy_pct}%)\n" \
                  f"• Bán (Bid): ~{sell_v:,} HĐ ({sell_pct}%)\n" \
                  f"• Delta ròng: {delta_str}\n" \
                  f"• Trạng thái: {absorp_desc}\n\n"

    st = load_state()
    active_ids = [t['id'] for t in st.get("active_trades", [])]
    closed_ids = set(st.get("closed_ids", []))
    
    risk_dist = max(atr_val * 1.2, 7.0)
    sl_pips = int(risk_dist * 10)
    
    if act in ["BUY", "SELL"]:
        is_buy = (act == "BUY")
        sl_calc = round(curr_price - risk_dist if is_buy else curr_price + risk_dist, 2)
        tp1_calc = round(curr_price + (risk_dist * 2.0) if is_buy else curr_price - (risk_dist * 2.0), 2)
        tp2_calc = round(curr_price + (risk_dist * 3.0) if is_buy else curr_price - (risk_dist * 3.0), 2)
        tp1_pips = int(risk_dist * 20)
        tp2_pips = int(risk_dist * 30)
        
        photo_caption = f"{header_info}" \
                        f"🚨 QUYẾT ĐỊNH: VÀO {act} VÀNG (MARKET)\n" \
                        f"📍 Entry: {curr_price}\n" \
                        f"🛑 Cắt lỗ (SL): {sl_calc} (-{sl_pips} pips)\n" \
                        f"🎯 TP1 (Chuẩn R:R 1:2): {tp1_calc} (+{tp1_pips} pips) -> Dời hòa vốn\n" \
                        f"🚀 TP2 (Chuẩn R:R 1:3): {tp2_calc} (+{tp2_pips} pips)"
                        
        if (t_id not in active_ids) and (t_id not in closed_ids):
            st['active_trades'].append({
                "id": t_id, "action": act, "timeframe": "H1",
                "entry": curr_price, "sl": sl_calc, "tp1": tp1_calc, "tp2": tp2_calc,
                "sl_pips": sl_pips, "tp1_pips": tp1_pips, "tp2_pips": tp2_pips, "tp1_hit": False,
                "context": f"Order Flow Delta {delta_str}, {sell_pct}% Bán vs {buy_pct}% Mua"
            })
            save_state(st)
    else:
        photo_caption = f"{header_info}🚫 QUYẾT ĐỊNH: ĐỨNG NGOÀI QUAN SÁT (AI ĐÁNH GIÁ CHƯA ĐỦ ĐIỀU KIỆN)"

    send_telegram_photo(chart_file, photo_caption)
    time.sleep(1)
    
    # BÀI PHÂN TÍCH THỰC SỰ TỪ BỘ NÃO AI (HOÀN TOÀN KHÔNG DÙNG CÂU MẪU)
    send_telegram_msg(f"💡 [PHÂN TÍCH TƯ DUY AI ORDER FLOW - {t_id}]\n\n{analysis}")
    if os.path.exists(chart_file): os.remove(chart_file)

if __name__ == "__main__":
    main()

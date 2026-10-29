import os
import time
import base64
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

TICKER = "GC=F"
LOOKBACK = 60

client = Groq(api_key=GROQ_API_KEY)

def get_h1_and_h4_data():
    gold = yf.Ticker(TICKER)
    df_h1 = gold.history(period="1mo", interval="1h")
    df_h1 = df_h1[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df_h1.index = pd.to_datetime(df_h1.index)
    df_h1 = df_h1.iloc[:-1] # Bỏ nến H1 đang chạy dở
    
    # 1. Tính EMA 21 trên H1
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    
    # 2. TÍNH CHỈ SỐ ATR 14 ĐỂ ĐO ĐỘ BIẾN ĐỘNG LINH HOẠT
    high_low = df_h1['High'] - df_h1['Low']
    high_close = np.abs(df_h1['High'] - df_h1['Close'].shift())
    low_close = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_h1['ATR'] = tr.rolling(14).mean()
    
    # 3. Gộp nến H1 thành H4 để lấy xu hướng lớn
    df_h4 = df_h1.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    
    return df_h1.tail(LOOKBACK), h4_is_bull

def find_smc_levels(df):
    fvg_list = []
    for i in range(2, len(df)):
        c1_high, c1_low = df['High'].iloc[i-2], df['Low'].iloc[i-2]
        c3_high, c3_low = df['High'].iloc[i], df['Low'].iloc[i]
        if c3_low > c1_high:
            fvg_list.append({'idx_start': i-2, 'idx_end': len(df)-1, 'top': c3_low, 'bottom': c1_high, 'color': 'green'})
        elif c3_high < c1_low:
            fvg_list.append({'idx_start': i-2, 'idx_end': len(df)-1, 'top': c1_low, 'bottom': c3_high, 'color': 'red'})
    return fvg_list[-3:]

def generate_chart(df, fvg_list, h4_trend_text):
    chart_path = "xauusd_h1_chart.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    
    fig, axlist = mpf.plot(
        df, type='candle', style='binance', addplot=ema_plot,
        title=f"\n[XAUUSD H1] Xu huong H4: {h4_trend_text}", figsize=(14, 7), returnfig=True
    )
    ax = axlist[0]
    for fvg in fvg_list:
        w = fvg['idx_end'] - fvg['idx_start'] + 1
        h = fvg['top'] - fvg['bottom']
        rect = patches.Rectangle((fvg['idx_start'], fvg['bottom']), w, h, edgecolor=fvg['color'], facecolor=fvg['color'], alpha=0.25)
        ax.add_patch(rect)
        
    fig.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return chart_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

PROMPT_TEMPLATE = """
Bạn là Phù thủy Giao dịch Vàng (XAUUSD CFD) theo trường phái Price Action & SMC TraderViet.
Quan sát biểu đồ nến H1 đính kèm (có đường EMA 21 cam, các hộp FVG xanh/đỏ).
Xu hướng khung lớn H4 đang là: {H4_Trend}.
Kế hoạch lệnh Market ngay: {Action} tại giá {Price}.
Khoảng cách Cắt lỗ (SL) linh hoạt theo ATR: {Risk_Dist} giá | Chốt lời (TP) R:R 1:3: {Reward_Dist} giá.

HÃY ĐƯA RA NHẬN ĐỊNH NGẮN GỌN (Dưới 400 ký tự):
1. Phân tích nến H1: Nến vừa đóng cửa có từ chối cản, Pin Bar hay thuận đà xu hướng H4 không?
2. Vì sao khoảng cách SL theo ATR ({Risk_Dist} giá) tại đây là hợp lý và an toàn tránh râu nến quét?
3. Lời khuyên vào lệnh kỷ luật.
"""

def send_telegram_photo(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})

def main():
    print("=== QUÉT VÀNG: H4 TREND + H1 TRIGGER + SL LINH HOẠT THEO ATR ===")
    try:
        df_h1, h4_is_bull = get_h1_and_h4_data()
        fvgs = find_smc_levels(df_h1)
        
        h4_trend_str = "TĂNG (ƯU TIÊN BUY)" if h4_is_bull else "GIẢM (ƯU TIÊN SELL)"
        chart_file = generate_chart(df_h1, fvgs, h4_trend_str)
        base64_img = encode_image(chart_file)
        
        curr_price = round(df_h1['Close'].iloc[-1], 2)
        
        # 1. ĐO ĐỘ BIẾN ĐỘNG THEO ATR HIỆN TẠI
        raw_atr = df_h1['ATR'].iloc[-1]
        if np.isnan(raw_atr) or raw_atr == 0:
            raw_atr = 8.5 # Giá trị dự phòng nếu thiếu nến
            
        # Hệ số 1.2 * ATR để SL nằm an toàn ngoài tầm râu nến quét
        risk_dist = round(raw_atr * 1.2, 2)
        reward_dist = round(risk_dist * 3.0, 2) # Tỷ lệ R:R = 1:3
        
        # 2. TÍNH ENTRY, SL VÀ TP LINH HOẠT
        if h4_is_bull:
            action = "BUY NGAY"
            entry = curr_price
            sl = round(entry - risk_dist, 2)
            tp = round(entry + reward_dist, 2)
        else:
            action = "SELL NGAY"
            entry = curr_price
            sl = round(entry + risk_dist, 2)
            tp = round(entry - reward_dist, 2)
            
        prompt = PROMPT_TEMPLATE.replace("{H4_Trend}", h4_trend_str)\
                                .replace("{Action}", action)\
                                .replace("{Price}", str(curr_price))\
                                .replace("{Risk_Dist}", str(risk_dist))\
                                .replace("{Reward_Dist}", str(reward_dist))
                                
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]}],
            max_tokens=350
        )
        ai_comment = response.choices[0].message.content.strip()
        
        tele_msg = f"🏆 [CHIẾN LƯỢC VÀNG: H4 TREND + H1 TRIGGER]\n" \
                   f"• Xu hướng H4: {h4_trend_str}\n" \
                   f"👉 HÀNH ĐỘNG: {action} (MARKET ORDER)\n" \
                   f"• Giá vào ngay (Entry): {entry}\n" \
                   f"• Cắt lỗ (SL): {sl} (Linh hoạt: {risk_dist} giá theo ATR)\n" \
                   f"• Chốt lời (TP): {tp} (Mục tiêu: {reward_dist} giá - R:R 1:3)\n\n" \
                   f"💡 PHÂN TÍCH NHANH TỪ AI:\n{ai_comment}"
                   
        print("Đang gửi báo cáo lệnh linh hoạt theo ATR về Telegram...")
        send_telegram_photo(chart_file, tele_msg)
        
        if os.path.exists(chart_file):
            os.remove(chart_file)
            
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()

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
client = Groq(api_key=GROQ_API_KEY)

def get_market_data():
    gold = yf.Ticker(TICKER)
    # Lấy 10 ngày gần nhất (siêu nhẹ, không bao giờ bị nghẽn)
    df_h1 = gold.history(period="10d", interval="1h")
    
    if df_h1.empty or len(df_h1) < 20:
        raise ValueError("Chưa lấy được dữ liệu nến từ sàn, đang thử lại!")
        
    df_h1 = df_h1[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df_h1.index = pd.to_datetime(df_h1.index)
    df_h1 = df_h1.iloc[:-1] # Bỏ nến đang chạy dở
    
    # Tính EMA 21 & ATR 14 cho H1
    df_h1['EMA21'] = df_h1['Close'].ewm(span=21, adjust=False).mean()
    high_low = df_h1['High'] - df_h1['Low']
    high_close = np.abs(df_h1['High'] - df_h1['Close'].shift())
    low_close = np.abs(df_h1['Low'] - df_h1['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_h1['ATR'] = tr.rolling(14).mean()
    
    # Gộp nến H1 thành H4
    df_h4 = df_h1.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    df_h4['EMA21'] = df_h4['Close'].ewm(span=21, adjust=False).mean()
    
    last_h1_time = df_h1.index[-1]
    hour_in_4h = (last_h1_time.hour % 4) + 1
    is_4h_closed = (hour_in_4h == 4)
    
    return df_h1.tail(60), df_h4.tail(40), is_4h_closed, hour_in_4h

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

def generate_chart(df, fvg_list, title_text):
    chart_path = "xauusd_chart.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    
    fig, axlist = mpf.plot(
        df, type='candle', style='binance', addplot=ema_plot,
        title=f"\n{title_text}", figsize=(14, 7), returnfig=True
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

PROMPT_4H_CLOSED = """
Bạn là Phù thủy Giao dịch SMC & Price Action TraderViet.
Quan sát biểu đồ nến 4H của Vàng (XAUUSD) VỪA CHÍNH THỨC ĐÓNG CỬA.
Giá hiện tại: {Price}. Đường EMA 21 màu cam.

HÃY ĐÁNH GIÁ (Dưới 500 ký tự):
1. GIẢI PHẪU CÂY NẾN 4H VỪA ĐÓNG: Là nến gì? Phản ứng cản/FVG ra sao?
2. QUYẾT ĐỊNH LỆNH SWING 4H: [VÀO LỆNH NGAY (BUY/SELL)] HOẶC [KHÔNG VÀO LỆNH]. (Kèm SL: {SL} | TP: {TP} nếu vào).
3. DỰ ĐOÁN NẾN 4H MỚI: Dự kiến hướng đi của cây nến 4H tiếp theo vừa ra đời.
"""

PROMPT_HOURLY_H1 = """
Bạn là Phù thủy Giao dịch SMC & Price Action TraderViet.
Quan sát biểu đồ nến H1 vừa đóng cửa. Giá hiện tại: {Price}. Xu hướng lớn 4H: {H4_Trend}.
Cây nến 4H hiện tại đang ở TIẾN TRÌNH GIỜ THỨ {Hour_In_4H}/4.

HÃY PHÂN TÍCH VÀ DỰ ĐOÁN (Dưới 450 ký tự):
1. QUYẾT ĐỊNH NẾN 1H VỪA ĐÓNG: [VÀO LỆNH NGAY (BUY/SELL)] HOẶC [KHÔNG VÀO LỆNH (ĐỨNG NGOÀI)].
- Nêu lý do: Tín hiệu nến 1H vừa đóng có đủ chuẩn hay đang giằng co?

2. DỰ ĐOÁN NẾN 4H TRONG 1 GIỜ TIẾP THEO:
- Trong 1 tiếng tới, nến 4H dự kiến sẽ di chuyển thế nào (tiếp diễn đà tăng/giảm hay rút râu)?
- Dự kiến trong 1 giờ tới nến 4H sẽ kiểm tra mức giá/vùng cản nào?
"""

def send_telegram_photo(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        res = requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})
        print(f"👉 KẾT QUẢ GỬI TELEGRAM: {res.status_code}")

def main():
    print("=== BẮT ĐẦU QUÉT NẾN VÀNG ===")
    df_h1, df_h4, is_4h_closed, hour_in_4h = get_market_data()
    curr_price = round(df_h1['Close'].iloc[-1], 2)
    
    raw_atr = df_h1['ATR'].iloc[-1]
    risk_dist = round((raw_atr if not np.isnan(raw_atr) else 8.5) * 1.2, 2)
    reward_dist = round(risk_dist * 3.0, 2)
    
    h4_is_bull = df_h4['Close'].iloc[-1] > df_h4['EMA21'].iloc[-1]
    h4_trend_str = "TĂNG" if h4_is_bull else "GIẢM"
    
    if is_4h_closed:
        print("🔥 NẾN 4H VỪA ĐÓNG CỬA! GỬI BÁO CÁO SWING 4H...")
        fvgs_h4 = find_smc_levels(df_h4)
        chart_file = generate_chart(df_h4, fvgs_h4, f"[XAUUSD 4H] NEN 4H VUA DONG CUA - XU HUONG: {h4_trend_str}")
        base64_img = encode_image(chart_file)
        
        sl_val = round(curr_price - risk_dist if h4_is_bull else curr_price + risk_dist, 2)
        tp_val = round(curr_price + reward_dist if h4_is_bull else curr_price - reward_dist, 2)
        
        prompt = PROMPT_4H_CLOSED.replace("{Price}", str(curr_price))\
                                 .replace("{SL}", str(sl_val))\
                                 .replace("{TP}", str(tp_val))
                                 
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]}],
            max_tokens=450
        )
        analysis = response.choices[0].message.content.strip()
        
        caption = f"👑 [BẢN TIN ĐẶC BIỆT: NẾN 4H VỪA ĐÓNG CỬA]\n" \
                  f"⏰ Mức giá đóng nến 4H: {curr_price}\n" \
                  f"📊 Cấu trúc xu hướng 4H: {h4_trend_str}\n\n" \
                  f"{analysis}"
    else:
        print(f"-> Nến 1H vừa đóng. Nến 4H đang ở giờ thứ {hour_in_4h}/4...")
        fvgs_h1 = find_smc_levels(df_h1)
        chart_file = generate_chart(df_h1, fvgs_h1, f"[XAUUSD H1] Tien trinh 4H: Gio {hour_in_4h}/4")
        base64_img = encode_image(chart_file)
        
        prompt = PROMPT_HOURLY_H1.replace("{Price}", str(curr_price))\
                                 .replace("{H4_Trend}", h4_trend_str)\
                                 .replace("{Hour_In_4H}", str(hour_in_4h))
                                 
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]}],
            max_tokens=450
        )
        analysis = response.choices[0].message.content.strip()
        
        caption = f"🔔 [BÁO CÁO NẾN 1H & DỰ ĐOÁN 4H (TIẾP THEO 1 GIỜ)]\n" \
                  f"⏰ Giá nến 1H vừa đóng: {curr_price}\n" \
                  f"⏳ Tiến trình nến 4H: Đang ở giờ thứ {hour_in_4h}/4\n" \
                  f"🎯 SL tham chiếu (ATR): {risk_dist} giá | TP (1:3): {reward_dist} giá\n\n" \
                  f"{analysis}"
    
    print("Đang gửi báo cáo về Telegram...")
    send_telegram_photo(chart_file, caption)
    
    if os.path.exists(chart_file):
        os.remove(chart_file)

if __name__ == "__main__":
    main()

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

WATCHLIST = [
    {"ticker": "GC=F",    "name": "VÀNG (XAUUSD)"},
    {"ticker": "BTC-USD", "name": "BITCOIN (BTC/USD)"}
]

TIMEFRAME = "15m"
LOOKBACK = 60

client = Groq(api_key=GROQ_API_KEY)

def get_market_data(ticker):
    asset = yf.Ticker(ticker)
    df = asset.history(period="5d", interval=TIMEFRAME)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.iloc[:-1].tail(LOOKBACK)
    
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # Tính ATR 14
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    return df

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

def generate_chart(df, fvg_list, asset_name):
    clean_name = asset_name.replace("/", "_").replace(" ", "_")
    chart_path = f"{clean_name}_chart.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    
    fig, axlist = mpf.plot(
        df, type='candle', style='binance', addplot=ema_plot,
        title=f"\n[SMC & PA M15] {asset_name}", figsize=(14, 7), returnfig=True
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
Bạn là Chuyên gia Giao dịch SMC & Price Action TraderViet.
Quan sát biểu đồ M15 của {Asset_Name}.
Trader sẽ VÀO LỆNH TRỰC TIẾP NGAY LẬP TỨC (LỆNH MARKET, KHÔNG DÙNG LIMIT).
Hệ thống đã chọn lệnh: {Chosen_Action} tại giá hiện tại {Current_Price}.

HÃY ĐƯA RA LỜI BÌNH LUẬN NHANH (Viết dưới 500 ký tự):
1. Cây nến M15 vừa đóng cửa có tín hiệu gì (Pin Bar, đè nén nến, hay bám EMA 21)?
2. Lý do ủng hộ việc {Chosen_Action} NGAY LÚC NÀY.
3. Lưu ý ngắn gọn về quản trị lệnh.
"""

def send_telegram_photo(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})

def main():
    print(f"=== BẮT ĐẦU QUÉT LỆNH MARKET TRỰC TIẾP ===")
    
    for item in WATCHLIST:
        ticker = item["ticker"]
        name = item["name"]
        print(f"\n-> Đang xử lý: {name}...")
        
        try:
            df = get_market_data(ticker)
            fvgs = find_smc_levels(df)
            chart_file = generate_chart(df, fvgs, name)
            base64_img = encode_image(chart_file)
            
            curr_price = df['Close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            if np.isnan(atr) or atr == 0:
                atr = curr_price * 0.003
                
            # Xác định xu hướng để vào lệnh Market ngay
            is_uptrend = curr_price > df['EMA21'].iloc[-1]
            chosen_action = "BUY NGAY" if is_uptrend else "SELL NGAY"
            
            # Tính Entry tại giá đóng cửa hiện tại, SL theo ATR, TP tỷ lệ 1:2.5
            entry_val = round(curr_price, 2)
            if is_uptrend:
                sl_val = round(curr_price - (atr * 1.5), 2)
                tp_val = round(curr_price + ((entry_val - sl_val) * 2.5), 2)
            else:
                sl_val = round(curr_price + (atr * 1.5), 2)
                tp_val = round(curr_price - ((sl_val - entry_val) * 2.5), 2)
                
            prompt = PROMPT_TEMPLATE.replace("{Asset_Name}", name)\
                                    .replace("{Current_Price}", str(curr_price))\
                                    .replace("{Chosen_Action}", chosen_action)
                                    
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]}],
                max_tokens=400
            )
            ai_opinion = response.choices[0].message.content.strip()
            
            # ĐÓNG GÓI BẢN TIN VÀO LỆNH MARKET THỰC CHIẾN
            tele_caption = f"⚡ [LỆNH THỊ TRƯỜNG M15: {name}]\n" \
                           f"👉 HÀNH ĐỘNG: {chosen_action} (MARKET ORDER)\n" \
                           f"• Giá vào ngay (Entry): {entry_val}\n" \
                           f"• Cắt lỗ (SL): {sl_val}\n" \
                           f"• Chốt lời (TP): {tp_val} (R:R 1:2.5)\n\n" \
                           f"💡 PHÂN TÍCH NHANH CỦA AI:\n{ai_opinion}"
            
            print(f"-> Đang gửi lệnh Market {name} về Telegram...")
            send_telegram_photo(chart_file, tele_caption)
                
            if os.path.exists(chart_file):
                os.remove(chart_file)
            time.sleep(1)
            
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()

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
    {"ticker": "GC=F",     "name": "VÀNG (XAUUSD)"},
    {"ticker": "BTC-USD",  "name": "BITCOIN (BTC/USD)"},
    {"ticker": "EURUSD=X", "name": "EUR/USD"},
    {"ticker": "GBPUSD=X", "name": "GBP/USD"}
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

# PROMPT TÍCH HỢP CẢ KÈO A+ VÀ KÈO B+
PROMPT_TEMPLATE = """
Bạn là Chuyên gia Giao dịch SMC & Price Action TraderViet phân tích khung M15 cho: {Asset_Name}.
Quan sát biểu đồ nến M15 đính kèm (Nến Nhật, đường EMA 21 cam, các hộp FVG xanh/đỏ).

HÃY ĐÁNH GIÁ VÀ XẾP LOẠI KÈO THEO 3 TRƯỜNG HỢP:

1. [KÈO A+] - SETUP KIM CƯƠNG (R:R = 1:3):
- Đủ 4 tiêu chí: Thuận trend EMA 21 + Chạm FVG + Có quét thanh khoản đỉnh đáy + Nến Pin Bar rút râu hoặc Fakey.
Mẫu trả về:
🚨 [KÈO A+ M15: {Asset_Name}]
• Hướng lệnh: [BUY / SELL]
• Tín hiệu: [Nêu hành vi quét thanh khoản + Pin Bar rút râu]
• Entry: [Mức giá] | SL: [Sau râu nến] | TP: [R:R 1:3]
• Khuyến nghị vốn: Đi đủ 1% Risk

2. [KÈO B+] - SETUP LƯỚT SÓNG NÂNG CAO (R:R = 1:2):
- Thỏa mãn: Thuận trend EMA 21 + Giá hồi (Pullback) chạm EMA 21 hoặc FVG + Xuất hiện nến Pin Bar hoặc Engulfing (không cần quét đáy sâu).
Mẫu trả về:
⚡ [KÈO B+ M15: {Asset_Name}]
• Hướng lệnh: [BUY / SELL]
• Tín hiệu: [Phản ứng bật tăng/giảm đẹp tại EMA 21 hoặc FVG]
• Entry: [Mức giá] | SL: [Sau đáy/đỉnh nến gần nhất] | TP: [R:R 1:2]
• Khuyến nghị vốn: Đi 0.5% Risk (Lướt sóng nhanh)

3. [THEO DÕI] - KHÔNG CÓ KÈO:
- Thị trường nến nhỏ giằng co, quấn chặt EMA hoặc không có nến đảo chiều.
Mẫu trả về đúng 2 dòng:
📊 [{Asset_Name} - M15]: Xu hướng [Tăng/Giảm/Sideway]. Trạng thái: [Đang tích lũy, ĐỨNG NGOÀI].
"""

def send_telegram_photo(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})

def send_telegram_text(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text})

def main():
    print(f"=== BẮT ĐẦU QUÉT ĐA TÀI SẢN M15 (LỌC KÈO A+ & B+) ===")
    summary_reports = []
    
    for item in WATCHLIST:
        ticker = item["ticker"]
        name = item["name"]
        print(f"\n-> Đang phân tích: {name}...")
        
        try:
            df = get_market_data(ticker)
            fvgs = find_smc_levels(df)
            chart_file = generate_chart(df, fvgs, name)
            base64_img = encode_image(chart_file)
            
            prompt = PROMPT_TEMPLATE.replace("{Asset_Name}", name)
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]}],
                max_tokens=450
            )
            analysis = response.choices[0].message.content.strip()
            print(analysis)
            
            # NẾU CÓ KÈO A+ HOẶC KÈO B+ -> BẮN NGAY ẢNH + THÔNG SỐ VÀO LỆNH
            if "[KÈO A+]" in analysis or "[KÈO B+]" in analysis:
                print(f"🔥 Phát hiện cơ hội vào lệnh trên {name}! Đang gửi Telegram...")
                send_telegram_photo(chart_file, analysis)
            else:
                summary_reports.append(analysis)
                
            if os.path.exists(chart_file):
                os.remove(chart_file)
            time.sleep(1)
            
        except Exception as e:
            print(f"Lỗi quét {name}: {e}")
            
    # GỬI TỔNG HỢP CÁC CẶP ĐANG THEO DÕI
    if summary_reports:
        digest_msg = "📡 [BÁO CÁO TỔNG QUAN THỊ TRƯỜNG M15]\n" + "\n".join(summary_reports)
        send_telegram_text(digest_msg)

if __name__ == "__main__":
    main()

import os
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

# Lấy Key từ GitHub Secrets
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

TICKER = "GC=F"
TIMEFRAME = "1h"
LOOKBACK = 60

client = Groq(api_key=GROQ_API_KEY)

def get_gold_data():
    gold = yf.Ticker(TICKER)
    df = gold.history(period="7d", interval=TIMEFRAME)
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

def generate_gold_chart(df, fvg_list):
    chart_path = "xauusd_chart.png"
    ema_plot = mpf.make_addplot(df['EMA21'], color='#ff9800', width=1.5)
    
    fig, axlist = mpf.plot(
        df, type='candle', style='binance', addplot=ema_plot,
        title=f"\n[SMC & PA] XAUUSD CFD - 1H", figsize=(14, 7), returnfig=True
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

XAUUSD_PROMPT = """
Bạn là Phù thủy Giao dịch Vàng (XAUUSD CFD) theo trường phái SMC và Price Action TraderViet.
Quan sát biểu đồ nến Vàng đính kèm (Nến Nhật, đường EMA 21 cam, các hộp FVG xanh/đỏ):

NHIỆM VỤ:
1. Xác định xu hướng hiện tại (Tăng hay Giảm so với EMA 21).
2. Kiểm tra vùng POI: Giá có chạm FVG hoặc EMA 21 không?
3. Kiểm tra nến kích hoạt: Có nến Pin Bar rút râu quét thanh khoản hoặc bẫy giá Fakey không?

QUY TẮC:
- Nếu thị trường chưa rõ hoặc không có setup đẹp: Chỉ trả về duy nhất từ "SKIP".
- Nếu xuất hiện KÈO A+ ĐẠT CHUẨN: Trả về bản tin theo mẫu sau:

🏆 [CẢNH BÁO BẮN KÈO XAUUSD CFD - H1]
• Lệnh đề xuất: [BUY / SELL]
• Lý do: [Phân tích ngắn gọn hành vi quét thanh khoản/FVG]
• Điểm vào (Entry): [Giá cụ thể]
• Cắt lỗ (Stop Loss): [Đặt sau râu nến quét thanh khoản]
• Chốt lời (Take Profit): [Target tỷ lệ R:R = 1:3]
"""

def send_telegram(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})

def main():
    print("Đang quét dữ liệu nến Vàng...")
    df = get_gold_data()
    fvgs = find_smc_levels(df)
    chart_file = generate_gold_chart(df, fvgs)
    
    base64_image = encode_image(chart_file)
    
    # Gửi sang Groq Vision (Mô hình Llama 3.2 Vision siêu tốc độ)
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": XAUUSD_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )
    
    analysis = response.choices[0].message.content.strip()
    
    if "SKIP" in analysis or len(analysis) < 15:
        print("Không có kèo đạt chuẩn. Đứng ngoài.")
    else:
        print("Tìm thấy kèo A+! Đang gửi Telegram...")
        send_telegram(chart_file, analysis)
        
    if os.path.exists(chart_file):
        os.remove(chart_file)

if __name__ == "__main__":
    main()

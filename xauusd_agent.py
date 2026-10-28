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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

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

# PROMPT ĐẶC BIỆT CHO CHẾ ĐỘ BÁO CÁO MỖI TIẾNG (DƯỚI 800 KÝ TỰ)
XAUUSD_HOURLY_PROMPT = """
Bạn là Chuyên gia Phân tích Vàng (XAUUSD CFD) theo trường phái SMC và Price Action TraderViet.
Quan sát biểu đồ nến Vàng H1 đính kèm (Nến Nhật, đường EMA 21 cam, các hộp FVG xanh/đỏ).

Hãy đưa ra báo cáo ngắn gọn (dưới 800 ký tự) theo đúng 1 trong 2 trường hợp sau:

TRƯỜNG HỢP 1: NẾU CÓ KÈO A+ (Đủ 4 tiêu chí: Cấu trúc + POI + Quét thanh khoản + Nến Pin Bar/Fakey):
🚨 [KÈO VÀNG A+ ĐẠT CHUẨN - H1]
• Lệnh: [BUY / SELL]
• Tín hiệu: [Mô tả Pin Bar/Fakey quét FVG]
• Entry: [Mức giá vào]
• Stop Loss: [Cắt lỗ sau râu nến]
• Take Profit: [Chốt lời tỷ lệ 1:3]

TRƯỜNG HỢP 2: NẾU CHƯA CÓ KÈO (Thị trường đi ngang / Chưa có bẫy thanh khoản):
📊 [BÁO CÁO THỊ TRƯỜNG VÀNG - H1]
• Xu hướng: [Tăng / Giảm / Đi ngang quanh EMA 21]
• Trạng thái nến: [Nêu hành động nến hiện tại quanh vùng cản/FVG]
• Khuyến nghị: ĐỨNG NGOÀI QUAN SÁT (Chưa có setup đạt chuẩn)
• Kịch bản chờ đợi: [Nêu ngắn gọn cần chờ giá quét đỉnh/đáy nào để có kèo]
"""

def send_telegram(photo_path, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        res = requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': text}, files={'photo': photo})
        print(f"👉 KẾT QUẢ GỬI TELEGRAM: {res.status_code}")

def main():
    print("Đang quét dữ liệu nến Vàng...")
    df = get_gold_data()
    fvgs = find_smc_levels(df)
    chart_file = generate_gold_chart(df, fvgs)
    
    base64_image = encode_image(chart_file)
    
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": XAUUSD_HOURLY_PROMPT},
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
    print("AI Phân tích:\n", analysis)
    
    # Gửi cả ảnh biểu đồ + phân tích chi tiết của AI về Telegram
    send_telegram(chart_file, analysis)
        
    if os.path.exists(chart_file):
        os.remove(chart_file)

if __name__ == "__main__":
    main()

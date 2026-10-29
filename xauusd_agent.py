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

# 1. LẤY KHÓA BẢO MẬT VÀ LÀM SẠCH KÝ TỰ
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

if TELEGRAM_BOT_TOKEN.startswith("bot"):
    TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN[3:]

TICKER = "GC=F" # Vàng XAUUSD
client = Groq(api_key=GROQ_API_KEY)

# 2. LẤY VÀ XỬ LÝ DỮ LIỆU NẾN H1 & H4
def get_market_data():
    gold = yf.Ticker(TICKER)
    df_h1 = gold.history(period="1mo", interval="1h")
    df_h1 = df_h1[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df_h1.index = pd.to_datetime(df_h1.index)
    df_h1 = df_h

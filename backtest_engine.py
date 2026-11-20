import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()
VN_TZ = timezone(timedelta(hours=7))
SPREAD_PTS = 0.30  # Phí giãn spread trung bình của Vàng XAU/USD (~3 pips)

def fetch_historical_h1(outputsize=1000):
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Thiếu TWELVE_DATA_API_KEY trong GitHub Secrets!")
    url = (
        f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h"
        f"&outputsize={outputsize}&timezone=Asia/Ho_Chi_Minh&apikey={TWELVE_DATA_API_KEY}"
    )
    for attempt in range(3):
        try:
            res = requests.get(url, timeout=20).json()
            if "values" in res:
                break
        except Exception as e:
            if attempt == 2:
                raise ValueError(f"Lỗi tải dữ liệu lịch sử: {e}")
            time.sleep(5)

    df = pd.DataFrame(res["values"]).iloc[::-1].reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)

    # Chỉ báo kỹ thuật & Data nến
    df["Range"] = df["High"] - df["Low"]
    df["Volume"] = (df["Range"] * 1200).astype(int) + 600
    body = df["Close"] - df["Open"]
    u_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
    l_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]

    eff_score = np.where(df["Range"] > 0, (body + (l_wick - u_wick) * 0.6) / df["Range"], 0.0)
    buy_ratio = np.clip(0.5 + eff_score * 0.45, 0.12, 0.88)
    df["Buy_Vol"] = (df["Volume"] * buy_ratio * 15).astype(int)
    df["Sell_Vol"] = (df["Volume"] * (1 - buy_ratio) * 15).astype(int)
    df["Delta"] = df["Buy_Vol"] - df["Sell_Vol"]
    df["CVD_6"] = df["Delta"].rolling(6).sum()

    typ_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["VWAP"] = (typ_price * df["Volume"]).rolling(24).sum() / df["Volume"].rolling(24).sum()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    h_l = df["High"] - df["Low"]
    h_c = np.abs(df["High"] - df["Close"].shift())
    l_c = np.abs(df["Low"] - df["Close"].shift())
    df["ATR"] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    return df.dropna()

def calc_vp_fast(window_df, n_bins=24):
    p_min, p_max = window_df["Low"].min(), window_df["High"].max()
    if p_max <= p_min:
        return p_min, p_max, p_min
    bins = np.linspace(p_min, p_max, n_bins)
    vol_prof = np.zeros(n_bins - 1)
    highs, lows, vols = window_df["High"].values, window_df["Low"].values, window_df["Volume"].values
    for h, l, v in zip(highs, lows, vols):
        if h == l or v == 0:
            continue
        ov = np.maximum(0, np.minimum(h, bins[1:]) - np.maximum(l, bins[:-1]))
        vol_prof += v * (ov / (h - l))
    b_cents = (bins[:-1] + bins[1:]) / 2.0
    poc_idx = int(np.argmax(vol_prof))
    poc = float(b_cents[poc_idx])
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
            break
    return round(poc, 2), round(float(b_cents[right]), 2), round(float(b_cents[left]), 2)

def simulate_trade(df, start_idx, act, entry, sl, tp1, tp2, max_bars=36):
    """Giả lập khớp lệnh có tính Spread & dời SL về hòa vốn (BE) khi chạm TP1."""
    adj_entry = entry + SPREAD_PTS if act == "BUY" else entry - SPREAD_PTS
    tp1_hit = False
    cur_sl = sl
    tp1_pips = int(round(abs(tp1 - adj_entry) * 10))
    tp2_pips = int(round(abs(tp2 - adj_entry) * 10))
    sl_pips = int(round(abs(adj_entry - sl) * 10))

    end_idx = min(start_idx + max_bars, len(df) - 1)
    for j in range(start_idx + 1, end_idx + 1):
        row = df.iloc[j]
        if act == "BUY":
            if row["Low"] <= cur_sl:
                if tp1_hit:
                    return "TP1 + BE", int(tp1_pips * 0.5), j
                return "SL", -sl_pips, j
            if row["High"] >= tp2:
                return "FULL TP2", int(tp1_pips * 0.5 + tp2_pips * 0.5), j
            if row["High"] >= tp1 and not tp1_hit:
                tp1_hit = True
                cur_sl = adj_entry
        else:
            if row["High"] >= cur_sl:
                if tp1_hit:
                    return "TP1 + BE", int(tp1_pips * 0.5), j
                return "SL", -sl_pips, j
            if row["Low"] <= tp2:
                return "FULL TP2", int(tp1_pips * 0.5 + tp2_pips * 0.5), j
            if row["Low"] <= tp1 and not tp1_hit:
                tp1_hit = True
                cur_sl = adj_entry

    close_p = df.iloc[end_idx]["Close"]
    raw_pips = int(round((close_p - adj_entry) * 10)) if act == "BUY" else int(round((adj_entry - close_p) * 10))
    return "TIME EXIT", raw_pips, end_idx

def run_bot1_backtest(df):
    """Kiểm thử Bot #1: Price Action TraderViet (Pin Bar, Fakey, Engulfing, Trend Bar)."""
    trades = []
    i = 60
    while i < len(df) - 2:
        t = df.index[i]
        if t.hour in [2, 3, 4, 5]:
            i += 1
            continue
        c0, c1, c2 = df.iloc[i], df.iloc[i - 1], df.iloc[i - 2]
        curr_price = c0["Close"]
        atr = float(np.clip(c0["ATR"], 5.5, 12.0))
        r0 = max(c0["Range"], 0.1)
        body0 = abs(c0["Close"] - c0["Open"])
        u_wick0 = c0["High"] - max(c0["Open"], c0["Close"])
        l_wick0 = min(c0["Open"], c0["Close"]) - c0["Low"]
        body1 = abs(c1["Close"] - c1["Open"])
        r1 = max(c1["Range"], 0.1)

        # Xu hướng vĩ mô đơn giản hóa tương đương H4
        uptrend = (c0["Close"] >= c0["EMA21"] - 2.0) and (c0["EMA21"] >= c0["EMA50"] - 3.0)
        downtrend = (c0["Close"] <= c0["EMA21"] + 2.0) and (c0["EMA21"] <= c0["EMA50"] + 3.0)

        is_inside = (c1["High"] <= c2["High"] + 1.0) and (c1["Low"] >= c2["Low"] - 1.0)
        bull_fakey = is_inside and (c0["Low"] < min(c1["Low"], c2["Low"])) and (c0["Close"] > c1["Low"] + r1 * 0.4) and (c0["Close"] > c0["Open"])
        bear_fakey = is_inside and (c0["High"] > max(c1["High"], c2["High"])) and (c0["Close"] < c1["High"] - r1 * 0.4) and (c0["Close"] < c0["Open"])
        bull_pin = (l_wick0 >= r0 * 0.45) and (u_wick0 <= r0 * 0.30) and (r0 >= 4.5)
        bear_pin = (u_wick0 >= r0 * 0.45) and (l_wick0 <= r0 * 0.30) and (r0 >= 4.5)
        bull_eng = (c1["Close"] < c1["Open"]) and (c0["Close"] > c0["Open"]) and (body0 >= body1 * 1.05) and (body0 >= r0 * 0.45)
        bear_eng = (c1["Close"] > c1["Open"]) and (c0["Close"] < c0["Open"]) and (body0 >= body1 * 1.05) and (body0 >= r0 * 0.45)
        bull_tb = uptrend and (c0["Close"] > c0["Open"]) and (body0 >= r0 * 0.60) and (body0 >= atr * 0.65)
        bear_tb = downtrend and (c0["Close"] < c0["Open"]) and (body0 >= r0 * 0.60) and (body0 >= atr * 0.65)

        act, setup = None, None
        if uptrend and (c0["Low"] <= max(c0["EMA10"], c0["EMA21"]) + 3.0):
            if bull_fakey: act, setup = "BUY", "Fakey"
            elif bull_pin: act, setup = "BUY", "Pin Bar"
            elif bull_eng: act, setup = "BUY", "Engulfing"
            elif bull_tb: act, setup = "BUY", "Trend Bar"
        elif downtrend and (c0["High"] >= min(c0["EMA10"], c0["EMA21"]) - 3.0):
            if bear_fakey: act, setup = "SELL", "Fakey"
            elif bear_pin: act, setup = "SELL", "Pin Bar"
            elif bear_eng: act, setup = "SELL", "Engulfing"
            elif bear_tb: act, setup = "SELL", "Trend Bar"

        if act:
            raw_dist = (curr_price - min(c0["Low"], c1["Low"]) + 2.0) if act == "BUY" else (max(c0["High"], c1["High"]) - curr_price + 2.0)
            if raw_dist <= 15.0:
                risk = float(np.clip(raw_dist, 8.5, 15.0))
                sl = round(curr_price - risk, 2) if act == "BUY" else round(curr_price + risk, 2)
                tp1 = round(curr_price + risk * 1.5, 2) if act == "BUY" else round(curr_price - risk * 1.5, 2)
                tp2 = round(curr_price + risk * 2.6, 2) if act == "BUY" else round(curr_price - risk * 2.6, 2)
                res_lbl, pips, exit_idx = simulate_trade(df, i, act, curr_price, sl, tp1, tp2, max_bars=36)
                trades.append({
                    "time": t.strftime("%d/%m %H:%M"),
                    "idx": i,
                    "action": act,
                    "setup": setup,
                    "entry": round(curr_price, 2),
                    "result": res_lbl,
                    "pips": pips
                })
                i = exit_idx + 1
                continue
        i += 1
    return trades

def run_bot3_backtest(df):
    """Kiểm thử Bot #3: Order Flow (CVD/Delta) + Volume Profile (POC/VAH/VAL) + Data nến."""
    trades = []
    i = 60
    while i < len(df) - 2:
        t = df.index[i]
        if t.hour in [2, 3, 4, 5, 6]:
            i += 1
            continue
        window = df.iloc[i - 35 : i + 1]
        poc, vah, val = calc_vp_fast(window)
        c0, c1 = df.iloc[i], df.iloc[i - 1]
        curr_price = c0["Close"]
        ema21, ema50, vwap, cvd6 = c0["EMA21"], c0["EMA50"], c0["VWAP"], c0["CVD_6"]
        atr = float(np.clip(c0["ATR"], 5.0, 12.0))
        u_wick0 = c0["High"] - max(c0["Open"], c0["Close"])
        l_wick0 = min(c0["Open"], c0["Close"]) - c0["Low"]
        ema_slope = ema21 - df.iloc[i - 3]["EMA21"]

        is_down = (ema21 < ema50 - 1.5 and curr_price <= ema50 + 2.0) or (curr_price < ema21 and curr_price < vwap and cvd6 < -80000) or (ema_slope < -4.0)
        is_up = (ema21 > ema50 + 1.5 and curr_price >= ema50 - 2.0) or (curr_price > ema21 and curr_price > vwap and cvd6 > 80000) or (ema_slope > 4.0)
        is_climax = c0["Range"] > max(atr * 1.4, 11.5)

        act, setup = None, None
        if is_down and not is_up and not is_climax:
            touched_res = (c0["High"] >= ema21 - 2.0) or (c0["High"] >= vwap - 2.0) or (abs(c0["High"] - poc) <= 3.5)
            bear_ok = (c0["Close"] < c0["Open"] and c0["Delta"] < 0) or (u_wick0 >= c0["Range"] * 0.40)
            if touched_res and bear_ok:
                act, setup = "SELL", "Trend Pullback CVD"
            elif c1["Close"] >= val - 1.0 and curr_price < val - 1.5 and c0["Delta"] < -30000:
                act, setup = "SELL", "VAL Breakdown"
        elif is_up and not is_down and not is_climax:
            touched_sup = (c0["Low"] <= ema21 + 2.0) or (c0["Low"] <= vwap + 2.0) or (abs(c0["Low"] - poc) <= 3.5)
            bull_ok = (c0["Close"] > c0["Open"] and c0["Delta"] > 0) or (l_wick0 >= c0["Range"] * 0.40)
            if touched_sup and bull_ok:
                act, setup = "BUY", "Trend Pullback CVD"
            elif c1["Close"] <= vah + 1.0 and curr_price > vah + 1.5 and c0["Delta"] > 30000:
                act, setup = "BUY", "VAH Breakout"
        elif not is_climax:
            va_w = max(vah - val, 10.0)
            if abs(curr_price - poc) >= va_w * 0.30:
                if c0["Low"] <= val + 2.5 and curr_price > val + 0.5 and l_wick0 >= c0["Range"] * 0.35 and cvd6 > -60000:
                    act, setup = "BUY", "VAL Mean Reversion"
                elif c0["High"] >= vah - 2.5 and curr_price < vah - 0.5 and u_wick0 >= c0["Range"] * 0.35 and cvd6 < 60000:
                    act, setup = "SELL", "VAH Mean Reversion"

        if act:
            raw_dist = (curr_price - min(c0["Low"], c1["Low"]) + 2.5) if act == "BUY" else (max(c0["High"], c1["High"]) - curr_price + 2.5)
            if raw_dist <= 13.5:
                risk = float(np.clip(raw_dist, 8.5, 13.5))
                sl = round(curr_price - risk, 2) if act == "BUY" else round(curr_price + risk, 2)
                tp1 = round(curr_price + risk * 1.15, 2) if act == "BUY" else round(curr_price - risk * 1.15, 2)
                tp2 = round(curr_price + risk * 2.1, 2) if act == "BUY" else round(curr_price - risk * 2.1, 2)
                res_lbl, pips, exit_idx = simulate_trade(df, i, act, curr_price, sl, tp1, tp2, max_bars=18)
                trades.append({
                    "time": t.strftime("%d/%m %H:%M"),
                    "idx": i,
                    "action": act,
                    "setup": setup,
                    "entry": round(curr_price, 2),
                    "result": res_lbl,
                    "pips": pips
                })
                i = exit_idx + 1
                continue
        i += 1
    return trades

def summarize_backtest(trades, split_idx):
    def calc_metrics(t_list):
        if not t_list:
            return {"total": 0, "wins": 0, "losses": 0, "winrate": 0, "net_pips": 0, "profit_factor": 0, "max_dd": 0}
        wins = [t["pips"] for t in t_list if t["pips"] > 0]
        losses = [abs(t["pips"]) for t in t_list if t["pips"] < 0]
        gross_p = sum(wins)
        gross_l = sum(losses)
        pf = round(gross_p / gross_l, 2) if gross_l > 0 else (9.99 if gross_p > 0 else 0)
        eq, peak, max_dd = 0, 0, 0
        for t in t_list:
            eq += t["pips"]
            peak = max(peak, eq)
            max_dd = max(max_dd, peak - eq)
        return {
            "total": len(t_list),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": round(len(wins) / len(t_list) * 100, 1),
            "net_pips": sum(t["pips"] for t in t_list),
            "profit_factor": pf,
            "max_dd": max_dd
        }

    in_sample = [t for t in trades if t["idx"] < split_idx]
    out_sample = [t for t in trades if t["idx"] >= split_idx]

    # Đường cong vốn (Equity Curve)
    equity_curve = []
    cum = 0
    for t in trades:
        cum += t["pips"]
        equity_curve.append({"time": t["time"], "pips": cum})

    # Phân rã hiệu suất từng Setup (Playbook)
    setup_stats = {}
    for t in trades:
        s = t["setup"]
        if s not in setup_stats:
            setup_stats[s] = {"trades": 0, "wins": 0, "pips": 0}
        setup_stats[s]["trades"] += 1
        if t["pips"] > 0:
            setup_stats[s]["wins"] += 1
        setup_stats[s]["pips"] += t["pips"]

    playbook = []
    for k, v in setup_stats.items():
        wr = round(v["wins"] / v["trades"] * 100, 1) if v["trades"] > 0 else 0
        verdict = "GIỮ LẠI" if (v["pips"] > 0 and wr >= 45) else "CẦN TỐI ƯU / TẮT"
        playbook.append({"setup": k, "trades": v["trades"], "winrate": wr, "pips": v["pips"], "verdict": verdict})
    playbook.sort(key=lambda x: x["pips"], reverse=True)

    return {
        "overall": calc_metrics(trades),
        "in_sample": calc_metrics(in_sample),
        "out_sample": calc_metrics(out_sample),
        "equity_curve": equity_curve,
        "playbook": playbook,
        "recent_trades": trades[-15:][::-1]
    }

def load_json_safe(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"active_trades": [], "closed_trades": []}

def main():
    print("🚀 Bắt đầu chạy Agentic Backtest Engine (XAU/USD H1)...")
    df = fetch_historical_h1(1000)
    split_idx = int(len(df) * 0.70)

    b1_trades = run_bot1_backtest(df)
    b3_trades = run_bot3_backtest(df)

    b1_live = load_json_safe("trade_state.json")
    b3_live = load_json_safe("bot3_state.json")

    dashboard = {
        "updated_at": datetime.now(VN_TZ).strftime("%H:%M - %d/%m/%Y"),
        "candles_tested": len(df),
        "spread_pips": int(SPREAD_PTS * 10),
        "bot1": {
            "name": "Bot #1 — Price Action Swing (H1/H4)",
            "backtest": summarize_backtest(b1_trades, split_idx),
            "live_active": b1_live.get("active_trades", []),
            "live_closed": b1_live.get("closed_trades", [])[-15:][::-1]
        },
        "bot3": {
            "name": "Bot #3 — Order Flow + Volume Profile + Data Nến",
            "backtest": summarize_backtest(b3_trades, split_idx),
            "live_active": b3_live.get("active_trades", []),
            "live_closed": b3_live.get("closed_trades", [])[-15:][::-1]
        }
    }

    with open("dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)
    print("✅ Đã xuất dữ liệu dashboard_data.json thành công!")

if __name__ == "__main__":
    main()

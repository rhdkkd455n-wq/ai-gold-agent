import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "").strip()
VN_TZ = timezone(timedelta(hours=7))


def fetch_h1_data(bars=1000):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": "1h",
        "outputsize": bars,
        "apikey": API_KEY,
        "timezone": "Asia/Ho_Chi_Minh"
    }
    r = requests.get(url, params=params, timeout=30).json()
    if "values" not in r:
        raise RuntimeError(f"Lỗi lấy dữ liệu TwelveData: {r}")

    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)

    # Tính chỉ báo kỹ thuật & Proxy Order Flow
    df["Range"] = (df["High"] - df["Low"]).replace(0, 0.1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean().bfill()

    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Proxy Volume & Delta từ cấu trúc nến
    df["Vol"] = df["Range"] * 12000
    body_ratio = (df["Close"] - df["Open"]) / df["Range"]
    df["Delta"] = df["Vol"] * body_ratio
    df["CVD_6"] = df["Delta"].rolling(6).sum().fillna(0)

    # Anchored VWAP theo ngày
    df["Date"] = df.index.date
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["PV"] = tp * df["Vol"]
    df["VWAP"] = df.groupby("Date")["PV"].cumsum() / df.groupby("Date")["Vol"].cumsum()

    return df


def calc_vp_fast(window):
    low_min, high_max = window["Low"].min(), window["High"].max()
    if high_max <= low_min:
        mid = float(window["Close"].iloc[-1])
        return mid, mid + 2.0, mid - 2.0
    bins = np.linspace(low_min, high_max, 25)
    hist, edges = np.histogram(window["Close"], bins=bins, weights=window["Vol"])
    poc_idx = int(np.argmax(hist))
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)

    total_vol = hist.sum()
    target_vol = total_vol * 0.70
    acc_vol = hist[poc_idx]
    lo, hi = poc_idx, poc_idx
    while acc_vol < target_vol and (lo > 0 or hi < len(hist) - 1):
        v_lo = hist[lo - 1] if lo > 0 else -1
        v_hi = hist[hi + 1] if hi < len(hist) - 1 else -1
        if v_hi >= v_lo and hi < len(hist) - 1:
            hi += 1
            acc_vol += hist[hi]
        elif lo > 0:
            lo -= 1
            acc_vol += hist[lo]
        else:
            break
    vah = float(edges[hi + 1])
    val = float(edges[lo])
    return poc, vah, val


def simulate_trade(df, entry_idx, action, entry, sl, tp1, tp2, max_bars=36, spread=0.30):
    """Mô phỏng khớp lệnh có trừ Spread (3 pips = 0.30 giá vàng), quản lý TP1 dời BE."""
    hit_tp1 = False
    curr_sl = sl
    realized_pips = 0.0

    for j in range(entry_idx + 1, min(entry_idx + 1 + max_bars, len(df))):
        bar = df.iloc[j]
        if action == "BUY":
            if bar["Low"] <= curr_sl:
                if hit_tp1:
                    return "TP1 + BE", round(realized_pips - spread * 10, 1), j
                loss = (entry - curr_sl + spread) * 10
                return "SL", round(-loss, 1), j
            if not hit_tp1 and bar["High"] >= tp1:
                hit_tp1 = True
                realized_pips += (tp1 - entry) * 10 * 0.5
                curr_sl = entry + 0.20
            if bar["High"] >= tp2:
                rem = (tp2 - entry) * 10 * (0.5 if hit_tp1 else 1.0)
                return "FULL TP2", round(realized_pips + rem - spread * 10, 1), j
        else:
            if bar["High"] >= curr_sl:
                if hit_tp1:
                    return "TP1 + BE", round(realized_pips - spread * 10, 1), j
                loss = (curr_sl - entry + spread) * 10
                return "SL", round(-loss, 1), j
            if not hit_tp1 and bar["Low"] <= tp1:
                hit_tp1 = True
                realized_pips += (entry - tp1) * 10 * 0.5
                curr_sl = entry - 0.20
            if bar["Low"] <= tp2:
                rem = (entry - tp2) * 10 * (0.5 if hit_tp1 else 1.0)
                return "FULL TP2", round(realized_pips + rem - spread * 10, 1), j

    exit_bar = df.iloc[min(entry_idx + max_bars, len(df) - 1)]
    close_p = exit_bar["Close"]
    if action == "BUY":
        rem = (close_p - entry) * 10 * (0.5 if hit_tp1 else 1.0)
    else:
        rem = (entry - close_p) * 10 * (0.5 if hit_tp1 else 1.0)
    total = round(realized_pips + rem - spread * 10, 1)
    return ("TP1 + BE" if hit_tp1 else "TIME EXIT"), total, min(entry_idx + max_bars, len(df) - 1)


def run_bot1_backtest(df):
    """Bot #1 Tối Ưu: Bỏ Fakey, siết Pin Bar (râu >= 55% + đóng cửa thuận chiều), giữ Trend Bar & Engulfing."""
    trades = []
    i = 60
    while i < len(df) - 2:
        t = df.index[i]
        if t.hour in [2, 3, 4, 5]:
            i += 1
            continue
        c0, c1 = df.iloc[i], df.iloc[i - 1]
        curr_price = c0["Close"]
        atr = float(np.clip(c0["ATR"], 5.5, 12.0))
        r0 = max(c0["Range"], 0.1)
        body0 = abs(c0["Close"] - c0["Open"])
        u_wick0 = c0["High"] - max(c0["Open"], c0["Close"])
        l_wick0 = min(c0["Open"], c0["Close"]) - c0["Low"]
        body1 = abs(c1["Close"] - c1["Open"])

        uptrend = (c0["Close"] >= c0["EMA21"] - 2.0) and (c0["EMA21"] >= c0["EMA50"] - 3.0)
        downtrend = (c0["Close"] <= c0["EMA21"] + 2.0) and (c0["EMA21"] <= c0["EMA50"] + 3.0)

        bull_pin = (l_wick0 >= r0 * 0.55) and (u_wick0 <= r0 * 0.20) and (c0["Close"] > c0["Open"]) and (r0 >= 5.5)
        bear_pin = (u_wick0 >= r0 * 0.55) and (l_wick0 <= r0 * 0.20) and (c0["Close"] < c0["Open"]) and (r0 >= 5.5)

        bull_eng = (c1["Close"] < c1["Open"]) and (c0["Close"] > c0["Open"]) and (body0 >= body1 * 1.05) and (body0 >= r0 * 0.45)
        bear_eng = (c1["Close"] > c1["Open"]) and (c0["Close"] < c0["Open"]) and (body0 >= body1 * 1.05) and (body0 >= r0 * 0.45)
        bull_tb = uptrend and (c0["Close"] > c0["Open"]) and (body0 >= r0 * 0.60) and (body0 >= atr * 0.65)
        bear_tb = downtrend and (c0["Close"] < c0["Open"]) and (body0 >= r0 * 0.60) and (body0 >= atr * 0.65)

        act, setup = None, None
        if uptrend and (c0["Low"] <= max(c0["EMA10"], c0["EMA21"]) + 3.0):
            if bull_tb:
                act, setup = "BUY", "Trend Bar"
            elif bull_eng:
                act, setup = "BUY", "Engulfing"
            elif bull_pin:
                act, setup = "BUY", "Pin Bar (Strict)"
        elif downtrend and (c0["High"] >= min(c0["EMA10"], c0["EMA21"]) - 3.0):
            if bear_tb:
                act, setup = "SELL", "Trend Bar"
            elif bear_eng:
                act, setup = "SELL", "Engulfing"
            elif bear_pin:
                act, setup = "SELL", "Pin Bar (Strict)"

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
    """Bot #3 Tối Ưu: Tập trung 100% Trend Pullback CVD + VAL Breakdown / VAH Breakout."""
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


def summarize_trades(trades, total_bars):
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "winrate": 0,
            "total_pips": 0, "profit_factor": 0, "max_dd": 0,
            "equity_curve": [0], "is_oos": {}, "by_setup": [], "recent_trades": []
        }

    wins = sum(1 for t in trades if t["pips"] > 0)
    losses = len(trades) - wins
    gross_win = sum(t["pips"] for t in trades if t["pips"] > 0)
    gross_loss = abs(sum(t["pips"] for t in trades if t["pips"] <= 0))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 9.99

    eq = [0.0]
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cur = round(eq[-1] + t["pips"], 1)
        eq.append(cur)
        if cur > peak:
            peak = cur
        dd = cur - peak
        if dd < max_dd:
            max_dd = dd

    split_idx = int(total_bars * 0.70)
    is_tr = [t for t in trades if t["idx"] < split_idx]
    oos_tr = [t for t in trades if t["idx"] >= split_idx]

    def calc_sub(sub):
        if not sub:
            return {"trades": 0, "winrate": 0, "pf": 0, "pips": 0}
        w = sum(1 for x in sub if x["pips"] > 0)
        gw = sum(x["pips"] for x in sub if x["pips"] > 0)
        gl = abs(sum(x["pips"] for x in sub if x["pips"] <= 0))
        return {
            "trades": len(sub),
            "winrate": round(w / len(sub) * 100, 1),
            "pf": round(gw / gl, 2) if gl > 0 else 9.99,
            "pips": round(sum(x["pips"] for x in sub), 1)
        }

    setups = {}
    for t in trades:
        s = t["setup"]
        setups.setdefault(s, []).append(t["pips"])

    by_setup = []
    for s, plist in setups.items():
        w = sum(1 for p in plist if p > 0)
        tot = round(sum(plist), 1)
        wr = round(w / len(plist) * 100, 1)
        by_setup.append({
            "setup": s,
            "trades": len(plist),
            "winrate": wr,
            "pips": tot,
            "verdict": "GIỮ LẠI" if tot > 0 and wr >= 45 else "CẦN TỐI ƯU / TẮT"
        })
    by_setup.sort(key=lambda x: x["pips"], reverse=True)

    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "winrate": round(wins / len(trades) * 100, 1),
        "total_pips": round(sum(t["pips"] for t in trades), 1),
        "profit_factor": pf,
        "max_dd": round(max_dd, 1),
        "equity_curve": eq,
        "is_oos": {"in_sample": calc_sub(is_tr), "out_of_sample": calc_sub(oos_tr)},
        "by_setup": by_setup,
        "recent_trades": list(reversed(trades[-15:]))
    }


def load_live_state(path):
    if not os.path.exists(path):
        return {"active_trades": [], "closed_trades": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Lọc trùng mã vé tự động để sổ Live luôn sạch
        for key in ["active_trades", "closed_trades"]:
            seen = set()
            clean = []
            for item in data.get(key, []):
                tid = item.get("id")
                if tid and tid not in seen:
                    seen.add(tid)
                    clean.append(item)
            data[key] = clean
        return data
    except Exception:
        return {"active_trades": [], "closed_trades": []}


def main():
    df = fetch_h1_data(1000)
    b1_trades = run_bot1_backtest(df)
    b3_trades = run_bot3_backtest(df)

    output = {
        "updated_at": datetime.now(VN_TZ).strftime("%H:%M - %d/%m/%Y"),
        "bars_analyzed": len(df),
        "spread_pips": 3.0,
        "bot1": {
            "name": "Bot #1 (Price Action)",
            "backtest": summarize_trades(b1_trades, len(df)),
            "live": load_live_state("bot_state.json")
        },
        "bot3": {
            "name": "Bot #3 (Order Flow + VP)",
            "backtest": summarize_trades(b3_trades, len(df)),
            "live": load_live_state("bot3_state.json")
        }
    }

    with open("dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("Đã cập nhật dashboard_data.json thành công!")


if __name__ == "__main__":
    main()

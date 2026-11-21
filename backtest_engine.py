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
        raise RuntimeError(f"TwelveData API error: {r}")

    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").set_index("datetime")
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)

    # Chỉ báo H1
    df["Range"] = (df["High"] - df["Low"]).replace(0, 0.1)
    prev_c = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_c).abs(),
        (df["Low"] - prev_c).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean().bfill()

    # ADX(14) chuẩn Wilder để đo Sideway vs Trend
    up_move = df["High"] - df["High"].shift(1)
    down_move = df["Low"].shift(1) - df["Low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    sm_tr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, 0.1)
    sm_pdm = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean()
    sm_mdm = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100.0 * sm_pdm / sm_tr
    mdi = 100.0 * sm_mdm / sm_tr
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1.0)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(20.0)

    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # Cấu trúc Swing 24 nến (Key Level)
    df["SwingHigh24"] = df["High"].shift(1).rolling(24).max().bfill()
    df["SwingLow24"] = df["Low"].shift(1).rolling(24).min().bfill()

    # Khung H4 không lookahead (dùng nến H4 đã đóng gần nhất)
    df_h4 = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    df_h4["H4_EMA21"] = df_h4["Close"].ewm(span=21, adjust=False).mean()
    df_h4["H4_EMA50"] = df_h4["Close"].ewm(span=50, adjust=False).mean()
    df_h4["H4_Trend"] = np.where(
        (df_h4["Close"] > df_h4["H4_EMA21"]) & (df_h4["H4_EMA21"] >= df_h4["H4_EMA50"]), 1,
        np.where((df_h4["Close"] < df_h4["H4_EMA21"]) & (df_h4["H4_EMA21"] <= df_h4["H4_EMA50"]), -1, 0)
    )
    h4_shifted = df_h4[["H4_Trend"]].shift(1)
    df = df.join(h4_shifted, how="left")
    df["H4_Trend"] = df["H4_Trend"].ffill().fillna(0).astype(int)

    # Proxy Order Flow & Anchored VWAP
    df["Vol"] = df["Range"] * 12000.0
    df["Delta"] = df["Vol"] * ((df["Close"] - df["Open"]) / df["Range"])
    df["CVD_6"] = df["Delta"].rolling(6).sum().fillna(0.0)
    df["Date"] = df.index.date
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    df["PV"] = tp * df["Vol"]
    df["VWAP"] = df.groupby("Date")["PV"].cumsum() / df.groupby("Date")["Vol"].cumsum()

    return df


def precompute_vp(df):
    n = len(df)
    pocs = np.zeros(n)
    vahs = np.zeros(n)
    vals = np.zeros(n)
    closes = df["Close"].values
    lows = df["Low"].values
    highs = df["High"].values
    vols = df["Vol"].values

    for i in range(60, n):
        w_l = lows[i - 35 : i + 1]
        w_h = highs[i - 35 : i + 1]
        w_c = closes[i - 35 : i + 1]
        w_v = vols[i - 35 : i + 1]
        lo_min, hi_max = float(w_l.min()), float(w_h.max())
        if hi_max <= lo_min:
            mid = float(w_c[-1])
            pocs[i], vahs[i], vals[i] = mid, mid + 2.0, mid - 2.0
            continue
        bins = np.linspace(lo_min, hi_max, 25)
        hist, edges = np.histogram(w_c, bins=bins, weights=w_v)
        poc_idx = int(np.argmax(hist))
        pocs[i] = float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)
        target = hist.sum() * 0.70
        acc = hist[poc_idx]
        lo, hi = poc_idx, poc_idx
        while acc < target and (lo > 0 or hi < len(hist) - 1):
            v_lo = hist[lo - 1] if lo > 0 else -1.0
            v_hi = hist[hi + 1] if hi < len(hist) - 1 else -1.0
            if v_hi >= v_lo and hi < len(hist) - 1:
                hi += 1
                acc += hist[hi]
            elif lo > 0:
                lo -= 1
                acc += hist[lo]
            else:
                break
        vahs[i] = float(edges[hi + 1])
        vals[i] = float(edges[lo])
    return pocs, vahs, vals


def get_session(hour):
    if 6 <= hour < 14:
        return "Á"
    if 14 <= hour < 19:
        return "Âu"
    if 19 <= hour <= 23:
        return "Âu-Mỹ"
    return "Mỹ muộn"


def simulate_trade(df, entry_idx, action, entry, sl, tp1, tp2, max_bars=36, spread=0.30, be_mode="early_be"):
    """Mô phỏng khớp lệnh quy đổi R + phát hiện lỗi quét râu (stop_hunted)."""
    risk = max(abs(entry - sl), 0.1)
    hit_tp1 = False
    curr_sl = sl
    realized_r = 0.0
    spread_r = spread / risk
    n = len(df)
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    exit_idx = min(entry_idx + max_bars, n - 1)
    res_lbl = "TIME EXIT"
    total_r = 0.0

    for j in range(entry_idx + 1, min(entry_idx + 1 + max_bars, n)):
        h, l = highs[j], lows[j]
        if action == "BUY":
            if l <= curr_sl:
                exit_idx = j
                if hit_tp1:
                    res_lbl, total_r = "TP1 + BE", round(realized_r - spread_r, 2)
                else:
                    res_lbl, total_r = "SL", round(-1.0 - spread_r, 2)
                break
            if be_mode != "no_be" and not hit_tp1 and h >= tp1:
                hit_tp1 = True
                realized_r += ((tp1 - entry) / risk) * 0.5
                curr_sl = entry + 0.20
            if h >= tp2:
                exit_idx = j
                rem_r = ((tp2 - entry) / risk) * (0.5 if hit_tp1 else 1.0)
                res_lbl, total_r = "FULL TP2", round(realized_r + rem_r - spread_r, 2)
                break
        else:
            if h >= curr_sl:
                exit_idx = j
                if hit_tp1:
                    res_lbl, total_r = "TP1 + BE", round(realized_r - spread_r, 2)
                else:
                    res_lbl, total_r = "SL", round(-1.0 - spread_r, 2)
                break
            if be_mode != "no_be" and not hit_tp1 and l <= tp1:
                hit_tp1 = True
                realized_r += ((entry - tp1) / risk) * 0.5
                curr_sl = entry - 0.20
            if l <= tp2:
                exit_idx = j
                rem_r = ((entry - tp2) / risk) * (0.5 if hit_tp1 else 1.0)
                res_lbl, total_r = "FULL TP2", round(realized_r + rem_r - spread_r, 2)
                break
    else:
        close_p = closes[exit_idx]
        rem_r = (((close_p - entry) if action == "BUY" else (entry - close_p)) / risk) * (0.5 if hit_tp1 else 1.0)
        total_r = round(realized_r + rem_r - spread_r, 2)
        res_lbl = "TP1 + BE" if hit_tp1 else "TIME EXIT"

    # Kiểm tra có bị quét râu SL rồi chạy đúng hướng chạm TP1 trong 6 nến sau không
    stop_hunted = False
    if res_lbl == "SL":
        fwd_end = min(exit_idx + 7, n)
        if fwd_end > exit_idx + 1:
            if action == "BUY" and highs[exit_idx + 1 : fwd_end].max() >= tp1:
                stop_hunted = True
            elif action == "SELL" and lows[exit_idx + 1 : fwd_end].min() <= tp1:
                stop_hunted = True

    pips = round(total_r * risk * 10.0, 1)
    return res_lbl, total_r, pips, exit_idx, stop_hunted


def evaluate_factors(df, i, act, poc, vah, val):
    c0 = df.iloc[i]
    h4_tr = int(c0["H4_Trend"])
    adx = float(c0["ADX"])
    atr = float(c0["ATR"])
    sess = get_session(df.index[i].hour)
    tol = max(atr * 0.55, 4.0)

    h4_ok = (act == "BUY" and h4_tr >= 0) or (act == "SELL" and h4_tr <= 0)
    h4_strict = (act == "BUY" and h4_tr == 1) or (act == "SELL" and h4_tr == -1)
    trend_ok = adx >= 20.0
    near_swing = (abs(c0["Low"] - c0["SwingLow24"]) <= tol) if act == "BUY" else (abs(c0["High"] - c0["SwingHigh24"]) <= tol)
    near_vp = (min(abs(c0["Close"] - poc), abs(c0["Close"] - vah), abs(c0["Close"] - val)) <= tol)
    level_ok = bool(near_swing or near_vp)
    sess_ok = sess in ["Âu", "Âu-Mỹ"]

    score = int(h4_strict) + int(trend_ok) + int(level_ok) + int(sess_ok)
    return {
        "h4_ok": bool(h4_strict),
        "h4_dir": "TĂNG" if h4_tr == 1 else ("GIẢM" if h4_tr == -1 else "SIDEWAY"),
        "adx": round(adx, 1),
        "trend_ok": bool(trend_ok),
        "level_ok": bool(level_ok),
        "session": sess,
        "sess_ok": bool(sess_ok),
        "score": score
    }


def diagnose_loss(res_lbl, rr, factors, stop_hunted):
    if rr > 0:
        if res_lbl == "TP1 + BE":
            return "Chốt sớm TP1 + Quét BE"
        return "Đạt chuẩn hội tụ" if factors["score"] >= 3 else "Thắng nhờ động lượng"
    if stop_hunted:
        return "SL quá sát (Quét râu rồi chạy đúng)"
    if not factors["h4_ok"]:
        return "Ngược xu hướng H4"
    if not factors["trend_ok"]:
        return "Vùng Sideway nhiễu (ADX < 20)"
    if not factors["level_ok"]:
        return "Nến lơ lửng (Thiếu Key Level)"
    if not factors["sess_ok"]:
        return f"Phiên thanh khoản kém ({factors['session']})"
    return "Nhiễu thị trường (Chi phí hệ thống)"


def extract_candle_window(df, entry_idx, exit_idx):
    start = max(0, entry_idx - 16)
    end = min(len(df), max(exit_idx + 5, entry_idx + 14))
    sub = df.iloc[start:end]
    bars = []
    for ts, row in sub.iterrows():
        bars.append({
            "t": ts.strftime("%d/%m %H:%M"),
            "o": round(float(row["Open"]), 2),
            "h": round(float(row["High"]), 2),
            "l": round(float(row["Low"]), 2),
            "c": round(float(row["Close"]), 2),
            "ema21": round(float(row["EMA21"]), 2),
            "vwap": round(float(row["VWAP"]), 2)
        })
    return {
        "entry_rel": entry_idx - start,
        "exit_rel": min(exit_idx - start, len(bars) - 1),
        "bars": bars
    }


def run_strategy(df, vp_cache, bot_type="bot1", h4_filter=False, adx_min=0, be_mode="early_be", sl_mult=1.0, collect_candles=False):
    pocs, vahs, vals = vp_cache
    trades = []
    i = 60
    n = len(df)
    while i < n - 2:
        t = df.index[i]
        if t.hour in [2, 3, 4, 5]:
            i += 1
            continue
        c0, c1 = df.iloc[i], df.iloc[i - 1]
        curr_price = float(c0["Close"])
        adx = float(c0["ADX"])
        h4_tr = int(c0["H4_Trend"])

        if adx_min > 0 and adx < adx_min:
            i += 1
            continue

        act, setup = None, None

        if bot_type == "bot1":
            atr = float(np.clip(c0["ATR"], 5.5, 12.0))
            r0 = max(float(c0["Range"]), 0.1)
            body0 = abs(float(c0["Close"] - c0["Open"]))
            u_wick0 = float(c0["High"] - max(c0["Open"], c0["Close"]))
            l_wick0 = float(min(c0["Open"], c0["Close"]) - c0["Low"])
            body1 = abs(float(c1["Close"] - c1["Open"]))

            uptrend = (c0["Close"] >= c0["EMA21"] - 2.0) and (c0["EMA21"] >= c0["EMA50"] - 3.0)
            downtrend = (c0["Close"] <= c0["EMA21"] + 2.0) and (c0["EMA21"] <= c0["EMA50"] + 3.0)

            bull_pin = (l_wick0 >= r0 * 0.55) and (u_wick0 <= r0 * 0.20) and (c0["Close"] > c0["Open"]) and (r0 >= 5.5)
            bear_pin = (u_wick0 >= r0 * 0.55) and (l_wick0 <= r0 * 0.20) and (c0["Close"] < c0["Open"]) and (r0 >= 5.5)
            bull_eng = (c1["Close"] < c1["Open"]) and (c0["Close"] > c0["Open"]) and (body0 >= body1 * 1.05) and (body0 >= r0 * 0.45)
            bear_eng = (c1["Close"] > c1["Open"]) and (c0["Close"] < c0["Open"]) and (body0 >= body1 * 1.05) and (body0 >= r0 * 0.45)
            bull_tb = uptrend and (c0["Close"] > c0["Open"]) and (body0 >= r0 * 0.60) and (body0 >= atr * 0.65)
            bear_tb = downtrend and (c0["Close"] < c0["Open"]) and (body0 >= r0 * 0.60) and (body0 >= atr * 0.65)

            if uptrend and (c0["Low"] <= max(c0["EMA10"], c0["EMA21"]) + 3.0):
                if bull_tb: act, setup = "BUY", "Trend Bar"
                elif bull_eng: act, setup = "BUY", "Engulfing"
                elif bull_pin: act, setup = "BUY", "Pin Bar"
            elif downtrend and (c0["High"] >= min(c0["EMA10"], c0["EMA21"]) - 3.0):
                if bear_tb: act, setup = "SELL", "Trend Bar"
                elif bear_eng: act, setup = "SELL", "Engulfing"
                elif bear_pin: act, setup = "SELL", "Pin Bar"
        else:
            if t.hour == 6:
                i += 1
                continue
            poc, vah, val = pocs[i], vahs[i], vals[i]
            ema21, ema50, vwap, cvd6 = float(c0["EMA21"]), float(c0["EMA50"]), float(c0["VWAP"]), float(c0["CVD_6"])
            atr = float(np.clip(c0["ATR"], 5.0, 12.0))
            u_wick0 = float(c0["High"] - max(c0["Open"], c0["Close"]))
            l_wick0 = float(min(c0["Open"], c0["Close"]) - c0["Low"])
            ema_slope = ema21 - float(df.iloc[i - 3]["EMA21"])

            is_down = (ema21 < ema50 - 1.5 and curr_price <= ema50 + 2.0) or (curr_price < ema21 and curr_price < vwap and cvd6 < -80000) or (ema_slope < -4.0)
            is_up = (ema21 > ema50 + 1.5 and curr_price >= ema50 - 2.0) or (curr_price > ema21 and curr_price > vwap and cvd6 > 80000) or (ema_slope > 4.0)
            is_climax = float(c0["Range"]) > max(atr * 1.4, 11.5)

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
            if h4_filter:
                if (act == "BUY" and h4_tr != 1) or (act == "SELL" and h4_tr != -1):
                    i += 1
                    continue

            pad = 2.0 if bot_type == "bot1" else 2.5
            max_r = 15.0 if bot_type == "bot1" else 13.5
            raw_dist = (curr_price - min(float(c0["Low"]), float(c1["Low"])) + pad) if act == "BUY" else (max(float(c0["High"]), float(c1["High"])) - curr_price + pad)
            if raw_dist <= max_r * sl_mult:
                risk = float(np.clip(raw_dist * sl_mult, 8.5, max_r * sl_mult))
                sl = round(curr_price - risk, 2) if act == "BUY" else round(curr_price + risk, 2)

                if bot_type == "bot1":
                    tp1_m = 1.8 if be_mode == "wide_tp" else 1.5
                    tp2_m = 2.8 if be_mode == "wide_tp" else 2.6
                    mb = 36
                else:
                    tp1_m = 1.5 if be_mode == "wide_tp" else 1.15
                    tp2_m = 2.4 if be_mode == "wide_tp" else 2.1
                    mb = 18

                tp1 = round(curr_price + risk * tp1_m, 2) if act == "BUY" else round(curr_price - risk * tp1_m, 2)
                tp2 = round(curr_price + risk * tp2_m, 2) if act == "BUY" else round(curr_price - risk * tp2_m, 2)

                res_lbl, r_val, pips_val, exit_idx, stop_hunted = simulate_trade(
                    df, i, act, curr_price, sl, tp1, tp2, max_bars=mb, be_mode=be_mode
                )

                trade_obj = {
                    "id": f"{bot_type.upper()}_{len(trades)+1:02d}",
                    "time": t.strftime("%d/%m %H:%M"),
                    "weekday": ["T2", "T3", "T4", "T5", "T6", "T7", "CN"][t.weekday()],
                    "idx": i,
                    "exit_idx": exit_idx,
                    "action": act,
                    "setup": setup,
                    "entry": round(curr_price, 2),
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "result": res_lbl,
                    "rr": r_val,
                    "pips": pips_val,
                    "stop_hunted": stop_hunted
                }
                if collect_candles:
                    factors = evaluate_factors(df, i, act, pocs[i], vahs[i], vals[i])
                    trade_obj["factors"] = factors
                    trade_obj["diagnosis"] = diagnose_loss(res_lbl, r_val, factors, stop_hunted)
                trades.append(trade_obj)
                i = exit_idx + 1
                continue
        i += 1
    return trades


def compute_metrics(trades, total_bars):
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "winrate": 0.0,
            "total_rr": 0.0, "total_pips": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "avg_rr": 0.0,
            "max_dd_rr": 0.0, "max_consec_loss": 0, "recovery": 0.0,
            "equity_curve": [0.0], "dd_curve": [0.0],
            "is_oos": {
                "in_sample": {"trades": 0, "winrate": 0, "pf": 0, "rr": 0},
                "out_of_sample": {"trades": 0, "winrate": 0, "pf": 0, "rr": 0}
            }
        }

    wins = [t["rr"] for t in trades if t["rr"] > 0]
    losses = [t["rr"] for t in trades if t["rr"] <= 0]
    gw = sum(wins)
    gl = abs(sum(losses))
    pf = round(gw / gl, 2) if gl > 0 else 9.99
    total_rr = round(sum(t["rr"] for t in trades), 2)
    total_pips = round(sum(t["pips"] for t in trades), 1)
    avg_win = round(gw / len(wins), 2) if wins else 0.0
    avg_loss = round(gl / len(losses), 2) if losses else 0.0
    avg_rr = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0
    expectancy = round(total_rr / len(trades), 2)

    eq = [0.0]
    dd_curve = [0.0]
    peak = 0.0
    max_dd = 0.0
    consec, max_consec = 0, 0
    for t in trades:
        cur = round(eq[-1] + t["rr"], 2)
        eq.append(cur)
        if cur > peak:
            peak = cur
        dd = round(cur - peak, 2)
        dd_curve.append(dd)
        if dd < max_dd:
            max_dd = dd
        if t["rr"] <= 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    recovery = round(total_rr / abs(max_dd), 2) if max_dd < 0 else 9.99

    split_idx = int(total_bars * 0.70)
    is_tr = [t for t in trades if t["idx"] < split_idx]
    oos_tr = [t for t in trades if t["idx"] >= split_idx]

    def calc_sub(sub):
        if not sub:
            return {"trades": 0, "winrate": 0.0, "pf": 0.0, "rr": 0.0}
        w = sum(1 for x in sub if x["rr"] > 0)
        g_w = sum(x["rr"] for x in sub if x["rr"] > 0)
        g_l = abs(sum(x["rr"] for x in sub if x["rr"] <= 0))
        return {
            "trades": len(sub),
            "winrate": round(w / len(sub) * 100, 1),
            "pf": round(g_w / g_l, 2) if g_l > 0 else 9.99,
            "rr": round(sum(x["rr"] for x in sub), 2)
        }

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "winrate": round(len(wins) / len(trades) * 100, 1),
        "total_rr": total_rr,
        "total_pips": total_pips,
        "profit_factor": pf,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_rr": avg_rr,
        "max_dd_rr": round(max_dd, 2),
        "max_consec_loss": max_consec,
        "recovery": recovery,
        "equity_curve": eq,
        "dd_curve": dd_curve,
        "is_oos": {"in_sample": calc_sub(is_tr), "out_of_sample": calc_sub(oos_tr)}
    }


def build_diagnostics(trades):
    def agg(sub):
        if not sub:
            return {"trades": 0, "winrate": 0.0, "rr": 0.0, "exp": 0.0}
        w = sum(1 for t in sub if t["rr"] > 0)
        tot = round(sum(t["rr"] for t in sub), 2)
        return {
            "trades": len(sub),
            "winrate": round(w / len(sub) * 100, 1),
            "rr": tot,
            "exp": round(tot / len(sub), 2)
        }

    # 1. So sánh Đạt vs Thiếu cho 4 yếu tố cốt lõi
    factor_rows = []
    checks = [
        ("Xu hướng H4 thuận chiều", "h4_ok"),
        ("Thị trường có Trend (ADX ≥ 20)", "trend_ok"),
        ("Vị trí Key Level / Volume Profile", "level_ok"),
        ("Phiên thanh khoản tốt (Âu / Âu-Mỹ)", "sess_ok")
    ]
    for label, key in checks:
        passed = [t for t in trades if t["factors"][key]]
        failed = [t for t in trades if not t["factors"][key]]
        factor_rows.append({"factor": label, "pass": agg(passed), "fail": agg(failed)})

    # 2. Điểm hội tụ (4/4 vs 3/4 vs <=2/4)
    score_rows = [
        {"bucket": "Đủ 4/4 yếu tố (A+)", **agg([t for t in trades if t["factors"]["score"] == 4])},
        {"bucket": "Đạt 3/4 yếu tố (B)", **agg([t for t in trades if t["factors"]["score"] == 3])},
        {"bucket": "Thiếu yếu tố (≤ 2/4)", **agg([t for t in trades if t["factors"]["score"] <= 2])}
    ]

    # 3. Top nguyên nhân thua lỗ
    losses = [t for t in trades if t["rr"] <= 0]
    cause_map = {}
    for t in losses:
        d = t["diagnosis"]
        cause_map.setdefault(d, []).append(t["rr"])
    loss_causes = [
        {"cause": k, "count": len(v), "rr": round(sum(v), 2), "pct": round(len(v) / max(len(losses), 1) * 100, 1)}
        for k, v in cause_map.items()
    ]
    loss_causes.sort(key=lambda x: x["rr"])

    # 4. Theo Phiên & Thứ
    by_session = [{"name": s, **agg([t for t in trades if t["factors"]["session"] == s])} for s in ["Á", "Âu", "Âu-Mỹ", "Mỹ muộn"]]
    by_weekday = [{"name": d, **agg([t for t in trades if t["weekday"] == d])} for d in ["T2", "T3", "T4", "T5", "T6"]]

    # 5. Theo Setup
    setups = {}
    for t in trades:
        setups.setdefault(t["setup"], []).append(t)
    by_setup = []
    for s, slist in setups.items():
        st = agg(slist)
        by_setup.append({
            "setup": s, **st,
            "verdict": "GIỮ LẠI" if st["rr"] > 0 and st["winrate"] >= 45 else "CẦN TỐI ƯU"
        })
    by_setup.sort(key=lambda x: x["rr"], reverse=True)

    return {
        "factor_audit": factor_rows,
        "by_score": score_rows,
        "loss_causes": loss_causes,
        "by_session": by_session,
        "by_weekday": by_weekday,
        "by_setup": by_setup
    }


def sample_timeline_curve(trades, total_bars, points=45):
    checkpoints = np.linspace(60, total_bars - 1, points).astype(int)
    curve = []
    cum = 0.0
    t_idx = 0
    for cp in checkpoints:
        while t_idx < len(trades) and trades[t_idx]["exit_idx"] <= cp:
            cum = round(cum + trades[t_idx]["rr"], 2)
            t_idx += 1
        curve.append(cum)
    return curve


def run_lab_sweep(df, vp_cache, bot_type):
    variants = []
    vid = 1
    total_bars = len(df)
    be_labels = {"early_be": "BE sớm", "no_be": "Không BE", "wide_tp": "TP rộng"}

    for h4_f in [False, True]:
        for adx_m in [0, 18, 22]:
            for be_m in ["early_be", "no_be", "wide_tp"]:
                for sl_m in [1.0, 1.3]:
                    tr = run_strategy(df, vp_cache, bot_type=bot_type, h4_filter=h4_f, adx_min=adx_m, be_mode=be_m, sl_mult=sl_m, collect_candles=False)
                    m = compute_metrics(tr, total_bars)
                    is_m = m["is_oos"]["in_sample"]
                    oos_m = m["is_oos"]["out_of_sample"]
                    is_baseline = (not h4_f and adx_m == 0 and be_m == "early_be" and sl_m == 1.0)
                    label = (
                        "BASELINE (Hiện tại)" if is_baseline else
                        f"{'H4+' if h4_f else 'NoH4'} | ADX≥{adx_m} | {be_labels[be_m]} | SLx{sl_m}"
                    )
                    variants.append({
                        "id": f"V{vid:02d}",
                        "label": label,
                        "is_baseline": is_baseline,
                        "params": {"h4": h4_f, "adx": adx_m, "be": be_m, "sl_mult": sl_m},
                        "trades": m["total_trades"],
                        "winrate": m["winrate"],
                        "pf": m["profit_factor"],
                        "total_rr": m["total_rr"],
                        "expectancy": m["expectancy"],
                        "max_dd_rr": m["max_dd_rr"],
                        "is_rr": is_m["rr"],
                        "oos_rr": oos_m["rr"],
                        "is_pf": is_m["pf"],
                        "oos_pf": oos_m["pf"],
                        "curve": sample_timeline_curve(tr, total_bars, points=45)
                    })
                    vid += 1

    # Xếp hạng theo chất lượng OOS + PF + ít Drawdown (yêu cầu tối thiểu 8 lệnh)
    def score_v(v):
        if v["trades"] < 8:
            return -999.0
        oos_bonus = 2.0 if v["oos_rr"] > 0 else -3.0
        is_bonus = 1.5 if v["is_rr"] > 0 else -2.0
        return v["total_rr"] + v["oos_rr"] * 1.5 - abs(v["max_dd_rr"]) * 0.6 + oos_bonus + is_bonus

    variants.sort(key=score_v, reverse=True)
    return variants


def load_live_state(paths):
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out = {"source_file": path, "active_trades": [], "closed_trades": []}
                for key in ["active_trades", "closed_trades"]:
                    seen = set()
                    clean = []
                    for idx, item in enumerate(data.get(key, [])):
                        tid = item.get("id") or item.get("ticket") or f"row_{idx}"
                        if tid in seen:
                            continue
                        seen.add(tid)
                        row = dict(item)
                        row["id"] = tid
                        if "rr" not in row:
                            pips = float(row.get("pips", 0) or 0)
                            sl_pips = float(row.get("sl_pips", 0) or 0)
                            if sl_pips >= 10:
                                row["rr"] = round(pips / sl_pips, 2)
                            else:
                                entry = float(row.get("entry", 0) or 0)
                                sl = float(row.get("sl", 0) or 0)
                                rp = abs(entry - sl) * 10.0
                                row["rr"] = round(pips / rp, 2) if rp >= 10 else round(pips / 120.0, 2)
                        clean.append(row)
                    out[key] = clean
                return out
            except Exception:
                pass
    return {"source_file": "none", "active_trades": [], "closed_trades": []}


def build_proposals(b1_base, b1_lab, b3_base, b3_lab):
    props = []
    for name, base, lab in [("Bot #1 (Price Action)", b1_base, b1_lab), ("Bot #3 (Order Flow)", b3_base, b3_lab)]:
        best = next((v for v in lab if not v["is_baseline"] and v["trades"] >= 8), None)
        if best and (best["total_rr"] > base["total_rr"] or abs(best["max_dd_rr"]) < abs(base["max_dd_rr"])):
            props.append({
                "id": f"PROP_{len(props)+1:02d}",
                "bot": name,
                "title": f"Áp dụng cấu hình {best['id']}: {best['label']}",
                "impact": f"Lợi nhuận: {base['total_rr']:+.2f}R → {best['total_rr']:+.2f}R | Max DD: {base['max_dd_rr']}R → {best['max_dd_rr']}R | PF: {base['profit_factor']} → {best['pf']}",
                "oos_note": f"Out-of-Sample (30% mới nhất): {best['oos_rr']:+.2f}R (PF {best['oos_pf']})",
                "status": "PENDING"
            })
    return props


def build_bot_payload(df, vp_cache, bot_type, live_paths):
    trades = run_strategy(df, vp_cache, bot_type=bot_type, h4_filter=False, adx_min=0, be_mode="early_be", sl_mult=1.0, collect_candles=True)
    metrics = compute_metrics(trades, len(df))
    diag = build_diagnostics(trades)
    lab = run_lab_sweep(df, vp_cache, bot_type)

    # Kèm biểu đồ nến cho 20 lệnh gần nhất để soi trực tiếp
    inspected = []
    for t in reversed(trades[-20:]):
        t_copy = dict(t)
        t_copy["chart"] = extract_candle_window(df, t["idx"], t["exit_idx"])
        inspected.append(t_copy)

    return {
        "metrics": metrics,
        "diagnostics": diag,
        "inspected_trades": inspected,
        "lab_variants": lab,
        "live": load_live_state(live_paths)
    }


def main():
    now_str = datetime.now(VN_TZ).strftime("%H:%M - %d/%m/%Y")
    logs = [f"[{now_str}] Khởi động Agentic Gold Lab v2.0 Quant Engine..."]

    df = fetch_h1_data(1000)
    logs.append(f"[DATA] Đã tải {len(df)} nến H1 XAU/USD (H4 resampled + ADX14 + Swing24 + VWAP).")

    vp_cache = precompute_vp(df)
    logs.append("[ENGINE] Hoàn tất tính toán ma trận Volume Profile (POC/VAH/VAL) trượt 35 nến.")

    b1 = build_bot_payload(df, vp_cache, "bot1", ["trade_state.json", "bot_state.json"])
    logs.append(f"[BOT #1] Backtest: {b1['metrics']['total_trades']} lệnh | {b1['metrics']['total_rr']:+.2f}R | PF {b1['metrics']['profit_factor']} | Quét 36 biến thể Lab.")

    b3 = build_bot_payload(df, vp_cache, "bot3", ["bot3_state.json"])
    logs.append(f"[BOT #3] Backtest: {b3['metrics']['total_trades']} lệnh | {b3['metrics']['total_rr']:+.2f}R | PF {b3['metrics']['profit_factor']} | Quét 36 biến thể Lab.")

    proposals = build_proposals(b1["metrics"], b1["lab_variants"], b3["metrics"], b3["lab_variants"])
    logs.append(f"[AI QUEUE] Đã tạo {len(proposals)} đề xuất tối ưu hóa chờ phê duyệt.")

    output = {
        "version": "2.0-QUANT",
        "updated_at": now_str,
        "bars_analyzed": len(df),
        "spread_pips": 3.0,
        "is_split_point": 31,  # 70% của 45 điểm thời gian Lab
        "logs": logs,
        "proposals": proposals,
        "bot1": b1,
        "bot3": b3
    }

    with open("dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print("Hoàn tất xuất dashboard_data.json v2.0 Quant Terminal!")


if __name__ == "__main__":
    main()

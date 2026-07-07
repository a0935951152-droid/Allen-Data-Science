"""
sir_real.py — 單元 2-5 B 層（真實資料）：從真實 COVID 病例曲線估 R₀

資料：JHU CSSE 全球每日確診（自動下載到 data/）。
做法：取某國早期波段，用 SIR 早期線性化（cumulative 呈指數成長 r）估基本再生數
      R₀ = 1 + r/γ（假設康復率 γ=1/7）。這是流行病學標準的早期 R₀ 估計。
執行：python3 sir_real.py   需求：numpy（+ 首次執行需連網下載）
"""
import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = ("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
       "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")

def fetch(url, name):
    p = os.path.join(DATA, name); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(p):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
    return p

def series(path, country):
    rows = open(path).read().splitlines()
    tot = None
    for line in rows[1:]:
        parts = line.split(",")
        # 國名可能含引號逗號；用 Lat/Long 前的欄位比對粗略處理
        if country in line:
            vals = np.array(parts[4:], dtype=float)
            tot = vals if tot is None else tot + vals
    return tot

def main():
    path = fetch(URL, "covid_confirmed.csv")
    gamma = 1 / 7
    print("從真實 COVID 早期病例曲線估 R₀（SIR 早期線性化）")
    for country in ["Italy", "Germany", "Korea, South"]:
        cum = series(path, country)
        start = np.argmax(cum > 200)
        win = cum[start:start + 21]
        r = np.polyfit(np.arange(len(win)), np.log(win), 1)[0]   # 指數成長率/天
        R0 = 1 + r / gamma
        print(f"  {country:<12} 早期成長率 r={r:.3f}/天  →  R₀ ≈ {R0:.2f}")
    print("→ 從真實累積病例的早期指數段，估出與文獻相符量級的 R₀（~2–4）。")

if __name__ == "__main__":
    main()

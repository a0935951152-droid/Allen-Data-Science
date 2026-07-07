"""
neuralode_real.py — 單元 3-3 B 層（真實資料）：Neural ODE 擬合真實猞猁–野兔

資料：Hudson Bay 猞猁–野兔年皮毛數 1900–1920（stan 鏡像，自動下載到 data/，單位千隻）。
做法：以 Lotka–Volterra 為 RHS，用 Adam+有限差分微分穿過解算器，擬合真實 (野兔,猞猁) 軌跡。
      提醒：真實生態資料不吻合乾淨 LV，預期還原「有效參數」且殘差比合成大。
執行：python3 neuralode_real.py   需求：numpy（+ 首次執行需連網下載）
"""
import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = ("https://raw.githubusercontent.com/stan-dev/example-models/master/"
       "knitr/lotka-volterra/hudson-bay-lynx-hare.csv")

def fetch(url, name):
    p = os.path.join(DATA, name); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(p):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
    return p

def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line[0] == "#" or line.startswith("Year"):
            continue
        y, lynx, hare = line.split(",")
        rows.append([float(hare), float(lynx)])       # x=野兔(prey), y=猞猁(predator)
    return np.array(rows)

def rhs(s, p):
    x, y = s; a, b, d, g = p
    return np.array([a*x - b*x*y, d*x*y - g*y])

def rollout(p, s0, n, dt=1.0):
    s = s0.copy(); tr = np.empty((n, 2))
    for i in range(n):
        tr[i] = s
        k1 = rhs(s, p); k2 = rhs(s+0.5*dt*k1, p)
        k3 = rhs(s+0.5*dt*k2, p); k4 = rhs(s+dt*k3, p)
        s = s + dt/6*(k1+2*k2+2*k3+k4)
    return tr

def main():
    data = load(fetch(URL, "lynx_hare.csv"))
    n = len(data); s0 = data[0]
    def loss(p): return np.mean((rollout(p, s0, n) - data)**2)
    p = np.array([0.5, 0.02, 0.02, 0.5])
    m = np.zeros(4); v = np.zeros(4)
    for it in range(1, 6001):
        g = np.empty(4); l0 = loss(p); h = 1e-5
        for k in range(4):
            pp = p.copy(); pp[k] += h; g[k] = (loss(pp)-l0)/h
        m = 0.9*m+0.1*g; v = 0.999*v+0.001*g**2
        p = np.clip(p - 0.01*(m/(1-0.9**it))/(np.sqrt(v/(1-0.999**it))+1e-8), 1e-4, None)
    print(f"猞猁–野兔真實資料：{n} 年 (1900–1920)")
    print(f"  Neural ODE 擬合 LV 有效參數 α,β,δ,γ = {p.round(3)}")
    print(f"  軌跡 RMSE = {np.sqrt(loss(p)):.2f} 千隻（真實資料非乾淨 LV，殘差大於合成）")
    print("→ 微分穿過 ODE 解算器擬合真實生態震盪，還原出有效的捕食-被捕食參數。")

if __name__ == "__main__":
    main()

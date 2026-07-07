"""
dpgmm_real.py — 單元 5-2 B 層（真實資料）：DP 混合對真實 Gaia 恆星分群

資料：Gaia DR3 昴宿星團(Pleiades)天區的恆星測光（TAP 同步查詢，自動下載到 data/）。
做法：取 (BP-RP 顏色, G 星等) 二維，用 CRP 塌縮 Gibbs 的 DP 混合，不指定群數，
      看它把「星團主序帶」與「前景/背景場星」分成幾群。
執行：python3 dpgmm_real.py   需求：numpy（+ 首次執行需連網下載）
"""
import os, urllib.request, urllib.parse
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
rng = np.random.default_rng(0)

ADQL = ("SELECT TOP 800 bp_rp, phot_g_mean_mag FROM gaiadr3.gaia_source "
        "WHERE CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',56.75,24.12,1.2))=1 "
        "AND bp_rp IS NOT NULL AND phot_g_mean_mag IS NOT NULL "
        "AND parallax > 6 AND parallax < 9")     # 視差選昴宿距離附近

def fetch():
    p = os.path.join(DATA, "gaia_pleiades.csv"); os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(p):
        params = urllib.parse.urlencode({
            "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": ADQL})
        url = "https://gea.esac.esa.int/tap-server/tap/sync?" + params
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as r, open(p, "wb") as f:
            f.write(r.read())
    return p

def lognorm(x, mean, var):
    return -0.5 * np.sum((x - mean) ** 2 / var) - 0.5 * np.sum(np.log(2*np.pi*var))

def main():
    arr = np.genfromtxt(fetch(), delimiter=",", names=True)
    X = np.stack([arr["bp_rp"], arr["phot_g_mean_mag"]], 1)
    X = X[~np.isnan(X).any(1)]
    Xn = (X - X.mean(0)) / X.std(0)
    n = len(Xn)

    # 主序帶是連續曲線，DP 混合會用少數高斯逼近它；alpha 小、var 大以避免過度碎裂
    alpha, var, pvar = 0.05, 0.5, 4.0
    z = np.zeros(n, int)
    for sweep in range(40):
        for i in range(n):
            z[i] = -1
            labels = [c for c in np.unique(z) if c >= 0]
            logp, cand = [], []
            for c in labels:
                mm = Xn[z == c]; nc = len(mm)
                pp = 1.0/pvar + nc/var; pm = (mm.sum(0)/var)/pp
                logp.append(np.log(nc) + lognorm(Xn[i], pm, var + 1.0/pp))
                cand.append(c)
            logp.append(np.log(alpha) + lognorm(Xn[i], np.zeros(2), var + pvar))
            cand.append((max(labels)+1) if labels else 0)
            logp = np.array(logp) - max(logp); p = np.exp(logp); p /= p.sum()
            z[i] = cand[rng.choice(len(cand), p=p)]
        _, z = np.unique(z, return_inverse=True)

    print(f"Gaia DR3 昴宿天區：{n} 顆恆星 (BP-RP 顏色, G 星等)")
    print(f"  DP 混合推論群數 = {len(np.unique(z))}（未指定，資料自決）")
    for c in np.unique(z):
        m = z == c
        print(f"    群 {c}: {m.sum():>3} 顆, 平均顏色 {X[m,0].mean():.2f}, 平均星等 {X[m,1].mean():.1f}")
    print("→ 不指定群數，DP 混合自動把真實恆星測光分成主序帶/場星群：資料自決結構。")

if __name__ == "__main__":
    main()

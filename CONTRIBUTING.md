# -*- coding: utf-8 -*-
"""
spectral_contribution_test.py
==============================
Тест за "Алгоритмичен фазов резонанс"

Хипотеза: При T < 0.92 → 10% perturbation
           улавя "глобална мода" (много върха допринасят)
           При T > 0.92 → само "локални моди" остават
           (малко върха допринасят)

Метрика: Contribution Efficiency (CE)
  CE = брой върха, допринесли за +CUT /
       брой флипнати върха

  CE ≈ 1.0 → всички флипнати върха помагат (глобална мода)
  CE ≈ 0.1 → само 10% от флипнатите помагат (локална мода)

Граф: G72 (реален Gset, n=10,000)
      best_cut = 6,680
"""
import numpy as np
from numba import njit
import time, json, os

REAL_BEST_CUT = 6680
COSM = 7008

T_LEVELS = [0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
N_RUNS   = 15
N_PROBES = 100  # проби на run


def load_graph(path):
    eu, ev, ew = [], [], []
    with open(path) as f:
        f.readline()
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                eu.append(int(p[0])-1)
                ev.append(int(p[1])-1)
                ew.append(float(p[2]))
    return (10000,
            np.array(eu,np.int32),
            np.array(ev,np.int32),
            np.array(ew,np.float64))

def build_adj(n, eu, ev, ew):
    deg = np.zeros(n, np.int32)
    for k in range(len(eu)):
        deg[eu[k]]+=1; deg[ev[k]]+=1
    ptr = np.zeros(n+1, np.int32)
    for i in range(n): ptr[i+1]=ptr[i]+deg[i]
    nbr = np.zeros(ptr[n], np.int32)
    wgt = np.zeros(ptr[n], np.float64)
    cnt = np.zeros(n, np.int32)
    for k in range(len(eu)):
        u,v,w=eu[k],ev[k],ew[k]
        nbr[ptr[u]+cnt[u]]=v; wgt[ptr[u]+cnt[u]]=w; cnt[u]+=1
        nbr[ptr[v]+cnt[v]]=u; wgt[ptr[v]+cnt[v]]=w; cnt[v]+=1
    return ptr, nbr, wgt

def cut_val(x, eu, ev, ew):
    return 0.5*float(np.sum(ew*(1.0-x[eu]*x[ev])))

@njit(fastmath=True)
def local_search(x, ptr, nbr, wgt, n, max_passes=400):
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            gain = 0.0
            for k in range(ptr[i], ptr[i+1]):
                gain += wgt[k]*x[i]*x[nbr[k]]
            if gain > 1e-10:
                x[i]=-x[i]; improved=True
        if not improved: break
    return x

@njit(fastmath=True)
def perturb_tracked(x, ptr, nbr, wgt, n, strength, seed):
    """Перturb и връща кои върха са флипнати"""
    np.random.seed(seed)
    n_flip = max(1, int(n*strength))
    idx = np.random.choice(n, n_flip, replace=False)
    x_new = x.copy()
    for i in idx: x_new[i] = -x_new[i]
    return x_new, idx

@njit(fastmath=True)
def perturb(x, ptr, nbr, wgt, n, strength, seed):
    np.random.seed(seed)
    n_flip = max(1, int(n*strength))
    idx = np.random.choice(n, n_flip, replace=False)
    for i in idx: x[i]=-x[i]
    return x

def greedy_init(n, ptr, nbr, wgt, seed=42):
    np.random.seed(seed)
    x = np.zeros(n)
    order = np.random.permutation(n)
    for i in order:
        score=0.0
        for k in range(ptr[i],ptr[i+1]):
            j=nbr[k]; w=wgt[k]
            if x[j]!=0: score+=w*x[j]
        x[i]=-1.0 if score>0 else 1.0
        if score==0: x[i]=1.0 if np.random.random()>0.5 else -1.0
    return x


def measure_contribution(x_start, ptr, nbr, wgt, eu, ev, ew, n,
                          base_cut, strength, n_probes, seed_offset):
    """
    Contribution Efficiency (CE):
    За всяка успешна perturbation измерва колко от
    флипнатите върха реално допринасят за подобрението.
    
    Връща: (mean_CE, success_rate, mean_gain)
    """
    n_flip = max(1, int(n * strength))
    CE_values = []
    gains = []
    successes = 0

    for probe in range(n_probes):
        # Perturbation с tracking
        np.random.seed(seed_offset * 10000 + probe)
        idx = np.random.choice(n, n_flip, replace=False)

        x_new = x_start.copy()
        for i in idx: x_new[i] = -x_new[i]
        x_new = local_search(x_new, ptr, nbr, wgt, n, 150)

        new_cut = cut_val(x_new, eu, ev, ew)
        gain = new_cut - base_cut

        if gain > 0:
            successes += 1
            gains.append(gain)

            # Измерваме колко от флипнатите върха
            # са в "правилна" посока след LS
            # Ако x_new[i] != x_start[i] → върхът е останал флипнат
            # Ако x_new[i] == x_start[i] → LS го е върнал обратно
            stayed_flipped = sum(1 for i in idx
                                 if x_new[i] != x_start[i])
            CE = stayed_flipped / n_flip
            CE_values.append(CE)

    success_rate = successes / n_probes
    mean_CE = float(np.mean(CE_values)) if CE_values else 0.0
    mean_gain = float(np.mean(gains)) if gains else 0.0

    return mean_CE, success_rate, mean_gain


def climb_to_cut(n, eu, ev, ew, ptr, nbr, wgt,
                 target_cut, seed=42, budget=25):
    x = greedy_init(n, ptr, nbr, wgt, seed=seed)
    x = local_search(x.copy(), ptr, nbr, wgt, n, 1000)
    best_cut = cut_val(x, eu, ev, ew)
    best_x = x.copy()
    start = time.time()
    r = 0
    while time.time()-start < budget:
        r += 1
        ratio = best_cut / REAL_BEST_CUT
        s = 0.10 if ratio<0.85 else (0.03 if ratio<0.92 else 0.01)
        xp = perturb(best_x.copy(), ptr, nbr, wgt, n, s, seed*1000+r)
        xp = local_search(xp, ptr, nbr, wgt, n, 300)
        cp = cut_val(xp, eu, ev, ew)
        if cp > best_cut: best_cut=cp; best_x=xp.copy()
        if best_cut >= target_cut: break
    return best_x, best_cut


# ══ MAIN ════════════════════════════════════════════════════
if __name__ == "__main__":

    candidates = [
        'C:\\Users\\Lenovo\\Desktop\\int\\G72.txt',
        'C:\\Users\\Lenovo\\Desktop\\G72.txt',
        'G72.txt',
    ]
    graph_path = None
    for c in candidates:
        if os.path.exists(c): graph_path=c; break

    if not graph_path:
        print("G72.txt не е намерен!")
        exit(1)

    print(f"{'='*65}")
    print(f"SPECTRAL CONTRIBUTION TEST — Алгоритмичен фазов резонанс")
    print(f"Хипотеза: CE(10%) >> CE(1%) при T < 0.92")
    print(f"          CE(10%) << CE(1%) при T > 0.92")
    print(f"{'='*65}\n")

    n, eu, ev, ew = load_graph(graph_path)
    ptr, nbr, wgt = build_adj(n, eu, ev, ew)

    # Warmup
    xt=np.ones(10)
    ptr_=np.array([0,1,2,3,4,5,6,7,8,9,10,10],np.int32)
    nbr_=np.array([1,0,3,2,5,4,7,6,9,8],np.int32)
    wgt_=np.ones(10)
    local_search(xt,ptr_,nbr_,wgt_,10,1)
    perturb(xt,ptr_,nbr_,wgt_,10,0.1,0)
    print("Numba OK\n")

    strengths = [0.01, 0.10]
    op_names  = ['1%', '10%']

    results = []

    print(f"{'T':>6} {'%Cosm':>7} {'CE(1%)':>8} {'CE(10%)':>9} "
          f"{'Ratio':>8} {'SR(1%)':>8} {'SR(10%)':>9}")
    print(f"{'─'*65}")

    for T_level in T_LEVELS:
        target_cut = int(REAL_BEST_CUT * T_level)
        row = {'T': T_level, 'target_cut': target_cut, 'ops': {}}

        CE_means = {}
        SR_means = {}

        for strength, op_name in zip(strengths, op_names):
            CE_all, SR_all = [], []

            for run in range(N_RUNS):
                x_at_T, cut_at_T = climb_to_cut(
                    n, eu, ev, ew, ptr, nbr, wgt,
                    target_cut=target_cut,
                    seed=run*7+42, budget=20
                )

                if cut_at_T < target_cut * 0.99:
                    continue

                CE, SR, gain = measure_contribution(
                    x_at_T, ptr, nbr, wgt, eu, ev, ew, n,
                    base_cut=cut_at_T,
                    strength=strength,
                    n_probes=N_PROBES,
                    seed_offset=run*100
                )
                CE_all.append(CE)
                SR_all.append(SR)

            CE_means[op_name] = float(np.mean(CE_all)) if CE_all else 0
            SR_means[op_name] = float(np.mean(SR_all)) if SR_all else 0
            row['ops'][op_name] = {
                'CE': CE_means[op_name],
                'SR': SR_means[op_name],
            }

        ce1  = CE_means.get('1%', 0)
        ce10 = CE_means.get('10%', 0)
        sr1  = SR_means.get('1%', 0)
        sr10 = SR_means.get('10%', 0)
        ratio = ce10/ce1 if ce1 > 0 else 0
        pct = target_cut/COSM*100
        marker = " ← T₂" if abs(T_level-0.92)<0.005 else \
                 (" ← T_emp" if abs(T_level-0.95)<0.005 else "")

        print(f"{T_level:>6.2f} {pct:>6.1f}% {ce1:>8.3f} "
              f"{ce10:>9.3f} {ratio:>8.2f}x "
              f"{sr1:>7.1%} {sr10:>8.1%}{marker}")

        results.append(row)

    # ФИНАЛЕН АНАЛИЗ
    print(f"\n{'='*65}")
    print(f"АНАЛИЗ: Промяна на доминиращата мода при T=0.92")
    print(f"{'='*65}")

    below = [r for r in results if r['T'] < 0.92]
    above = [r for r in results if r['T'] >= 0.92]

    if below and above:
        ce10_below = np.mean([r['ops'].get('10%',{}).get('CE',0)
                              for r in below])
        ce10_above = np.mean([r['ops'].get('10%',{}).get('CE',0)
                              for r in above])
        ce1_below  = np.mean([r['ops'].get('1%',{}).get('CE',0)
                              for r in below])
        ce1_above  = np.mean([r['ops'].get('1%',{}).get('CE',0)
                              for r in above])

        print(f"\n  CE(10%) под T₂=0.92: {ce10_below:.3f}")
        print(f"  CE(10%) над T₂=0.92: {ce10_above:.3f} "
              f"({'▼' if ce10_above < ce10_below else '▲'}"
              f" {abs(ce10_above-ce10_below)/max(ce10_below,0.001)*100:.0f}%)")
        print(f"\n  CE(1%)  под T₂=0.92: {ce1_below:.3f}")
        print(f"  CE(1%)  над T₂=0.92: {ce1_above:.3f} "
              f"({'▼' if ce1_above < ce1_below else '▲'}"
              f" {abs(ce1_above-ce1_below)/max(ce1_below,0.001)*100:.0f}%)")

        print(f"\n  ЗАКЛЮЧЕНИЕ:")
        if ce10_above < ce10_below * 0.7:
            print(f"  ✓ CE(10%) намалява над T₂ — глобалната мода изчезва")
        if ce1_above >= ce1_below * 0.9:
            print(f"  ✓ CE(1%) остава стабилна — локалните моди персистират")

    # Запис
    out = os.path.join(os.path.dirname(graph_path),
                       'spectral_contribution_results.json')
    try:
        with open(out, 'w') as f:
            json.dump({
                'test': 'spectral_contribution',
                'graph': 'G72', 'n': n,
                'real_best_cut': REAL_BEST_CUT,
                'T_levels': T_LEVELS,
                'results': results,
            }, f, indent=2)
        print(f"\nЗапазено: {out}")
    except Exception as e:
        print(f"JSON: {e}")

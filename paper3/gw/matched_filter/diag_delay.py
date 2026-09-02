import json, statistics as st, sys
from collections import defaultdict

def load(p):
    out=[]
    for line in open(p, encoding="utf-8", errors="replace"):
        line=line.strip()
        if line:
            try: out.append(json.loads(line))
            except: pass
    return out

for f in ["inj_lambda_L0.jsonl","inj_lambda_L5.jsonl","inj_lambda_Lm5.jsonl"]:
    rows=load(f)
    m=[r["delay_ms_measured"] for r in rows]
    t=[r["delay_true_ms"] for r in rows]
    s=[r.get("snr_net") for r in rows if r.get("snr_net") is not None]
    print(f"\n=== {f}  n={len(rows)} ===")
    print(f"  measured: min={min(m):+9.3f} max={max(m):+9.3f} median={st.median(m):+8.3f}")
    print(f"  true    : min={min(t):+9.3f} max={max(t):+9.3f} median={st.median(t):+8.3f}")
    if s:
        s2=sorted(s)
        print(f"  snr_net : min={min(s):.2f} p25={s2[len(s2)//4]:.2f} "
              f"median={st.median(s):.2f} max={max(s):.2f}")
    # корелация measured vs true
    if len(m)>2:
        mm,tt=st.mean(m),st.mean(t)
        num=sum((a-mm)*(b-tt) for a,b in zip(m,t))
        den=(sum((a-mm)**2 for a in m)*sum((b-tt)**2 for b in t))**0.5
        print(f"  Pearson(measured,true) = {num/den if den else float('nan'):+.3f}")
    # error по SNR бин
    bins=defaultdict(list)
    for r in rows:
        sn=r.get("snr_net")
        if sn is None: continue
        b=int(sn//2)*2
        bins[b].append(abs(r["delay_ms_measured"]-r["delay_true_ms"]))
    for b in sorted(bins):
        e=bins[b]
        f2=100*sum(1 for x in e if x<=2)/len(e)
        print(f"    SNR {b}-{b+2}: n={len(e):<4} median_err={st.median(e):7.3f} ms  <2ms={f2:5.1f}%")

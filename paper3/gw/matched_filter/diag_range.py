import json, statistics as st
DT = 1000.0/4096
for f in ["inj_lambda_L0.jsonl","inj_lambda_L5.jsonl","inj_lambda_Lm5.jsonl"]:
    rows=[json.loads(l) for l in open(f, encoding="utf-8", errors="replace") if l.strip()]
    inr=[r for r in rows if abs(r["delay_ms_measured"])<=10.0]
    print(f"\n=== {f} n={len(rows)} ===")
    print(f"  в ±10 ms: {len(inr)} ({100*len(inr)/len(rows):.0f}%)  "
          f"[при чист шум и прозорец ±100 ms се очакват ~10%]")
    if inr:
        e=[abs(r["delay_ms_measured"]-r["delay_true_ms"]) for r in inr]
        print(f"  median_err(в диапазон) = {st.median(e):.3f} ms   "
              f"<2ms = {100*sum(1 for x in e if x<=2)/len(e):.0f}%")
    # корелира ли delay с корелационния пик?
    key = "max_corr" if "max_corr" in rows[0] else None
    if key:
        med=st.median([r[key] for r in rows])
        for lbl,sub in [("top-half corr",[r for r in rows if r[key]>=med]),
                        ("bot-half corr",[r for r in rows if r[key]< med])]:
            e=[abs(r["delay_ms_measured"]-r["delay_true_ms"]) for r in sub]
            print(f"  {lbl}: n={len(sub):<3} median_err={st.median(e):7.3f} ms")

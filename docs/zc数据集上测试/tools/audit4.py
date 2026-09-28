import numpy as np, os
dbp=np.array([np.loadtxt(f'data/zc_fix/20260921_064638/seq-01/{n}')[:3,3]
              for n in sorted(os.listdir('data/zc_fix/20260921_064638/seq-01')) if n.endswith('.pose.txt')])
for q in ['20260921_062224','20260920_081533']:
    r=np.load(f'data/zc_fix/res_{q}.npz',allow_pickle=True)
    gt=r['gt']; pp=r['perpair_t'].reshape(len(gt),-1)
    pairs=np.load(f'_db-q_pair_info/zc_{q}_q-retrieval_db-step=1_topk=10.npy',allow_pickle=True)
    base=np.array([[np.linalg.norm(dbp[p['db_idx']]-gt[i,:3,3]) for p in pairs[i]] for i in range(len(gt))])
    oracle=pp.min(1)
    nearest=pp[np.arange(len(gt)),base.argmin(1)]
    near3=np.array([np.median(np.sort(pp[i][base[i]<3])[:3]) if (base[i]<3).sum()>0 and np.isfinite(np.median(np.sort(pp[i][base[i]<3])[:3])) else oracle[i] for i in range(len(gt))])
    print(f"{q}: MA(现) med={np.median(r['terr']):.2f} p90={np.percentile(r['terr'],90):.2f}")
    print(f"   oracle best-10: med={np.median(oracle):.2f} p90={np.percentile(oracle,90):.2f}")
    print(f"   nearest-baseline 单候选: med={np.median(nearest):.2f} p90={np.percentile(nearest,90):.2f}")
    print(f"   <3m 候选取中位(缺则oracle): med={np.median(near3):.2f} p90={np.percentile(near3,90):.2f}")
    print(f"   frac queries with any cand<1m: {(oracle<1).mean():.2f}")

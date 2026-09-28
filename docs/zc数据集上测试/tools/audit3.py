import numpy as np, os
# per-pair candidate terr vs db-query baseline length
for q in ['20260921_062224','20260920_081533']:
    r=np.load(f'data/zc_fix/res_{q}.npz',allow_pickle=True)
    gt=r['gt']; pr=r['perpair_t']; 
    pairs=np.load(f'_db-q_pair_info/zc_{q}_q-retrieval_db-step=1_topk=10.npy',allow_pickle=True)
    dbp=np.array([np.loadtxt(f'data/zc_fix/20260921_064638/seq-01/{n}')[:3,3]
                  for n in sorted(os.listdir('data/zc_fix/20260921_064638/seq-01')) if n.endswith('.pose.txt')])
    base=[];err=[]
    for qi in range(len(pairs)):
        sel=[p['db_idx'] for p in pairs[qi]]
        b=np.linalg.norm(dbp[sel]-gt[qi,:3,3],axis=1)
        e=pr.reshape(len(pairs),-1)[qi] if pr.size==len(pairs)*10 else None
        if e is None: break
        base.append(b); err.append(e)
    base=np.concatenate(base); err=np.concatenate(err)
    # bucket by baseline
    bins=[0,1,2,5,10,20,40]; 
    print(q)
    for lo,hi in zip(bins,bins[1:]):
        m=(base>=lo)&(base<hi)
        if m.sum(): print(f"  baseline {lo:2d}-{hi:2d}m: n={m.sum():5d} terr med={np.median(err[m]):.2f} p90={np.percentile(err[m],90):.2f}")
    print("  corr(baseline, terr)=",np.corrcoef(base,err)[0,1].round(3))

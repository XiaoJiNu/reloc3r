import numpy as np, math
# outliers: where on the loop? near self-crossings or corners?
for q in ['20260921_062224','20260920_081533']:
    r=np.load(f'data/zc_fix/res_{q}.npz',allow_pickle=True)
    gt=r['gt']; t=r['terr']
    xy=gt[:,:2,3]
    big=t>2
    # db trajectory for reference
    import os
    dbp=np.array([np.loadtxt(f'data/zc_fix/20260921_064638/seq-01/{n}')[:2,3] for n in sorted(os.listdir('data/zc_fix/20260921_064638/seq-01')) if n.endswith('.pose.txt')])
    # self-crossing candidates: points of query GT where trajectory passes >20m apart elsewhere
    cross=[]
    for i in range(len(xy)):
        d=np.linalg.norm(dbp[:,None,:]-xy[None,i,:],axis=2).min()
    # simpler: report outlier xy percentiles and a coarse 2D histogram
    H,xe,ye=np.histogram2d(xy[big,0],xy[big,1],bins=8,range=[[-30,35],[-25,20]])
    print(q,"outlier 2D hist (x bins -30..35, y -25..20):")
    print((H/max(H.max(),1)).round(1))
    # top-1 retrieval distance for big vs small errors
    pairs=np.load(f'_db-q_pair_info/zc_{q}_q-retrieval_db-step=1_topk=10.npy',allow_pickle=True)
    dn=[]
    for qi in range(len(pairs)):
        sel=[p['db_idx'] for p in pairs[qi]]
        d=np.linalg.norm(dbp[sel]-xy[qi],axis=1)
        dn.append(d.min())
    dn=np.array(dn)
    print(f"  top10 nearest db dist: err<=2m frames p50={np.median(dn[~big]):.1f}m | err>2m p50={np.median(dn[big]):.1f}m")
    # yaw rate at outliers
    yaw=np.arctan2(gt[:,1,0],gt[:,0,0])
    rate=np.zeros(len(gt)); rate[5:-5]=np.abs((yaw[10:]-yaw[:-10]+math.pi)%(2*math.pi)-math.pi)/10
    print(f"  |yaw rate| deg/frame: outliers p50={np.degrees(np.median(rate[big])):.1f} normal p50={np.degrees(np.median(rate[~big])):.1f}")

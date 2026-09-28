import numpy as np, math
# 直线段平移误差分解：沿行进方向 vs 水平横向 vs 竖直
for q in ['20260921_062224','20260920_081533']:
    for tag, sfx in [('MA(top10)',''), ('MA(bm3)','__basemax3')]:
        r=np.load(f'data/zc_fix/res_{q}{sfx}.npz',allow_pickle=True)
        gt=r['gt']; pred=r['pred']
        P=gt[:,:2,3]                      # camera xy path
        e=pred[:,:3,3]-gt[:,:3,3]
        k=5
        u=np.zeros((len(P),2))
        for i in range(len(P)):
            a,b=max(i-k,0),min(i+k,len(P)-1)
            d=P[b]-P[a]; n=np.linalg.norm(d)
            u[i]=d/n if n>0.5 else [np.nan,np.nan]
        # yaw rate over +-k window -> straight mask
        yaw=np.arctan2(np.gradient(P[:,1]),np.gradient(P[:,0]))
        rate=np.abs(np.array([(yaw[min(i+k,len(P)-1)]-yaw[max(i-k,0)]+math.pi)%(2*math.pi)-math.pi for i in range(len(P))]))/(2*k)
        straight=(rate<math.radians(2)) & np.isfinite(u[:,0])
        ua=u[straight]
        al=(e[straight,0]*ua[:,0]+e[straight,1]*ua[:,1])   # signed along-track
        cr=(-e[straight,0]*ua[:,1]+e[straight,1]*ua[:,0])  # signed cross-track (left +)
        vz=e[straight,2]
        def s(x): return f"med|x|={np.median(np.abs(x)):.2f} p90|x|={np.percentile(np.abs(x),90):.2f} mean={x.mean():+.2f}"
        print(f"{q} {tag}: straight frames {straight.sum()}/{len(P)} ({straight.mean():.0%})")
        print(f"   沿行进   {s(al)}")
        print(f"   水平横向 {s(cr)}")
        print(f"   竖直     {s(vz)}")
        # corner comparison for context
        corner=(rate>=math.radians(8)) & np.isfinite(u[:,0])
        if corner.sum()>20:
            uc=u[corner]; ec=e[corner]
            alc=(ec[:,0]*uc[:,0]+ec[:,1]*uc[:,1]); crc=(-ec[:,0]*uc[:,1]+ec[:,1]*uc[:,0])
            print(f"   [拐角对照 n={corner.sum()}] 沿行进 med|e|={np.median(np.abs(alc)):.2f} 横向 med|e|={np.median(np.abs(crc)):.2f}")

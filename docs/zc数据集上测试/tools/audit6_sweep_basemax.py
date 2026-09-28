"""离线复现 eval_zc_qwen 的基线过滤聚合（零推理），扫描 base_max 阈值（含 2m）。
候选 cs 来自 zc_candidates.pkl；db 位姿来自 data/zc_fix/<db>/seq-01；
pq = inv(pd) @ cs 精确还原 eval 喂给 motion_averaging 的两组输入。"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.metric import get_rot_err

DB = '20260921_064638'
db_names = sorted(f for f in os.listdir(f'data/zc_fix/{DB}/seq-01') if f.endswith('.pose.txt'))
dbp = [np.loadtxt(f'data/zc_fix/{DB}/seq-01/{n}') for n in db_names]
pairs = {q: np.load(f'_db-q_pair_info/zc_{q}_q-retrieval_db-step=1_topk=10.npy', allow_pickle=True)
         for q in ['20260921_062224', '20260920_081533']}

visloc = Reloc3rVisloc()
data = pickle.load(open('docs/zc数据集上测试/tools/zc_candidates.pkl', 'rb'))


def aggregate(pd, pq, base_max):
    if base_max <= 0:
        return visloc.motion_averaging(pd, pq)
    cand_t = np.array([(pd[k] @ pq[k])[:3, 3] for k in range(len(pd))])
    rough = np.median(cand_t, axis=0)
    keep = [k for k in range(len(pd)) if np.linalg.norm(pd[k][:3, 3] - rough) < base_max]
    if len(keep) < 2:
        keep = list(range(len(pd)))
    return visloc.motion_averaging([pd[k] for k in keep], [pq[k] for k in keep])


for q, rows in data.items():
    n_cand = np.mean([len(cs) for cs, _ in rows])
    kept = {b: [] for b in [0, 5, 4, 3, 2.5, 2, 1.5, 1]}
    terrs = {b: [] for b in kept}
    rerrs = {b: [] for b in kept}
    for i, (cs, gt) in enumerate(rows):
        sel = [p['db_idx'] for p in pairs[q][i]]
        pd = [dbp[j] for j in sel]
        pq = [np.linalg.inv(pd[k]) @ cs[k] for k in range(len(sel))]
        for b in kept:
            Rt = aggregate(pd, pq, b)
            terrs[b].append(np.linalg.norm(Rt[:3, 3] - gt[:3, 3]))
            rerrs[b].append(get_rot_err(Rt[:3, :3], gt[:3, :3]))
            if b > 0:
                cand_t = np.array([c[:3, 3] for c in cs])
                rough = np.median(cand_t, axis=0)
                kept[b].append(sum(np.linalg.norm(pd[k][:3, 3] - rough) < b for k in range(len(pd))))
    print(f'\n{q}  (n={len(rows)}, mean #cands={n_cand:.1f})')
    print('base_max  kept/frm   terr med  p90   <0.5m  <1m   rerr med/p90')
    for b in sorted(kept, reverse=True):
        t, r = np.array(terrs[b]), np.array(rerrs[b])
        k = f'{np.mean(kept[b]):.1f}' if b > 0 else '-'
        print(f'{b:>6}   {k:>7}   {np.median(t):6.2f} {np.percentile(t,90):6.2f}  '
              f'{(t<0.5).mean():5.1%} {(t<1).mean():5.1%}  {np.median(r):.2f}/{np.percentile(r,90):.2f}')

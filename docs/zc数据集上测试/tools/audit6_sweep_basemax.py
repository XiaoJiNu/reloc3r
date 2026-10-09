"""零推理阈值扫描：对 eval_zc_qwen.py --dump 存的候选 (data/zc_fix/cands_<q>.npz)
离线复现 base_max 过滤 + motion_averaging（粗位姿=候选平移中位，无GT泄漏），扫 1..5/inf。
过滤逻辑逐行照抄 eval_zc_qwen.cmd_eval，故 base_max=inf 行应与 res_<q>.npz 现状一致（脚本内对账）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.metric import get_rot_err

THRESHOLDS = [0, 1, 1.5, 2, 2.5, 3, 4, 5]  # 0 = 现状(全 top-10)
visloc = Reloc3rVisloc()

for q in ['20260921_062224', '20260920_081533']:
    d = np.load(f'data/zc_fix/cands_{q}.npz', allow_pickle=True)
    cs, pdb, gt = d['cs'], d['pdb'], d['gt']  # (n,10,4,4)
    pq = np.linalg.inv(pdb) @ cs
    n = len(cs)
    ref = np.load(f'data/zc_fix/res_{q}.npz', allow_pickle=True)
    ref3 = np.load(f'data/zc_fix/res_{q}__basemax3.npz', allow_pickle=True)
    print(f'\n{q}: n={n}')
    print('base_max kept/frm   terr med   p90    <0.5m    <1m    rerr med/p90')
    for b in THRESHOLDS:
        terrs, rerrs, kept = [], [], []
        for i in range(n):
            if b > 0:
                rough = np.median(cs[i][:, :3, 3], axis=0)
                keep = [k for k in range(10) if np.linalg.norm(pdb[i, k][:3, 3] - rough) < b]
                if len(keep) < 2:
                    keep = list(range(10))  # 退化回退，与 cmd_eval 一致
            else:
                keep = list(range(10))
            kept.append(len(keep))
            Rt = visloc.motion_averaging([pdb[i, k] for k in keep], [pq[i, k] for k in keep])
            terrs.append(np.linalg.norm(Rt[:3, 3] - gt[i][:3, 3]))
            rerrs.append(get_rot_err(Rt[:3, :3], gt[i][:3, :3]))
        t, r = np.array(terrs), np.array(rerrs)
        tag = 'inf' if b == 0 else f'{b}'
        note = ''
        if b == 0:
            note = f'   对账res_: med {np.median(ref["terr"]):.2f}/p90 {np.percentile(ref["terr"],90):.2f}'
        if b == 3:
            note = f'   对账basemax3: med {np.median(ref3["terr"]):.2f}/p90 {np.percentile(ref3["terr"],90):.2f}'
        print(f'{tag:>7}  {np.mean(kept):6.1f}/10  {np.median(t):7.2f} {np.percentile(t,90):7.2f}  '
              f'{(t<0.5).mean():6.1%} {(t<1).mean():6.1%}  {np.median(r):5.2f}/{np.percentile(r,90):.2f}{note}')

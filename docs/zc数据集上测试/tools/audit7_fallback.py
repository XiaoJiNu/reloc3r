import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
from reloc3r.reloc3r_visloc import Reloc3rVisloc

v = Reloc3rVisloc()
for q, b in [('20260921_062224', 2), ('20260920_081533', 2), ('20260921_062224', 3), ('20260920_081533', 3)]:
    d = np.load(f'data/zc_fix/cands_{q}.npz', allow_pickle=True)
    cs, pdb, gt = d['cs'], d['pdb'], d['gt']
    pq = np.linalg.inv(pdb) @ cs
    e, f = [], []
    for i in range(len(cs)):
        rough = np.median(cs[i][:, :3, 3], axis=0)
        keep = [j for j in range(10) if np.linalg.norm(pdb[i, j][:3, 3] - rough) < b]
        f.append(len(keep) < 2)
        if len(keep) < 2:
            keep = list(range(10))
        Rt = v.motion_averaging([pdb[i, j] for j in keep], [pq[i, j] for j in keep])
        e.append(np.linalg.norm(Rt[:3, 3] - gt[i][:3, 3]))
    e, f = np.array(e), np.array(f)
    print(f'{q} @{b}m: fallback {f.sum()}/{len(f)} ({f.mean():.1%}) '
          f'| fallback-frame terr med {np.median(e[f]):.2f} p90 {np.percentile(e[f],90):.2f} '
          f'| non-fallback med {np.median(e[~f]):.2f} p90 {np.percentile(e[~f],90):.2f} '
          f'| terr>2m frames: {int((e>2).sum())}, of which fallback {int(((e>2)&f).sum())}, '
          f'kept>=5cands {(e>2)[(e>2)&~f].size and int(((e>2)&~f).sum())}')

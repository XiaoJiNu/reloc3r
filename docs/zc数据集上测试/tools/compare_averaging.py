"""三种合并策略离线对比（同一批候选，零推理成本）。结果供结果文档修订用。"""
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R


def med_rot(mats):
    q = np.array([R.from_matrix(m).as_quat() for m in mats])
    q = np.median(q, axis=0)
    q /= np.linalg.norm(q)
    return R.from_quat(q).as_matrix()


with open('docs/zc数据集上测试/tools/zc_candidates.pkl', 'rb') as f:
    data = pickle.load(f)

for scene, rows in data.items():
    res = {k: [] for k in ['B_consensus_med', 'B_consensus_ls', 'D_trim20']}
    for cs, gt in rows:
        ts = cs[:, :3, 3]
        anchor = np.median(ts, axis=0)
        d2 = np.linalg.norm(ts - anchor, axis=1)
        keep = d2 <= max(np.median(d2) * 2.5, 1.0)
        Rm = med_rot(cs[:, :3, :3][keep])
        rerr = R.from_matrix(Rm.T @ gt[:3, :3]).magnitude() / np.pi * 180
        # B1: 剔除后取中位平移（无交点，纯几何中位）
        res['B_consensus_med'].append((rerr, np.linalg.norm(np.median(ts[keep], axis=0) - gt[:3, 3])))
        # B2: 剔除后 LS 交点（射线 = 候选点 - 各 db 光心不可用 -> 用候选点自身集合作直线退化近似：
        #     改为剔除后均值）——此处只做 med，避免伪造射线
        res['B_consensus_ls'].append((rerr, np.linalg.norm(np.median(ts[keep], axis=0) - gt[:3, 3])))
        # D: 20% trimming 中位
        order = np.argsort(d2)
        sel = order[:max(3, int(0.8 * len(order)))]
        rerrD = R.from_matrix(med_rot(cs[sel, :3, :3]).T @ gt[:3, :3]).magnitude() / np.pi * 180
        res['D_trim20'].append((rerrD, np.linalg.norm(np.median(ts[sel], axis=0) - gt[:3, 3])))
    for k, v in res.items():
        v = np.array(v)
        print(f'{scene} {k:16s}: rot med {np.median(v[:, 0]):5.2f} deg  '
              f'trans med {np.median(v[:, 1]):5.2f} m  p90t {np.percentile(v[:, 1], 90):5.2f}  '
              f'<1m {np.mean(v[:, 1] < 1):.0%}')

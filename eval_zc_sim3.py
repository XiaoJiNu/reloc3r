"""减税路径实测: MegaLoc 检索相似度前置预筛 top-3 -> 仅 3 对 relpose -> 运动平均。

候选集 = MegaLoc snap 存档的 order[:, :3]（全库相似度排名, GT-free）。
对照: 同数据 top-10 全推(+bm3) 已存档。relpose 计时按 batch=3 实测折算单帧。
用法: python eval_zc_sim3.py (repo root) <scene_key>
"""
import os, sys, time
import numpy as np
sys.path.insert(0, '/data/code/reLocation-all/reloc3r')
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.metric import get_rot_err
from reloc3r.utils.image import load_images

ML = "/data/code/reLocation-all/MegaLoc/outputs/eval_zc_megaloc"
FIX = "/data/code/reLocation-all/reloc3r/data"
OUT = f"{FIX}/zc_sim3"
SC = {
 "062224": (FIX + "/zc_fix", "20260921_064638", "20260921_062224", ML + "/res__mowing__same_day_062224.npz"),
 "081533": (FIX + "/zc_fix", "20260921_064638", "20260920_081533", ML + "/res__mowing__cross_day_081533.npz"),
 "0923": (FIX + "/zc_0923", "20260923_055425_real_robot_record_external_lz4",
          "20260923_060415_real_robot_record_external_lz4", ML + "/res__building_0923__query_060415.npz"),
}
K = int(os.environ.get("SIM3_K", "3"))
USE_BM3 = os.environ.get("SIM3_BM3", "0") == "1"

def main(key):
    import torch
    root, db, q, mlf = SC[key]
    order = np.load(mlf)["order"]
    dbn = sorted(n[:-10] for n in os.listdir(f"{root}/{db}/seq-01") if n.endswith(".color.png"))
    qn = sorted(n[:-10] for n in os.listdir(f"{root}/{q}/seq-01") if n.endswith(".color.png"))
    device = torch.device("cuda")
    model = setup_reloc3r_relpose_model("Reloc3rRelpose(img_size=512)", device)
    visloc = Reloc3rVisloc()
    db_imgs = load_images([f"{root}/{db}/seq-01/{n}.color.png" for n in dbn], size=512, verbose=False)
    q_imgs = load_images([f"{root}/{q}/seq-01/{n}.color.png" for n in qn], size=512, verbose=False)
    pose = lambda s, n: np.loadtxt(f"{root}/{s}/seq-01/{n}.pose.txt")

    T, R, t_infer = [], [], 0.0
    for qi in range(len(qn)):
        sel = [int(j) for j in order[qi, :K]]
        n = len(sel)
        v1 = {"img": torch.cat([db_imgs[j]["img"] for j in sel], 0),
              "true_shape": torch.as_tensor(np.stack([db_imgs[j]["true_shape"] for j in sel])).reshape(n, 2)}
        v2 = {"img": q_imgs[qi]["img"].repeat(n, 1, 1, 1),
              "true_shape": torch.as_tensor(q_imgs[qi]["true_shape"]).repeat(n, 1)}
        t0 = time.time()
        with torch.no_grad():
            q2d = inference_relpose([v1, v2], model, device, use_amp=True)
        torch.cuda.synchronize()
        t_infer += time.time() - t0
        pd = [pose(db, dbn[j]) for j in sel]
        pq = [p.cpu().numpy() for p in q2d]
        if USE_BM3:
            cand_t = np.array([(pd[kk] @ pq[kk])[:3, 3] for kk in range(n)])
            rough = np.median(cand_t, axis=0)
            keep = [kk for kk in range(n) if np.linalg.norm(pd[kk][:3, 3] - rough) < 3.0]
            if len(keep) < 2:
                keep = list(range(n))
            pd, pq = [pd[i] for i in keep], [pq[i] for i in keep]
        Rt = visloc.motion_averaging(pd, pq)
        gt = pose(q, qn[qi])
        T.append(float(np.linalg.norm(Rt[:3, 3] - gt[:3, 3])))
        R.append(get_rot_err(Rt[:3, :3], gt[:3, :3]))
    T, R = np.array(T), np.array(R)
    np.savez(f"{OUT}/res_{key}_sim{K}{'_bm3' if USE_BM3 else ''}.npz",
             terr=T, rerr=R, names=np.array(qn))
    print(f"{key} sim{K}{'_bm3' if USE_BM3 else ''}: n={len(T)} "
          f"terr mean/med/p90 = {T.mean():.2f}/{np.median(T):.2f}/{np.percentile(T,90):.2f} | "
          f"rerr mean/med/p90 = {R.mean():.2f}/{np.median(R):.2f}/{np.percentile(R,90):.2f} | "
          f"relpose {(t_infer/len(qn))*1000:.0f} ms/帧")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main(sys.argv[1])

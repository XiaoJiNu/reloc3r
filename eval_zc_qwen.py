"""ZC 数据集 Reloc3r 评测（qwen 会话版，修正外参链）。

与 glm 分支 eval_zc.py 的区别（本会话实测结论）：
  /map 位姿的体坐标系是 x前/y左/z上（实测：体z与map竖直轴点积恒 0.999，体x=行进方向 p50 3.1°），
  而 hesai 雷达系是 x前/y右/z下（imu_calib 重力实测 + 鱼眼地标方位实测），两者差绕前进轴 180° 滚转。
  相机链：T_map_cam = T_map_body @ diag(1,-1,-1) @ inv(T_lidar_cam)。
  旁证：camera_01 光轴=-y_lidar=车左（侧装），与 2m/s 直行时纯横向光流（|d|p50=37px, dy≈0, 无扩张）
  及画面地平线高于中心（下俯~10.4°）一致；glm 版把 /map 当雷达系，旋转 GT 差 180° 滚转。

复用 glm 版已提取的去畸变图片与 NetVLAD 检索 pair（均与外参链无关）：
  data/zc/<bag>/seq-01/frame-*.color.png          图片（symlink 进 data/zc_fix/）
  _db-q_pair_info/zc_<qbag>_q-retrieval_db-step=1_topk=10.npy
子命令: fixpos / sanity / eval --q <bag>
"""
import argparse
import json
import os

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True

DATA_ROOT = './data/zc'
FIX_ROOT = './data/zc_fix'
PAIR_DIR = './_db-q_pair_info'
CALIB = '/data/data/location_data/zc_data/研发大楼/022419/calib/hesai_022419'
BAGS = {'db': '20260921_064638', 'q_same_day': '20260921_062224', 'q_cross_day': '20260920_081533'}
TOPK = 10
# lidar 系 -> /map 体系（body = x前/y左/z上，实测：体x=行进3.1°、体z=map竖直0.999）。
# 定稿 R_BL（列=雷达轴在体系中的方向），四路独立证据：
#  (1) 标定方注记(imu_calib.json, 0921包四路互证): 鱼眼 rig front(车前)≈雷达-y, right(车右)≈雷达+x
#      => x_l=车右, y_l=车后, z_l=下(重力实测)
#  (2) 杆臂物理性: 相机在雷达系 (0.01,-1.42,1.01) m => 体系 = 雷达前1.42m/下1.01m = 前保险杠
#      低置相机(离地~0.6m) ✓; 若 x_l=车前 则相机在车体外侧 1.42 m(车宽<0.6m), 不可能
#  (3) 光轴: z_cam = -y_l = 车前, 下俯10.4° => 画面地平线应在中心上方 ~10° ✓ (实测35%高度)
#  (4) sanity 判别器(大转角帧对): 本链与镜像候选旋转误差相同(yaw运动共轭不变), 但 OLD(glm)
#      恒等链 = 相机倒置(cam-down·up=+0.98) 旋转误差 87° 被否
R_BL = np.eye(4)
R_BL[:3, :3] = [[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]  # x_l=右, y_l=后, z_l=下


def bl_candidates():
    return {
        'C1 x_l=前,y_l=右': np.diag([1.0, -1.0, -1.0]),
        'C3 x_l=右,y_l=后 (定稿)': R_BL[:3, :3],
        'OLD(glm) 恒等(/map=雷达系, 相机倒置)': np.eye(3),
    }


def T_lidar_cam():
    txt = open(os.path.join(CALIB, 'image_02', 'lidar_to_image_02.txt')).read()
    v = {}
    for line in txt.strip().splitlines():
        k, *r = line.replace('\r', '').split(':')
        v[k.strip()] = np.array([float(x) for x in r[0].split()])
    T = np.eye(4)
    T[:3, :3] = v['R'].reshape(3, 3)
    T[:3, 3] = v['T']
    return T  # p_cam = T @ p_lidar


def fix_poses(bags):
    """从旧 pose.txt 反解 T_map_body（旧链 = T_mb @ inv(T_lc)），重算修正链，图片 symlink。"""
    T_lc = T_lidar_cam()
    T_cl = np.linalg.inv(T_lc)
    for b in bags:
        d = f'{DATA_ROOT}/{b}/seq-01'
        o = f'{FIX_ROOT}/{b}/seq-01'
        os.makedirs(o, exist_ok=True)
        n = 0
        for f in sorted(os.listdir(d)):
            if not f.endswith('.pose.txt'):
                continue
            p_old = np.loadtxt(f'{d}/{f}')
            t_mb = p_old @ T_lc                      # 旧链的逆
            np.savetxt(f'{o}/{f}', t_mb @ R_BL @ T_cl)
            img = f.replace('.pose.txt', '.color.png')
            if not os.path.exists(f'{o}/{img}'):
                os.symlink(os.path.abspath(f'{d}/{img}'), f'{o}/{img}')
            n += 1
        print(f'{b}: {n} poses fixed -> {o}')


def frame_names(bag):
    return sorted(n[:-len('.color.png')] for n in os.listdir(f'{FIX_ROOT}/{bag}/seq-01')
                  if n.endswith('.color.png'))


def load_pose(bag, name):
    return np.loadtxt(f'{FIX_ROOT}/{bag}/seq-01/{name}.pose.txt')


def _model_and_images(db_names, q_names, db_bag, q_bag, device):
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model
    from reloc3r.utils.image import load_images
    model = setup_reloc3r_relpose_model('Reloc3rRelpose(img_size=512)', device)
    db_imgs = load_images([f'{FIX_ROOT}/{db_bag}/seq-01/{n}.color.png' for n in db_names],
                          size=512, verbose=False)
    q_imgs = load_images([f'{FIX_ROOT}/{q_bag}/seq-01/{n}.color.png' for n in q_names],
                         size=512, verbose=False)
    return model, db_imgs, q_imgs


def _to_dev(v, device):
    return {'img': v['img'].to(device), 'true_shape': torch.as_tensor(v['true_shape']).to(device)}


def cmd_sanity(args):
    """大转角判别：db 环线上相距 >90° 航向的两帧，模型 relpose（只依赖图像）vs
    各候选链条的 GT relpose。旋转角在共轭下不变、轴会变 -> 错误链条直接暴露。
    （小步长帧对无判别力：Δyaw~0.5°、平移无米制尺度。）"""
    from reloc3r.reloc3r_relpose import inference_relpose
    from reloc3r.utils.device import to_numpy
    from reloc3r.utils.image import load_images
    import cv2, math
    device = torch.device('cuda')
    db = BAGS['db']
    all_names = frame_names(db)
    T_lc = T_lidar_cam()
    T_mb_all = [np.loadtxt(f'{DATA_ROOT}/{db}/seq-01/{n}.pose.txt') @ T_lc for n in all_names]
    yaw = [math.atan2(t[1, 0], t[0, 0]) for t in T_mb_all]
    pairs = []
    for a in range(0, len(all_names), 40):
        for b in range(a + 20, len(all_names), 20):
            dyaw = abs((yaw[b] - yaw[a] + math.pi) % (2 * math.pi) - math.pi)
            if dyaw > math.radians(90):
                pairs.append((a, b, math.degrees(yaw[b] - yaw[a])))
                break
        if len(pairs) >= 6:
            break
    print(f'{len(pairs)} big-step pairs, yaw diffs: {[round(p[2]) for p in pairs]}')
    imgs = load_images([f'{DATA_ROOT}/{db}/seq-01/{all_names[i]}.color.png'
                        for a, b, _ in pairs for i in (a, b)], size=512, verbose=False)
    model = setup_model(device) if False else None
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model
    model = setup_reloc3r_relpose_model('Reloc3rRelpose(img_size=512)', device)
    preds = []
    for k in range(len(pairs)):
        v1, v2 = imgs[2 * k], imgs[2 * k + 1]
        with torch.no_grad():
            preds.append(to_numpy(inference_relpose([_to_dev(v1, device), _to_dev(v2, device)],
                                                    model, device, use_amp=True))[0])
    for tag, Rb in bl_candidates().items():
        M = np.eye(4); M[:3, :3] = Rb
        poses = [t @ M @ np.linalg.inv(T_lc) for t in T_mb_all]
        rr = []
        for k, (a, b, _) in enumerate(pairs):
            gt = np.linalg.inv(poses[a]) @ poses[b]
            rerr = np.linalg.norm(cv2.Rodrigues(preds[k][:3, :3].T @ gt[:3, :3])[0]) * 180 / np.pi
            rr.append(rerr)
        dn = np.mean([poses[i][2, 1] for i in range(0, len(poses), 50)])  # cam-down·map-up
        print(f'{tag}: rot err med {np.median(rr):6.2f}° (per pair: {[f"{x:.1f}" for x in rr]})  cam-down·up={dn:+.2f}')
    # 平移方向角判别（相邻帧，模型 t 无尺度但方向可信，MegaDepth AUC@30=81% 级别）
    step_names = all_names[400:412]
    step_imgs = load_images([f'{DATA_ROOT}/{db}/seq-01/{n}.color.png' for n in step_names], size=512, verbose=False)
    sp = [T_mb_all[400 + i] for i in range(len(step_names))]
    spreds = []
    for k in range(1, len(step_names)):
        v1, v2 = step_imgs[k - 1], step_imgs[k]
        with torch.no_grad():
            spreds.append(to_numpy(inference_relpose([_to_dev(v1, device), _to_dev(v2, device)],
                                                     model, device, use_amp=True))[0])
    for tag, Rb in bl_candidates().items():
        M = np.eye(4); M[:3, :3] = Rb
        poses = [t @ M @ np.linalg.inv(T_lc) for t in sp]
        ang = []
        for k in range(1, len(poses)):
            gt = np.linalg.inv(poses[k - 1]) @ poses[k]
            a = gt[:3, 3] / np.linalg.norm(gt[:3, 3])
            b = spreds[k - 1][:3, 3] / np.linalg.norm(spreds[k - 1][:3, 3])
            ang.append(math.degrees(math.acos(max(-1, min(1, float(a @ b))))))
        print(f'{tag}: STEP translation direction err med {np.median(ang):5.1f}° (pairs {[f"{x:.0f}" for x in ang]})')


def cmd_eval(args):
    from reloc3r.reloc3r_relpose import inference_relpose
    from reloc3r.reloc3r_visloc import Reloc3rVisloc
    from reloc3r.utils.metric import get_rot_err
    device = torch.device('cuda')
    db = BAGS['db']
    db_names = frame_names(db)
    pairs = np.load(f'{PAIR_DIR}/zc_{args.q}_q-retrieval_db-step=1_topk={TOPK}.npy', allow_pickle=True)
    q_names = frame_names(args.q)
    # 检索 pair 文件按 query 顺序（与当年 names_color 一致）
    assert len(pairs) == len(q_names), f'{len(pairs)} vs {len(q_names)}'
    model, db_imgs, q_imgs = _model_and_images(db_names, q_names, db, args.q, device)
    visloc = Reloc3rVisloc()

    terrs, rerrs, hit, pred, gtlist = [], [], [], [], []
    pr, pt = [], []
    cslist, pdlist = [], []
    for qi in range(len(q_names)):
        sel = [p['db_idx'] for p in pairs[qi]]
        view1 = {'img': torch.cat([db_imgs[j]['img'] for j in sel], 0),
                 'true_shape': torch.as_tensor(np.stack([db_imgs[j]['true_shape'] for j in sel])).reshape(len(sel), 2)}
        view2 = {'img': q_imgs[qi]['img'].repeat(len(sel), 1, 1, 1),
                 'true_shape': torch.as_tensor(q_imgs[qi]['true_shape']).repeat(len(sel), 1)}
        with torch.no_grad():
            q2d = inference_relpose([view1, view2], model, device, use_amp=not args.fp32)
        pd = [load_pose(db, db_names[j]) for j in sel]
        pq = [p.cpu().numpy() for p in q2d]
        gt = load_pose(args.q, q_names[qi])
        for k in range(len(sel)):
            cand = pd[k] @ pq[k]
            pr.append(get_rot_err(cand[:3, :3], gt[:3, :3]))
            pt.append(float(np.linalg.norm(cand[:3, 3] - gt[:3, 3])))
        if args.base_max > 0:
            # 粗位姿 = 候选平移中位数（不依赖 GT），只保留与粗位姿基线 < base_max 的候选
            cand_t = np.array([(pd[k] @ pq[k])[:3, 3] for k in range(len(sel))])
            rough = np.median(cand_t, axis=0)
            keep = [k for k in range(len(sel))
                    if np.linalg.norm(pd[k][:3, 3] - rough) < args.base_max]
            if len(keep) < 2:
                keep = list(range(len(sel)))  # 退化则全保留
            Rt = visloc.motion_averaging([pd[k] for k in keep], [pq[k] for k in keep])
        else:
            Rt = visloc.motion_averaging(pd, pq)
        terrs.append(float(np.linalg.norm(Rt[:3, 3] - gt[:3, 3])))
        rerrs.append(get_rot_err(Rt[:3, :3], gt[:3, :3]))
        hit.append(any(np.linalg.norm(pd[k][:3, 3] - gt[:3, 3]) < 2.0 for k in range(len(sel))))
        pred.append(Rt); gtlist.append(gt)
        if args.dump:
            cslist.append(np.stack([pd[k] @ pq[k] for k in range(len(sel))]))
            pdlist.append(np.stack([pd[k] for k in range(len(sel))]))
    terrs, rerrs = np.array(terrs), np.array(rerrs)
    m = {'n': len(terrs),
         'terr_m': {'median': float(np.median(terrs)), 'mean': float(terrs.mean()), 'p90': float(np.percentile(terrs, 90))},
         'rerr_deg': {'median': float(np.median(rerrs)), 'mean': float(rerrs.mean()), 'p90': float(np.percentile(rerrs, 90))},
         'succ_t': {f'<{t}m': float((terrs < t).mean()) for t in (0.25, 0.5, 1.0)},
         'succ_r': {f'<{r}deg': float((rerrs < r).mean()) for r in (5, 10, 20)},
         'succ_joint': {'t<0.5&r<10': float(((terrs < 0.5) & (rerrs < 10)).mean()),
                        't<1&r<20': float(((terrs < 1.0) & (rerrs < 20)).mean())},
         'hit2m': float(np.mean(hit)),
         'perpair': {'rot_med': float(np.median(pr)), 'rot_p90': float(np.percentile(pr, 90)),
                     'terr_med': float(np.median(pt)), 'terr_p90': float(np.percentile(pt, 90))}}
    print(json.dumps(m, indent=1))
    tag = args.q + (f'__basemax{args.base_max:g}' if args.base_max > 0 else '')
    np.savez(f'{FIX_ROOT}/res_{tag}.npz', terr=terrs, rerr=rerrs, hit=np.array(hit),
             pred=np.array(pred), gt=np.array(gtlist), names=np.array(q_names),
             perpair_r=np.array(pr), perpair_t=np.array(pt))
    with open(f'{FIX_ROOT}/metrics_{tag}.json', 'w') as f:
        json.dump(m, f, indent=1)
    print(f'-> {FIX_ROOT}/res_{tag}.npz')
    if args.dump:
        np.savez(f'{FIX_ROOT}/cands_{args.q}.npz',
                 cs=np.array(cslist), pdb=np.array(pdlist),  # (n,10,4,4)
                 gt=np.array(gtlist), names=np.array(q_names))
        print(f'-> {FIX_ROOT}/cands_{args.q}.npz (for offline aggregation sweep)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    fp = sub.add_parser('fixpos')
    fp.add_argument('--bags', default=','.join(BAGS.values()))
    sub.add_parser('sanity')
    ep = sub.add_parser('eval')
    ep.add_argument('--q', required=True)
    ep.add_argument('--fp32', action='store_true')
    ep.add_argument('--base-max', dest='base_max', type=float, default=0.0,
                    help='>0: only average candidates whose db pose is within this baseline (m) of the median candidate position')
    ep.add_argument('--dump', action='store_true',
                    help='save per-query candidate poses (cs, pd) npz for offline aggregation sweeps (zero extra inference)')
    a = ap.parse_args()
    if a.cmd == 'fixpos':
        fix_poses(a.bags.split(','))
    elif a.cmd == 'sanity':
        cmd_sanity(a)
    else:
        cmd_eval(a)

"""Reloc3r relocalization test on the ZC self-collected mcap dataset.

Subcommands:
    extract   mcap -> undistorted png + camera c2w pose txt (map frame, metric)
    retrieve  NetVLAD db descriptors + query top-k pair file
    eval      per-pair relpose regression -> motion averaging -> error metrics
    sanity    self-check: adjacent-db-frame relpose vs /map-derived GT displacement

Protocol: docs/zc数据集上测试/测试方案_glm-5.3flash_202609281412.md
"""
import argparse
import json
import os

import cv2
import numpy as np

BAGS = {  # role: bag stem
    'db': '20260921_064638',
    'q_same_day': '20260921_062224',
    'q_cross_day': '20260920_081533',
}
QUERY_ROLES = ('q_same_day', 'q_cross_day')
BAG_DIR = '/data/data/reLocationData/zc_data'
CALIB_DIR = '/data/data/location_data/zc_data/研发大楼/022419/calib/hesai_022419'
DATA_ROOT = './data/zc'
IMG_STEP = 5          # every 5th 10Hz frame -> 2 Hz
RESOLUTION = (512, 384)
TOPK = 10
SYNC_MAX_MS = 100.0   # drop frames whose nearest /map sample is farther than this


def read_calib_T_lidar_cam():
    """lidar->image_02 extrinsic from the authoritative calib dir (Kitti R:/T: format)."""
    txt = open(os.path.join(CALIB_DIR, 'image_02', 'lidar_to_image_02.txt')).read()
    vals = {}
    for line in txt.strip().splitlines():
        k, *v = line.replace('\r', '').split(':')
        vals[k.strip()] = np.array([float(x) for x in v[0].split()])
    T = np.eye(4)
    T[:3, :3] = vals['R'].reshape(3, 3)
    T[:3, 3] = vals['T']
    return T  # p_cam = T @ p_lidar


def load_poses(stem):
    """All written (name -> T_map_cam) of a scene, in name order."""
    d = os.path.join(DATA_ROOT, stem, 'seq-01')
    out = {}
    for n in sorted(os.listdir(d)):
        if n.endswith('.pose.txt'):
            out[n.replace('.pose.txt', '')] = np.loadtxt(os.path.join(d, n))
    return out


# ---------------------------------------------------------------- extract
def cmd_extract(args):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    T_lidar_cam = read_calib_T_lidar_cam()
    T_cam_lidar = np.linalg.inv(T_lidar_cam)

    K = D = None
    stats = {}
    for role, stem in BAGS.items():
        bag = os.path.join(BAG_DIR, stem + '.mcap')
        outdir = os.path.join(DATA_ROOT, stem, 'seq-01')
        os.makedirs(outdir, exist_ok=True)
        map_t, map_p, map_q = [], [], []
        with open(bag, 'rb') as f:
            r = make_reader(f, decoder_factories=[DecoderFactory()])
            for s, ch, msg, dec in r.iter_decoded_messages(
                    topics=['/map', '/camera_01/color/camera_info']):
                if ch.topic == '/map':
                    map_t.append(msg.log_time)
                    p = dec.pose.pose.position
                    q = dec.pose.pose.orientation
                    map_p.append((p.x, p.y, p.z))
                    map_q.append((q.x, q.y, q.z, q.w))
                elif K is None:
                    K = np.array(dec.k, dtype=np.float64).reshape(3, 3)
                    D = np.array(dec.d, dtype=np.float64)
        map_t = np.array(map_t)
        map_p = np.array(map_p)
        map_q = np.array(map_q)
        print(f'{stem}: /map samples {len(map_t)}')

        newmap = cv2.initUndistortRectifyMap(K, D, None, K, (640, 480), cv2.CV_32FC1)
        n_img = written = dropped = 0
        sync_resid = []
        with open(bag, 'rb') as f:
            r = make_reader(f, decoder_factories=[DecoderFactory()])
            for s, ch, msg, dec in r.iter_decoded_messages(
                    topics=['/camera_01/color/image_raw']):
                n_img += 1
                if (n_img - 1) % IMG_STEP:
                    continue
                i = int(np.clip(np.searchsorted(map_t, msg.log_time), 1, len(map_t) - 1))
                j = i if abs(map_t[i] - msg.log_time) <= abs(map_t[i - 1] - msg.log_time) else i - 1
                dt_ms = abs(map_t[j] - msg.log_time) / 1e6
                if dt_ms > SYNC_MAX_MS:
                    dropped += 1
                    continue
                sync_resid.append(dt_ms)
                raw = np.frombuffer(bytes(dec.data), np.uint8).reshape(480, 640, 3)
                und = cv2.remap(raw, newmap[0], newmap[1], cv2.INTER_LINEAR)
                name = f'frame-{written:06d}'
                cv2.imwrite(os.path.join(outdir, name + '.color.png'),
                            cv2.cvtColor(und, cv2.COLOR_RGB2BGR))
                T_map_lidar = np.eye(4)
                from scipy.spatial.transform import Rotation as Rot
                T_map_lidar[:3, :3] = Rot.from_quat(map_q[j]).as_matrix()
                T_map_lidar[:3, 3] = map_p[j]
                np.savetxt(os.path.join(outdir, name + '.pose.txt'),
                           T_map_lidar @ T_cam_lidar)
                written += 1

        sr = np.array(sync_resid)
        stats[role] = {'bag': stem, 'n_img': n_img, 'written': written,
                       'dropped_sync': dropped,
                       'sync_ms': {'p50': float(np.percentile(sr, 50)),
                                   'p95': float(np.percentile(sr, 95)),
                                   'max': float(sr.max())}}
        print(f"  -> {written} frames ({dropped} dropped), sync p50={np.percentile(sr, 50):.1f}"
              f" p95={np.percentile(sr, 95):.1f} max={sr.max():.1f} ms")

    # trajectory overlay plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    for role, stem in BAGS.items():
        ps = np.array([p[:2, 3] for p in load_poses(stem).values()])
        lbl = {'db': 'db (064638)', 'q_same_day': 'query same-day (062224)',
               'q_cross_day': 'query cross-day (081533)'}[role]
        ax.plot(ps[:, 0], ps[:, 1], '-', lw=1, label=lbl)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlabel('map x [m]'); ax.set_ylabel('map y [m]')
    ax.set_title('ZC extracted trajectories (camera positions, map frame)')
    out_png = 'docs/zc数据集上测试/tools/zc_extracted_traj.png'
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    stats['traj_plot'] = out_png
    stats['K'] = K.tolist()
    stats['D'] = D.tolist()
    stats['T_lidar_cam'] = T_lidar_cam.tolist()
    with open('docs/zc数据集上测试/tools/extract_stats.json', 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print('stats -> docs/zc数据集上测试/tools/extract_stats.json')


# ---------------------------------------------------------------- retrieve
def cmd_retrieve(args):
    import torch
    from reloc3r.image_retrieval.topk_retrieval import TopkRetrieval
    from reloc3r.datasets.zc_retrieval import ZcRetrieval

    class A:  # minimal cfg shim for TopkRetrieval
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cache_folder = './_db-q_pair_info'
        db_descs_file_mask = 'zc_{}_db-descs_step={}.txt'
        pair_info_file_mask = 'zc_{}_q-retrieval_db-step={}_topk={}.npy'
        db_step = 1
        topk = TOPK
        load_bd_desc = False
    os.makedirs(A.cache_folder, exist_ok=True)

    dataset_db = ZcRetrieval(scene=BAGS['db'], split='train')
    runner = TopkRetrieval(A)
    runner.build_database(dataset_db)
    for role in QUERY_ROLES:
        q_scene = BAGS[role]
        dataset_q = ZcRetrieval(scene=q_scene, split='test')
        all_retrieved = runner.retrieve_topk(dataset_db, dataset_q)
        pair_path = os.path.join(A.cache_folder, A.pair_info_file_mask).format(q_scene, 1, TOPK)
        np.save(pair_path, all_retrieved, allow_pickle=True)
        print(f'{role}: pairs -> {pair_path}')


# ---------------------------------------------------------------- sanity
def cmd_sanity(args):
    """Adjacent db frames: Reloc3r relpose vs /map-chain GT relpose (both metric)."""
    import torch
    from PIL import Image
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
    from reloc3r.utils.image import load_images
    from reloc3r.utils.device import to_numpy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = setup_reloc3r_relpose_model('Reloc3rRelpose(img_size=512)', device)
    d = os.path.join(DATA_ROOT, BAGS['db'], 'seq-01')
    names = sorted(n for n in os.listdir(d) if n.endswith('.color.png'))[:12]
    imgs = load_images([os.path.join(d, n) for n in names], size=RESOLUTION[0])
    poses = load_poses(BAGS['db'])
    errs_r, errs_t = [], []
    for k in range(1, len(imgs)):
        def tv(v):
            import torch
            img, ts = v['img'], v['true_shape']
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(device)
                ts = torch.from_numpy(np.asarray(ts)[None]).to(device)
            elif not torch.is_tensor(img):
                raise TypeError(f'bad img type {type(img)}')
            else:
                img = img[None] if img.dim() == 3 else img
                ts = ts if torch.is_tensor(ts) else torch.from_numpy(np.asarray(ts))
                img, ts = img.to(device), ts.to(device)
            return {'img': img, 'true_shape': ts}
        v1, v2 = tv(imgs[k - 1]), tv(imgs[k])
        batch = [v1, v2]
        q2d = to_numpy(inference_relpose(batch, model, device, use_amp=True))[0]
        n1 = names[k - 1].replace('.color.png', '')
        n2 = names[k].replace('.color.png', '')
        gt = np.linalg.inv(poses[n1]) @ poses[n2]
        rerr = np.linalg.norm(cv2.Rodrigues(q2d[:3, :3].T @ gt[:3, :3])[0]) / np.pi * 180
        terr = np.linalg.norm(q2d[:3, 3] - gt[:3, 3])
        errs_r.append(rerr); errs_t.append(terr)
        print(f'{n1}->{n2}: rot {rerr:6.2f} deg  trans {terr*100:6.1f} cm '
              f'(gt step {np.linalg.norm(gt[:3,3])*100:.0f} cm)')
    print(f'\nadjacent-frame: rot median {np.median(errs_r):.2f} deg, '
          f'trans median {np.median(errs_t)*100:.1f} cm')
    assert np.median(errs_t) < 1.0, 'translation sanity FAILED: check extrinsic chain'
    assert np.median(errs_r) < 10.0, 'rotation sanity FAILED: check extrinsic chain'
    print('sanity PASSED')


# ---------------------------------------------------------------- eval
def cmd_eval(args):
    import torch
    from eval_relpose import build_dataset
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
    from reloc3r.reloc3r_visloc import Reloc3rVisloc
    from reloc3r.utils.device import to_numpy
    from reloc3r.utils.metric import get_rot_err
    from reloc3r.datasets.zc_retrieval import ZcRelpose

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = setup_reloc3r_relpose_model('Reloc3rRelpose(img_size=512)', device)
    visloc = Reloc3rVisloc()
    db_scene = BAGS['db']
    db_dir = os.path.join(DATA_ROOT, db_scene, 'seq-01')
    db_names = sorted(os.listdir(db_dir))

    out = {}
    per_pair = {}
    for role in QUERY_ROLES:
        q_scene = BAGS[role]
        db_poses, q2ds, gts, q_insts = {}, {}, {}, {}
        for pair_id in range(TOPK):
            ds = ZcRelpose(scene=db_scene, query_scene=q_scene, pair_id=pair_id,
                           resolution=RESOLUTION, topk=TOPK)
            loader = build_dataset(ds, args.batch_size, args.num_workers, test=True)
            for batch in loader:
                pose_q2d = to_numpy(inference_relpose(batch, model, device, use_amp=True))
                view1, view2 = batch
                for sid in range(len(pose_q2d)):
                    inst = view2['instance'][sid]
                    db_inst = view1['instance'][sid]
                    q_insts[inst] = True
                    db_poses.setdefault(inst, {})[pair_id] = to_numpy(view1['camera_pose'][sid])
                    q2ds.setdefault(inst, {})[pair_id] = pose_q2d[sid]
                    gts.setdefault(inst, {})['gt'] = to_numpy(view2['camera_pose'][sid])
        # pair diagnostics + motion averaging
        rerrs, terrs, pair_rerrs, pair_terrs = [], [], [], []
        pred_pose = {}
        for inst in q_insts:
            pd, pq, gt = db_poses[inst], q2ds[inst], gts[inst]['gt']
            pose_db_list = [pd[k] for k in range(TOPK)]
            pose_q2d_list = [pq[k] for k in range(TOPK)]
            for k in range(TOPK):
                cand = pose_db_list[k] @ pose_q2d_list[k]
                pair_rerrs.append(get_rot_err(cand[:3, :3], gt[:3, :3]))
                pair_terrs.append(float(np.linalg.norm(cand[:3, 3] - gt[:3, 3])))
            Rt = visloc.motion_averaging(pose_db_list, pose_q2d_list)
            rerrs.append(get_rot_err(Rt[:3, :3], gt[:3, :3]))
            terrs.append(float(np.linalg.norm(Rt[:3, 3] - gt[:3, 3])))
            pred_pose[inst] = Rt
        # retrieval hit rate: any top-10 db within 2 m (GT cam-center distance)
        hit = miss = 0
        for inst in q_insts:
            gt = gts[inst]['gt'][:3, 3]
            if any(np.linalg.norm(db_poses[inst][k][:3, 3] - gt) < 2.0 for k in range(TOPK)):
                hit += 1
            else:
                miss += 1
        rerrs, terrs = np.array(rerrs), np.array(terrs)
        out[role] = {
            'n_query': len(rerrs),
            'trans_m': {'median': float(np.median(terrs)), 'mean': float(terrs.mean()),
                        'p90': float(np.percentile(terrs, 90))},
            'rot_deg': {'median': float(np.median(rerrs)), 'mean': float(rerrs.mean()),
                        'p90': float(np.percentile(rerrs, 90))},
            'success_trans': {'<0.25m': float((terrs < 0.25).mean()),
                              '<0.5m': float((terrs < 0.5).mean()),
                              '<1m': float((terrs < 1.0).mean())},
            'retrieval_hit_2m': hit / (hit + miss),
        }
        per_pair[role] = {'rot_deg': pair_rerrs, 'trans_m': pair_terrs}
        np.save('docs/zc数据集上测试/tools/zc_predtraj_{}.npy'.format(role),
                np.array([pred_pose[k] for k in sorted(pred_pose)]))
        np.save('docs/zc数据集上测试/tools/zc_gttraj_{}.npy'.format(role),
                np.array([gts[k]['gt'] for k in sorted(pred_pose)]))
        print(f"{role}: n={len(rerrs)} trans med {np.median(terrs):.3f} m "
              f"rot med {np.median(rerrs):.2f} deg hit@2m={out[role]['retrieval_hit_2m']:.1%}")

    import pickle
    with open('docs/zc数据集上测试/tools/zc_eval_perpair.pkl', 'wb') as f:
        pickle.dump(per_pair, f)
    with open('docs/zc数据集上测试/tools/zc_eval_metrics.json', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, indent=1))
    print('metrics -> docs/zc数据集上测试/tools/zc_eval_metrics.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('extract')
    sub.add_parser('retrieve')
    ep = sub.add_parser('eval')
    ep.add_argument('--batch_size', type=int, default=10)
    ep.add_argument('--num_workers', type=int, default=8)
    sub.add_parser('sanity')
    args = p.parse_args()
    {'extract': cmd_extract, 'retrieve': cmd_retrieve,
     'eval': cmd_eval, 'sanity': cmd_sanity}[args.cmd](args)

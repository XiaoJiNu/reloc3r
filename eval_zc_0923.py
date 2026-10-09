"""ZC 研发大楼数据 20260923 两包（db=055425, q=060415）Reloc3r 评测 + 资源统计。

复用 eval_zc_qwen.py 的外参链与评测实现；新增：
  - 资源采样线程（nvidia-smi CSV + /proc），按阶段输出 GPU/CPU/内存占用与耗时
子命令: extract / retrieve / eval
"""
import argparse, json, os, subprocess, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_zc_qwen as ez

B0923 = '/data/data/reLocationData/zc_data/研发大楼数据'
BAGS9 = {'db': '20260923_055425_real_robot_record_external_lz4',
         'q': '20260923_060415_real_robot_record_external_lz4'}
ROOT9 = './data/zc_0923'
IMG_STEP = 5
SYNC_MAX_MS = 100.0


# ---------------- resource sampler ----------------
class Sampler(threading.Thread):
    """poll nvidia-smi + /proc every 0.2s; records per-sample rows."""
    def __init__(self):
        super().__init__(daemon=True)
        self.rows = []
        self.stop_flag = threading.Event()
        self.t0 = time.time()

    def run(self):
        while not self.stop_flag.is_set():
            gpu = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                 '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
            try:
                u, mu, pw = [float(x) for x in gpu.split(',')]
            except Exception:
                u = mu = pw = -1
            with open('/proc/stat') as f:
                c = [int(x) for x in f.readline().split()[1:]]
            with open('/proc/meminfo') as f:
                mi = {l.split(':')[0]: int(l.split()[1]) for l in f}
            load = os.getloadavg()[0]
            self.rows.append((time.time() - self.t0, u, mu, pw, sum(c), c[3] + c[4], load,
                              (mi['MemTotal'] - mi['MemAvailable']) / 1024.0))
            time.sleep(0.2)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2)
        return np.array(self.rows)


def stats(rows, name, dt):
    if len(rows) == 0:
        return {}
    cpu_pct = 100.0 * np.diff(rows[:, 5]) / np.maximum(np.diff(rows[:, 4]), 1e-9)
    return {'stage': name, 'wall_s': round(dt, 1),
            'gpu_util_mean': round(float(rows[:, 1].mean()), 1),
            'gpu_util_p95': round(float(np.percentile(rows[:, 1], 95)), 1),
            'gpu_mem_max_mib': round(float(rows[:, 2].max()), 0),
            'gpu_power_mean_w': round(float(rows[:, 3].mean()), 1),
            'cpu_sys_pct_mean': round(float(cpu_pct.mean()), 1),
            'load1_mean': round(float(rows[:, 6].mean()), 2),
            'host_mem_used_mean_gib': round(float(rows[:, 7].mean()), 1)}


# ---------------- extract (same chain as eval_zc_qwen) ----------------
def extract(bag_role):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    from scipy.spatial.transform import Rotation as Rot
    stem = BAGS9[bag_role]
    bag = f'{B0923}/{stem}/bag_0.mcap'
    outdir = f'{ROOT9}/{stem}/seq-01'
    os.makedirs(outdir, exist_ok=True)
    T_lc = ez.T_lidar_cam()
    map_t, map_p, map_q = [], [], []
    K = D = None
    with open(bag, 'rb') as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for s, ch, m, dec in r.iter_decoded_messages(topics=['/map', '/camera_01/color/camera_info']):
            if ch.topic == '/map':
                p = dec.pose.pose
                map_t.append(m.log_time)
                map_p.append((p.position.x, p.position.y, p.position.z))
                map_q.append((p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))
            elif K is None:
                K = np.array(dec.k, dtype=np.float64).reshape(3, 3)
                D = np.array(dec.d, dtype=np.float64)
    map_t = np.array(map_t)
    newmap = cv2map(K, D)
    n_img = written = dropped = 0
    resid = []
    import cv2
    with open(bag, 'rb') as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for s, ch, m, dec in r.iter_decoded_messages(topics=['/camera_01/color/image_raw']):
            n_img += 1
            if (n_img - 1) % IMG_STEP:
                continue
            i = int(np.clip(np.searchsorted(map_t, m.log_time), 1, len(map_t) - 1))
            j = i if abs(map_t[i] - m.log_time) <= abs(map_t[i - 1] - m.log_time) else i - 1
            dt_ms = abs(map_t[j] - m.log_time) / 1e6
            if dt_ms > SYNC_MAX_MS:
                dropped += 1
                continue
            resid.append(dt_ms)
            raw = np.frombuffer(bytes(dec.data), np.uint8).reshape(dec.height, dec.width, 3)
            und = cv2.remap(raw, newmap[0], newmap[1], cv2.INTER_LINEAR)
            name = f'frame-{written:06d}'
            cv2.imwrite(f'{outdir}/{name}.color.png', cv2.cvtColor(und, cv2.COLOR_RGB2BGR))
            T_mb = np.eye(4)
            T_mb[:3, :3] = Rot.from_quat(map_q[j]).as_matrix()
            T_mb[:3, 3] = map_p[j]
            np.savetxt(f'{outdir}/{name}.pose.txt', T_mb @ ez.R_BL @ np.linalg.inv(T_lc))
            written += 1
    resid = np.array(resid)
    print(f'{stem}: {written} imgs ({dropped} dropped), sync p50={np.median(resid):.1f}ms p95={np.percentile(resid,95):.1f}ms')


def cv2map(K, D):
    import cv2
    return cv2.initUndistortRectifyMap(K, D, None, K, (640, 480), cv2.CV_32FC1)


# ---------------- retrieve ----------------
class Ret9:
    def __init__(self, role):
        self.scene = BAGS9[role]
        self.names_color = [f'{ROOT9}/{self.scene}/seq-01/{n}'
                            for n in sorted(os.listdir(f'{ROOT9}/{self.scene}/seq-01'))
                            if n.endswith('.color.png')]

    def load_image(self, name, device):
        import cv2, torch
        c = cv2.imread(name)
        t = torch.Tensor(c).permute(2, 0, 1)[None].to(device)
        return {'image': (t - t.min()) / (t.max() - t.min())}


def retrieve():
    import torch
    from reloc3r.image_retrieval.topk_retrieval import TopkRetrieval

    class A:
        device = torch.device('cuda')
        cache_folder = './_db-q_pair_info'
        db_descs_file_mask = 'zc9_{}_db-descs_step={}.txt'
        pair_info_file_mask = 'zc9_{}_q-retrieval_db-step={}_topk={}.npy'
        db_step = 1
        topk = 10
        load_bd_desc = False
    runner = TopkRetrieval(A)
    db = Ret9('db')
    runner.build_database(db)
    q = Ret9('q')
    all_retrieved = runner.retrieve_topk(db, q)
    np.save(f'./_db-q_pair_info/zc9_{q.scene}_q-retrieval_db-step=1_topk=10.npy',
            all_retrieved, allow_pickle=True)
    print('pairs saved')


# ---------------- eval (reuse ez machinery with 0923 roots) ----------------
def eval_q(base_max):
    import torch
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
    from reloc3r.reloc3r_visloc import Reloc3rVisloc
    from reloc3r.utils.metric import get_rot_err
    from reloc3r.utils.image import load_images
    from reloc3r.utils.device import to_numpy
    device = torch.device('cuda')
    db_s, q_s = BAGS9['db'], BAGS9['q']
    db_names = sorted(n[:-10] for n in os.listdir(f'{ROOT9}/{db_s}/seq-01') if n.endswith('.color.png'))
    q_names = sorted(n[:-10] for n in os.listdir(f'{ROOT9}/{q_s}/seq-01') if n.endswith('.color.png'))
    pairs = np.load(f'./_db-q_pair_info/zc9_{q_s}_q-retrieval_db-step=1_topk=10.npy', allow_pickle=True)
    model = setup_reloc3r_relpose_model('Reloc3rRelpose(img_size=512)', device)
    visloc = Reloc3rVisloc()
    db_imgs = load_images([f'{ROOT9}/{db_s}/seq-01/{n}.color.png' for n in db_names], size=512, verbose=False)
    q_imgs = load_images([f'{ROOT9}/{q_s}/seq-01/{n}.color.png' for n in q_names], size=512, verbose=False)

    def pose(s, n):
        return np.loadtxt(f'{ROOT9}/{s}/seq-01/{n}.pose.txt')

    out = {}
    acc = {mode: {'terrs': [], 'rerrs': [], 'hit': [], 'pred': [], 'gt': []} for mode in ([0, base_max] if base_max else [0])}
    t_infer = 0.0
    for qi in range(len(q_names)):
        sel = [p['db_idx'] for p in pairs[qi]]
        view1 = {'img': torch.cat([db_imgs[j]['img'] for j in sel], 0),
                 'true_shape': torch.as_tensor(np.stack([db_imgs[j]['true_shape'] for j in sel])).reshape(len(sel), 2)}
        view2 = {'img': q_imgs[qi]['img'].repeat(len(sel), 1, 1, 1),
                 'true_shape': torch.as_tensor(q_imgs[qi]['true_shape']).repeat(len(sel), 1)}
        t0 = time.time()
        with torch.no_grad():
            q2d = inference_relpose([view1, view2], model, device, use_amp=True)
        torch.cuda.synchronize()
        t_infer += time.time() - t0
        pd = [pose(db_s, db_names[j]) for j in sel]
        pq = [p.cpu().numpy() for p in q2d]
        gt = pose(q_s, q_names[qi])
        for mode in acc:
            if mode:
                cand_t = np.array([(pd[k] @ pq[k])[:3, 3] for k in range(len(sel))])
                rough = np.median(cand_t, axis=0)
                keep = [k for k in range(len(sel)) if np.linalg.norm(pd[k][:3, 3] - rough) < mode]
                if len(keep) < 2:
                    keep = list(range(len(sel)))
                Rt = visloc.motion_averaging([pd[k] for k in keep], [pq[k] for k in keep])
            else:
                Rt = visloc.motion_averaging(pd, pq)
            acc[mode]['terrs'].append(float(np.linalg.norm(Rt[:3, 3] - gt[:3, 3])))
            acc[mode]['rerrs'].append(get_rot_err(Rt[:3, :3], gt[:3, :3]))
            acc[mode]['hit'].append(any(np.linalg.norm(pd[k][:3, 3] - gt[:3, 3]) < 2.0 for k in range(len(sel))))
            acc[mode]['pred'].append(Rt); acc[mode]['gt'].append(gt)
    for mode, d in acc.items():
        terrs, rerrs = np.array(d['terrs']), np.array(d['rerrs'])
        m = {'n': len(terrs),
             'terr_m': {'median': float(np.median(terrs)), 'mean': float(terrs.mean()), 'p90': float(np.percentile(terrs, 90))},
             'rerr_deg': {'median': float(np.median(rerrs)), 'p90': float(np.percentile(rerrs, 90))},
             'succ_t': {f'<{t}m': float((terrs < t).mean()) for t in (0.25, 0.5, 1.0)},
             'succ_r': {f'<{r}deg': float((rerrs < r).mean()) for r in (5, 10)},
             'hit2m': float(np.mean(d['hit'])),
             'relpose_infer_s_per_query': round(t_infer / len(q_names), 3)}
        out['top10' if not mode else f'basemax{mode:g}'] = m
        np.savez(f'{ROOT9}/res_0923_{"top10" if not mode else "bm"+str(mode)}.npz',
                 terr=terrs, rerr=rerrs, hit=np.array(d['hit']), pred=np.array(d['pred']),
                 gt=np.array(d['gt']), names=np.array(q_names))
        print(mode or 'top10', json.dumps(m, ensure_ascii=False))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('extract')
    sub.add_parser('retrieve')
    ep = sub.add_parser('eval')
    ep.add_argument('--base-max', type=float, default=3.0)
    a = ap.parse_args()
    smp = Sampler(); smp.start()
    t0 = time.time()
    res = None
    if a.cmd == 'extract':
        extract('db'); extract('q')
    elif a.cmd == 'retrieve':
        retrieve()
    else:
        res = eval_q(a.base_max)
    st = stats(smp.stop(), a.cmd, time.time() - t0)
    print('RESOURCE', json.dumps(st))
    with open(f'{ROOT9}/perf_{a.cmd}.json', 'w') as f:
        json.dump({'resource': st, 'result': res}, f, indent=1, ensure_ascii=False)

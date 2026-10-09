"""0923 结果：沿行进/横向/竖直误差分解（全帧+直线段，含 mean）；单帧推理耗时与资源基准。"""
import numpy as np, math, subprocess, threading, time, json, os, sys
R = 'data/zc_0923'

# ---------- 1) error decomposition ----------
def decomp(npz, name):
    r = np.load(npz, allow_pickle=True)
    gt = r['gt']; pred = r['pred']
    P = gt[:, :2, 3]
    e = pred[:, :3, 3] - gt[:, :3, 3]
    k = 5
    u = np.full((len(P), 2), np.nan)
    for i in range(len(P)):
        a, b = max(i - k, 0), min(i + k, len(P) - 1)
        d = P[b] - P[a]; n = np.linalg.norm(d)
        if n > 0.3:
            u[i] = d / n
    yaw = np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0]))
    rate = np.abs(np.array([(yaw[min(i+k, len(P)-1)] - yaw[max(i-k, 0)] + math.pi) % (2*math.pi) - math.pi
                            for i in range(len(P))])) / (2 * k)
    straight = (rate < math.radians(2)) & np.isfinite(u[:, 0])
    valid = np.isfinite(u[:, 0])
    rows = {}
    for tag, msk in [('all', valid), ('straight', straight)]:
        ua = u[msk]; ee = e[msk]
        al = ee[:, 0] * ua[:, 0] + ee[:, 1] * ua[:, 1]
        cr = -ee[:, 0] * ua[:, 1] + ee[:, 1] * ua[:, 0]
        vz = ee[:, 2]
        rows[tag] = {'n': int(msk.sum()),
                     'along': {'mean': float(al.mean()), 'med|e|': float(np.median(np.abs(al))), 'p90|e|': float(np.percentile(np.abs(al), 90))},
                     'cross': {'mean': float(cr.mean()), 'med|e|': float(np.median(np.abs(cr))), 'p90|e|': float(np.percentile(np.abs(cr), 90))},
                     'vert': {'mean': float(vz.mean()), 'med|e|': float(np.median(np.abs(vz))), 'p90|e|': float(np.percentile(np.abs(vz), 90))},
                     '3d': {'mean': float(np.linalg.norm(ee, axis=1).mean()), 'med': float(np.median(np.linalg.norm(ee, axis=1)))}}
    print(name, json.dumps(rows, ensure_ascii=False, indent=1))
    return rows

out = {}
for tag, f in [('top10', 'res_0923_top10.npz'), ('basemax3', 'res_0923_bm3.0.npz')]:
    out[tag] = decomp(f'{R}/{f}', tag)

# ---------- 2) single-frame latency + resource benchmark ----------
class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.rows = []; self.stop_flag = threading.Event()
    def run(self):
        while not self.stop_flag.is_set():
            g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
            try: u, mu, pw = [float(x) for x in g.split(',')]
            except Exception: u = mu = pw = -1
            self.rows.append((u, mu, pw))
            time.sleep(0.1)
    def stop(self):
        self.stop_flag.set(); self.join(timeout=2); return np.array(self.rows)

import torch
sys.path.insert(0, '.')
import eval_zc_qwen as ez
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.utils.image import load_images

device = torch.device('cuda')
model = setup_reloc3r_relpose_model('Reloc3rRelpose(img_size=512)', device)
db_s = '20260923_055425_real_robot_record_external_lz4'
q_s = '20260923_060415_real_robot_record_external_lz4'
db_names = sorted(n[:-10] for n in os.listdir(f'{R}/{db_s}/seq-01') if n.endswith('.color.png'))
q_names = sorted(n[:-10] for n in os.listdir(f'{R}/{q_s}/seq-01') if n.endswith('.color.png'))
rng = np.random.default_rng(3)
db_idx = rng.choice(len(db_names), 10, replace=False)
q_idx = rng.choice(len(q_names), 20, replace=False)
db_imgs = load_images([f'{R}/{db_s}/seq-01/{db_names[j]}.color.png' for j in db_idx], size=512, verbose=False)
q_imgs = load_images([f'{R}/{q_s}/seq-01/{q_names[j]}.color.png' for j in q_idx], size=512, verbose=False)

def bench(k, n_iter=20):
    lat = []
    for it in range(n_iter):
        qi = it % len(q_imgs)
        sel = list(range(k))
        view1 = {'img': torch.cat([db_imgs[j]['img'] for j in sel], 0),
                 'true_shape': torch.as_tensor(np.stack([db_imgs[j]['true_shape'] for j in sel])).reshape(k, 2)}
        view2 = {'img': q_imgs[qi]['img'].repeat(k, 1, 1, 1),
                 'true_shape': torch.as_tensor(q_imgs[qi]['true_shape']).repeat(k, 1)}
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            inference_relpose([view1, view2], model, device, use_amp=True)
        torch.cuda.synchronize(); lat.append(time.time() - t0)
    lat = np.array(lat[4:])  # drop warmup
    return {'k': k, 'ms_mean': round(lat.mean()*1e3, 1), 'ms_p50': round(np.median(lat)*1e3, 1),
            'ms_p95': round(np.percentile(lat, 95)*1e3, 1), 'ms_min': round(lat.min()*1e3, 1)}

bench_out = {}
smp = Sampler(); smp.start()
for k in (1, 4, 10):
    bench_out[f'top{k}'] = bench(k)
    print(f'top{k}:', bench_out[f'top{k}'], flush=True)
rows = smp.stop()
bench_out['resource_during_bench'] = {
    'gpu_util_mean': round(float(rows[:, 0].mean()), 1), 'gpu_util_p95': round(float(np.percentile(rows[:, 0], 95)), 1),
    'gpu_mem_max_mib': float(rows[:, 1].max()), 'gpu_power_mean_w': round(float(rows[:, 2].mean()), 1)}
print('resource:', bench_out['resource_during_bench'])
# motion averaging CPU time (single frame, k=10)
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from scipy.spatial.transform import Rotation as Rot
visloc = Reloc3rVisloc()
rng2 = np.random.default_rng(1)
pd = []; pq = []
for k in range(10):
    a = np.eye(4); a[:3, :3] = Rot.from_euler('z', rng2.normal(scale=0.2)).as_matrix(); a[:3, 3] = [k*0.6, rng2.normal(scale=0.5), 0]
    b = np.eye(4); b[:3, :3] = Rot.from_euler('xyz', rng2.normal(scale=0.05, size=3)).as_matrix(); b[:3, 3] = [0.3, 0.05*(k-5), 0.01]
    pd.append(a); pq.append(b)
t0 = time.perf_counter()
for _ in range(100):
    visloc.motion_averaging(pd, pq)
bench_out['motion_averaging_ms'] = round((time.perf_counter() - t0) / 100 * 1e3, 2)
print('motion_averaging (k=10, CPU):', bench_out['motion_averaging_ms'], 'ms')
cpu = subprocess.run(['nproc'], capture_output=True, text=True).stdout.strip()
print('nproc:', cpu)

json.dump({'decomposition': out, 'bench': bench_out}, open(f'{R}/decomp_bench_0923.json', 'w'),
          ensure_ascii=False, indent=1)
print('saved ->', f'{R}/decomp_bench_0923.json')

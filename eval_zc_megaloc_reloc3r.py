"""Reloc3r 检索段替换为 MegaLoc：端到端精度 + 耗时/资源对比。

检索 top-10 直接取 MegaLoc 评测已存的 argsort(order)（帧集与 reloc3r 完全相同、
排序一致，零重推理）；relpose 与运动平均复用 eval_zc_0923 的实现口径。
另外基准 MegaLoc 单帧描述子耗时（检索段成本）。

Run: /data/software/conda/anaconda3/envs/mv2d/bin/python eval_zc_megaloc_reloc3r.py
产物: data/zc_megaloc_reloc3r/{res_*.npz, perf_*.json}
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_zc_0923 as e9
import eval_zc_qwen as ez

ML = '/data/code/reLocation-all/MegaLoc/outputs/eval_zc_megaloc'
OUT = 'data/zc_megaloc_reloc3r'
SCENES = {
    '062224': {'root': './data/zc_fix', 'db': '20260921_064638', 'q': '20260921_062224',
               'ml': f'{ML}/res__mowing__same_day_062224.npz'},
    '081533': {'root': './data/zc_fix', 'db': '20260921_064638', 'q': '20260920_081533',
               'ml': f'{ML}/res__mowing__cross_day_081533.npz'},
    '0923': {'root': './data/zc_0923',
             'db': '20260923_055425_real_robot_record_external_lz4',
             'q': '20260923_060415_real_robot_record_external_lz4',
             'ml': f'{ML}/res__building_0923__query_060415.npz'},
}


def eval_scene(key, base_max=3.0):
    import torch
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
    from reloc3r.reloc3r_visloc import Reloc3rVisloc
    from reloc3r.utils.metric import get_rot_err
    from reloc3r.utils.image import load_images

    cfg = SCENES[key]
    root = cfg['root']
    order = np.load(cfg['ml'])['order']
    db_names = sorted(n[:-10] for n in os.listdir(f"{root}/{cfg['db']}/seq-01") if n.endswith('.color.png'))
    q_names = sorted(n[:-10] for n in os.listdir(f"{root}/{cfg['q']}/seq-01") if n.endswith('.color.png'))
    assert order.shape[1] >= len(db_names) and order.shape[0] == len(q_names) or order.shape[0] >= len(q_names), (order.shape, len(db_names), len(q_names))

    device = torch.device('cuda')
    model = setup_reloc3r_relpose_model('Reloc3rRelpose(img_size=512)', device)
    visloc = Reloc3rVisloc()
    t_ld = time.time()
    db_imgs = load_images([f"{root}/{cfg['db']}/seq-01/{n}.color.png" for n in db_names], size=512, verbose=False)
    q_imgs = load_images([f"{root}/{cfg['q']}/seq-01/{n}.color.png" for n in q_names], size=512, verbose=False)
    load_s = time.time() - t_ld

    def pose(s, n):
        return np.loadtxt(f"{root}/{s}/seq-01/{n}.pose.txt")

    modes = [0, base_max]
    acc = {m: {'t': [], 'r': [], 'hit': [], 'pred': [], 'gt': []} for m in modes}
    t_infer = 0.0
    for qi in range(len(q_names)):
        sel = [int(j) for j in order[qi, :10]]
        view1 = {'img': torch.cat([db_imgs[j]['img'] for j in sel], 0),
                 'true_shape': torch.as_tensor(np.stack([db_imgs[j]['true_shape'] for j in sel])).reshape(10, 2)}
        view2 = {'img': q_imgs[qi]['img'].repeat(10, 1, 1, 1),
                 'true_shape': torch.as_tensor(q_imgs[qi]['true_shape']).repeat(10, 1)}
        t0 = time.time()
        with torch.no_grad():
            q2d = inference_relpose([view1, view2], model, device, use_amp=True)
        torch.cuda.synchronize()
        t_infer += time.time() - t0
        pd = [pose(cfg['db'], db_names[j]) for j in sel]
        pq = [p.cpu().numpy() for p in q2d]
        gt = pose(cfg['q'], q_names[qi])
        for mode in modes:
            if mode:
                cand_t = np.array([(pd[k] @ pq[k])[:3, 3] for k in range(10)])
                rough = np.median(cand_t, axis=0)
                keep = [k for k in range(10) if np.linalg.norm(pd[k][:3, 3] - rough) < mode]
                if len(keep) < 2:
                    keep = list(range(10))
                Rt = visloc.motion_averaging([pd[k] for k in keep], [pq[k] for k in keep])
            else:
                Rt = visloc.motion_averaging(pd, pq)
            acc[mode]['t'].append(float(np.linalg.norm(Rt[:3, 3] - gt[:3, 3])))
            acc[mode]['r'].append(get_rot_err(Rt[:3, :3], gt[:3, :3]))
            acc[mode]['hit'].append(any(np.linalg.norm(pd[k][:3, 3] - gt[:3, 3]) < 2.0 for k in range(10)))
            acc[mode]['pred'].append(Rt)
            acc[mode]['gt'].append(gt)
    out = {}
    for mode, d in acc.items():
        terrs, rerrs = np.array(d['t']), np.array(d['r'])
        tag = 'top10' if not mode else f'bm{mode:g}'
        m = {'n': len(terrs),
             'terr_m': {'median': float(np.median(terrs)), 'mean': float(terrs.mean()), 'p90': float(np.percentile(terrs, 90))},
             'rerr_deg': {'median': float(np.median(rerrs)), 'p90': float(np.percentile(rerrs, 90))},
             'succ_t': {f'<{t}m': float((terrs < t).mean()) for t in (0.25, 0.5, 1.0)},
             'hit2m_top10': float(np.mean(d['hit'])),
             'relpose_s_per_query': round(t_infer / len(q_names), 3)}
        out[tag] = m
        np.savez(f'{OUT}/res_{key}_{tag}.npz', terr=terrs, rerr=rerrs, hit=np.array(d['hit']),
                 pred=np.array(d['pred']), gt=np.array(d['gt']), names=np.array(q_names))
    return out, {'img_load_s': round(load_s, 1), 'relpose_total_s': round(t_infer, 1)}


def megaloc_bench():
    """MegaLoc 描述子提取耗时（= 替换掉的检索段成本）+ 单帧延迟。"""
    import torch
    from pathlib import Path
    sys.path.insert(0, '/data/code/reLocation-all/MegaLoc/scripts')
    sys.path.insert(0, '/data/code/reLocation-all/MegaLoc')
    from reproduce_inference import DEFAULT_WEIGHTS, build_transform, load_model
    from PIL import Image
    device = 'cuda'
    t0 = time.time()
    model = load_model(Path(DEFAULT_WEIGHTS), device)
    tr = build_transform()
    load_s = time.time() - t0
    pngs = sorted(os.listdir('./data/zc_0923/20260923_055425_real_robot_record_external_lz4/seq-01'))
    paths = [f'./data/zc_0923/20260923_055425_real_robot_record_external_lz4/seq-01/{n}' for n in pngs if n.endswith('.color.png')][:200]
    imgs = torch.stack([tr(Image.open(p).convert('RGB')) for p in paths]).to(device)
    with torch.inference_mode():
        for _ in range(3):
            model(imgs[:8])  # warmup
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = torch.cat([model(imgs[i:i+64]).float().cpu() for i in range(0, len(imgs), 64)])
        batch_s = time.time() - t0
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(imgs[:1])
        torch.cuda.synchronize()
        single_ms = (time.time() - t0) * 1e3
        mem = torch.cuda.max_memory_allocated() / 2**20
    return {'model_load_s': round(load_s, 1), 'per_frame_batch64_ms': round(batch_s / len(imgs) * 1e3, 1),
            'single_frame_ms': round(single_ms, 1), 'gpu_mem_alloc_mib': round(mem, 0), 'n': len(imgs)}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    results = {}
    # 检索段（替换掉的 NetVLAD 段）
    smp = e9.Sampler(); smp.start()
    bench = megaloc_bench()
    ret_st = e9.stats(smp.stop(), 'megaloc_embed', time.time() - smp.t0)
    print('MEGALOC_BENCH', json.dumps({**bench, 'resource': ret_st}))
    results['megaloc_embed'] = {**bench, 'resource': ret_st}
    # 端到端
    smp = e9.Sampler(); smp.start()
    for key in SCENES:
        out, extra = eval_scene(key)
        results[key] = {'metrics': out, 'timing': extra}
        print(key, json.dumps(out, ensure_ascii=False), flush=True)
    ev_st = e9.stats(smp.stop(), 'megaloc+reloc3r_eval_3bags', time.time() - smp.t0)
    results['eval_resource'] = ev_st
    print('RESOURCE', json.dumps(ev_st))
    json.dump(results, open(f'{OUT}/perf.json', 'w'), ensure_ascii=False, indent=1)

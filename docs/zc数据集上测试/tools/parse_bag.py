#!/usr/bin/env python
"""Parse one ZC mcap bag: topic counts + RTK/heading/map samples -> json + npz."""
import sys, json, math
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

WGS84_A, WGS84_F = 6378137.0, 1.0 / 298.257223563

def lla_to_ecef(lat_deg, lon_deg, alt):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = WGS84_F * (2 - WGS84_F)
    N = WGS84_A / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    return np.array([
        (N + alt) * math.cos(lat) * math.cos(lon),
        (N + alt) * math.cos(lat) * math.sin(lon),
        (N * (1 - e2) + alt) * math.sin(lat),
    ])

def ecef_to_enu(ecef, origin_ecef, lat0, lon0):
    lat, lon = math.radians(lat0), math.radians(lon0)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    R = np.array([
        [-so, co, 0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])
    return (R @ (ecef - origin_ecef).T).T

def main(bag, out_prefix):
    with open(bag, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        counts = {}
        for cid, ch in summary.channels.items():
            st = summary.statistics.channel_statistics[cid]
            counts[ch.topic] = st.message_count
        meta = {'bag': bag, 'counts': counts,
                'start_ns': summary.statistics.message_record_start_time,
                'end_ns': summary.statistics.message_record_end_time}
        print(json.dumps({k: v for k, v in counts.items()}, ensure_ascii=False, indent=None))

        fixes, heads, maps = [], [], []
        cam_info = None
        first = {'fix': True, 'head': True, 'map': True}
        n_img = 0
        for schema, ch, msg, decoded in reader.iter_decoded_messages():
            t = msg.log_time
            tp = ch.topic
            if tp == '/ll/position/gps/gnss_fix':
                m = decoded
                fixes.append((t, m.latitude, m.longitude, m.altitude, m.status.status))
            elif tp == '/ll/position/gps/heading_deg_stamped':
                m = decoded
                if first['head']:
                    print('heading fields:', [f.name for f in msg.contents.fields] if hasattr(msg, 'contents') else dir(m))
                    first['head'] = False
                # try common field names
                deg = getattr(m, 'heading_deg', None)
                if deg is None:
                    deg = getattr(m, 'heading', None)
                if deg is None:
                    deg = getattr(m, 'data', None)
                heads.append((t, float(deg)))
            elif tp == '/camera_01/color/camera_info' and cam_info is None:
                m = decoded
                cam_info = {'k': list(m.k), 'w': m.width, 'h': m.height,
                            'frame': m.header.frame_id}
                print('camera_info:', cam_info)
            elif tp == '/camera_01/color/image_raw':
                n_img += 1
            elif tp == '/map' and len(maps) < 40:
                m = decoded
                p, q = m.pose.pose.position, m.pose.pose.orientation
                maps.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    lat0, lon0 = fixes[0][1], fixes[0][2]
    ecef = np.array([lla_to_ecef(f[1], f[2], f[3]) for f in fixes])
    enu = ecef_to_enu(ecef, ecef[0], lat0, lon0)
    st = np.array([f[4] for f in fixes])
    ts = np.array([f[0] for f in fixes]) * 1e-9
    out = {
        'meta': meta,
        'camera_info': cam_info,
        'n_image_msgs_decoded_skipped': n_img,
        'n_fix': len(fixes),
        'fix_status_hist': dict(zip(*np.unique(st, return_counts=True))),
        'enu_extent_m': {'e': [float(enu[:, 0].min()), float(enu[:, 0].max())],
                          'n': [float(enu[:, 1].min()), float(enu[:, 1].max())]},
        'path_len_m': float(np.linalg.norm(np.diff(enu, axis=0), axis=1).sum()),
        'fix_dur_s': float(ts[-1] - ts[0]),
        'n_heading': len(heads),
        'heading_first_last': heads[:1] + heads[-1:],
        'n_map': len(maps),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    with open(out_prefix + '.json', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    np.savez(out_prefix + '.npz', enu=enu, ts=ts, status=st,
             heads=np.array(heads) if heads else np.zeros((0, 2)),
             maps=np.array(maps) if maps else np.zeros((0, 8)))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

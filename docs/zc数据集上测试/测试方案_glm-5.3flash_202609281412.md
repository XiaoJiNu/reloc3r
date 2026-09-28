# Reloc3r 在 ZC 自采数据集上的重定位测试方案（评审确认版）

时间戳：2026-09-28 14:12 CST（v3，评审已确认；分支 `reloc3r-glm-5.3flashx`；测试随后在本分支执行）
环境：conda env `mv2d`（Python 3.10.20，torch 2.7.1+cu128，RTX 5090 Laptop 24G，curope CUDA kernel 已编译）

## 0. 路线（一句话）

三个 mcap 包整理成 7Scenes 风格「去畸变图片 + 4×4 相机 c2w 位姿」目录（建图包 20260921_064638 = db，20260921_062224 = 同天查询，20260920_081533 = 跨天查询），复用仓库链路：NetVLAD top-10 检索 → Reloc3r-512 逐对相对位姿 → 运动平均出绝对位姿 → 与 `/map` 换算的相机 GT 比平移/旋转误差。

## 1. 评审确认结论（本版生效）

1. **GT 口径 = `/map`**（在线融合定位、RTK 全程 RTK-fixed；度量口径为「Reloc3r 相对在线系统的一致性」）。
2. **db 抽帧 = 每 5 帧**（≈2 Hz，约 690 张，帧距 ≈0.6 m）。
3. **topk = 10**，不做 topk=20 对照（视检索命中率再定）。
4. **坐标系 = lidar 系**（用户指定）：`/map` 位姿直接视为 hesai 雷达位姿 `T_map_lidar`，不引入车体系；相机位姿 `T_map_cam = T_map_lidar @ inv(T_lidar_cam)`，`T_lidar_cam` 取标定目录 `hesai_022419/image_02/lidar_to_image_02.txt`（README 采信、已人工验证；弃用含 180° 滚转的 `calib_velo_to_cam.txt`）。

## 2. 已实测的数据事实（支撑上述决策）

| 项 | 值 |
|---|---|
| 图像 | `/camera_01/color/image_raw` rgb8 640×480 @10 Hz，frame=`camera_01_color_optical_frame` |
| 内参 | K=(366.10, 366.01, 321.33, 242.17) 全包恒定；畸变 plumb_bob D=[-0.0317, 0.0348, -9.1e-5, 1.4e-4] → 提取时 `cv2.undistort`（alpha=0） |
| 外参 | T_lidar_cam 含 -79.96° 俯仰（z-down→光学系换轴）、光心在雷达系 (0.01,-1.42,1.01) m |
| /map | 82 Hz，三包同一条 327 m 闭合环形轨迹、同向绕行、叠画分米级吻合（`tools/traj_all.png`）→ 查询与 db 全程重叠 |
| 朝向核验 | yaw−行进方向差 1.4°/2.4°/3.2°，roll/pitch ±1°（p95<3.5°）→ 无隐藏换轴，/map 即车体前向 |
| 同步 | 图像↔/map 最近邻残差 p50≈4 ms / p95≈11 ms / max≈16 ms → 直接最近邻配对 |
| bag /tf | 0 条消息 → /map 刚体系=雷达系为显式假设（用户确认按 lidar 系处理，风险闭合） |

## 3. 实现（本分支新增 2 个文件，不改仓库现有代码）

- `eval_zc.py`：`extract`（mcap→图+位姿，含去畸变、同步残差统计、三包轨迹叠画）/ `retrieve`（NetVLAD 描述子 + top-10 pair 文件）/ `eval`（逐对回归→运动平均→指标）/ `sanity`（相邻 db 帧位姿自检）四个子命令，中间产物落盘可续跑。
- `reloc3r/datasets/zc_retrieval.py`：`ZcRetrieval`/`ZcRelpose`，照抄 sevenscenes 两件套接口（`_crop_resize_if_necessary`、(512,384)、c2w `.pose.txt`），内参换成本数据 K。
- 数据目录 `data/zc/<bag名>/seq-01/frame-XXXXXX.color.png + .pose.txt`（~3900 张 ≈2.5 G，磁盘余 233 G）。

## 4. 指标（分查询包报告 + 合计）

- 平移误差 median/mean/p90（m）；旋转误差 median/mean/p90（°）
- 成功率 terr<0.25 / 0.5 / 1 m
- 诊断：检索命中率（top-10 含 GT 距 <2 m db 帧）、运动平均前逐对候选误差分布、运动平均平移 SVD 最小奇异值占比（直线退化监测）
- 同天（062224）与跨天（081533）**分开**报告

## 5. 执行与产物

1. extract → 帧数/同步残差表 + 轨迹叠画 PNG
2. retrieve → 描述子与 pair 文件
3. eval → 上述指标 + 8 个随机查询帧可视化（query 图 + top-1 db 图 + 预测/GT 轨迹叠画）
4. 结果文档 `docs/zc数据集上测试/zc数据集上测试reloc3r效果_glm-5.3flash_YYYYMMDDHHMM.md`（结尾带时间戳）

## 6. 不做的事

鱼眼/深度/LiDAR/RTK 独立真值链、训练微调、时序滤波融合、仓库现有文件改动——除非主结果出现系统性偏差触发外参链排查。

---
文档时间戳：2026-09-28 14:12:00 CST

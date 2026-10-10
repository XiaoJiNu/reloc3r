#!/bin/bash
# ZC 域微调 Reloc3r（方案: relopt/docs/算法优化/ZC数据微调Reloc3r与XFeat并回测方案.md）
# 数据: 三场景检索对 even 半（≈1.16万, 定稿链 zc_fix）+ MegaDepth-1500 前750对（防遗忘锚，与遗忘检查eval后半不相交）
# 前提: checkpoints/reloc3r-512.pth 已由 safetensors 转换; 冒烟: SMOKE=1 ./scripts/finetune_zc.sh
set -e; cd "$(dirname "$0")/.."
PY=${PY:-/data/software/conda/anaconda3/envs/mv2d/bin/python}
ZC9="./_db-q_pair_info/zc9_{}_q-retrieval_db-step=1_topk=10.npy"
if [ "$SMOKE" = 1 ]; then
  N=200; ND=200; EP=1; OUT=checkpoints/_smoke_zc
else
  N=4000; ND=4000; EP=30; OUT=checkpoints/zc_finetune
fi
$PY train.py \
  --train_dataset "$N @ ZcRelpose(scene='20260921_064638', query_scene='20260920_081533', query_parity='even', resolution=[(512,384)], transform=ColorJitter) \
                 + $N @ ZcRelpose(scene='20260921_064638', query_scene='20260921_062224', query_parity='even', resolution=[(512,384)], transform=ColorJitter) \
                 + $N @ ZcRelpose(scene='20260923_055425_real_robot_record_external_lz4', query_scene='20260923_060415_real_robot_record_external_lz4', pairs_info_file_mask='$ZC9', query_parity='even', resolution=[(512,384)], transform=ColorJitter) \
                 + $ND @ MegaDepth_valid(split='train', pair_subset=750, resolution=(512,384), transform=ColorJitter)" \
  --test_dataset "2000 @ ZcRelpose(scene='20260921_064638', query_scene='20260920_081533', query_parity='odd', resolution=(512,384), seed=777)" \
  --model "Reloc3rRelpose(img_size=512)" \
  --pretrained "checkpoints/reloc3r-512.pth" \
  --lr 1e-5 --warmup_epochs 2 --epochs $EP --batch_size 8 --freeze_encoder \
  --save_freq 5 --keep_freq 10 --eval_freq 1 \
  --output_dir "$OUT"

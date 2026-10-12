#!/bin/bash
# §7 遗忘门槛补救链：finetune_pipeline.sh 全链(PIPELINE_DONE)后串行接棒。
# 背景：混30%(实测口径=4000/16000=25% MD)版遗忘 ❌（terr 2.05° vs 官方 1.56°，>+10%；
# ep10/ep20 检查点更差 2.71/2.01，早停无解）。预案=MD 混比提 ~50%：ND=12000/(12000+12000)。
# ponytail: MD 锚仍是前 750 对(与遗忘 eval 后半不相交、诚实)，16x 过采样换比例；
#           若 50% 还不过线再考虑扩 MD 全 1500 对+换 eval 子集。
set -u; cd "$(dirname "$0")/.."
PY=/data/software/conda/anaconda3/envs/mv2d/bin/python
FT=checkpoints/zc_finetune_mix50/checkpoint-best.pth

echo "[M0] waiting for main pipeline PIPELINE_DONE"
while ! grep -q PIPELINE_DONE logs_pipeline_sup.log logs_pipeline.log 2>/dev/null; do sleep 300; done
sleep 60  # 让 P6 报告写完

echo "[M1] mix50 finetune (30ep, resume 不允许=全新目录)" >> logs_mix50.log
# train.py 自动 resume 本目录 checkpoint-last.pth；看门狗管挂死
ND=12000 OUT_DIR=checkpoints/zc_finetune_mix50 bash scripts/finetune_zc.sh >> logs_mix50.log 2>&1
while [ ! -f checkpoints/zc_finetune_mix50/checkpoint-final.pth ]; do
  if ! pgrep -f "[t]rain.py" >/dev/null; then
    echo "  resuming mix50 $(date +%H:%M)" >> logs_mix50.log
    ND=12000 OUT_DIR=checkpoints/zc_finetune_mix50 bash scripts/finetune_zc.sh >> logs_mix50.log 2>&1 &
  fi
  sleep 120
done

echo "[M2] forgetting check mix50"
$PY scripts/eval_forgetting.py "$FT" mix50 >> logs_mix50.log 2>&1

echo "[M3] odd backtest mix50"
for k in 062224 081533 0923; do
  [ -f data/zc_sim3/res_${k}_sim5_bm3_mix50_odd.npz ] || \
    SIM3_K=5 SIM3_BM3=1 WEIGHTS=$FT EVAL_PARITY=odd FT_TAG=mix50 \
    $PY eval_zc_sim3.py $k >> logs_mix50.log 2>&1 || echo "EVAL_FAIL $k mix50"
done

echo "[M4] regenerate report"
$PY /data/code/reLocation-all/relopt/scripts/gen_reloc3r_finetune_report.py >> logs_mix50.log 2>&1
echo MIX50_DONE

#!/bin/bash
# Reloc3r ZC 微调全链（方案 §9 步骤3-5）：主训(AMP,even奇偶协议) -> 遗忘检查 -> 三场景奇数子集回测
# -> 上限行训练(all) -> 上限行回测(official/ft, all)。幂等：各阶段产物存在即跳过，可安全重启。
# 用法: cd reloc3r && bash scripts/finetune_pipeline.sh   （日志 logs_pipeline.log，完成标记 PIPELINE_DONE）
set -u; cd "$(dirname "$0")/.."
PY=/data/software/conda/anaconda3/envs/mv2d/bin/python
FT=checkpoints/zc_finetune/checkpoint-best.pth
FTALL=checkpoints/zc_finetune_all/checkpoint-best.pth
SC=("/data/code/reLocation-all/relopt/results/finetune")
mkdir -p "${SC[0]}"
ev(){ SIM3_K=5 SIM3_BM3=1 WEIGHTS="$1" EVAL_PARITY="$2" $PY eval_zc_sim3.py "$3" >> logs_pipeline.log 2>&1 || echo "EVAL_FAIL $3 $1 $2"; }

echo "[P1] main finetune (parity=even, 30ep)"
pgrep -f "[t]rain.py" >/dev/null && echo "  train.py already running, waiting" || bash scripts/finetune_zc.sh >> logs_pipeline.log 2>&1
while pgrep -f "[t]rain.py" >/dev/null; do sleep 120; done
[ -f "$FT" ] || { echo "PIPELINE_FAIL no $FT"; exit 1; }
echo "[P1] done -> $FT"

echo "[P2] forgetting check (ft arm, N_EVAL=200 same as official)"
[ -f results/forgetting_ft.json ] || $PY scripts/eval_forgetting.py "$FT" >> logs_pipeline.log 2>&1

echo "[P3] backtest sim5+bm3 odd subset (official already at *_official_odd.npz)"
for k in 062224 081533 0923; do
  [ -f data/zc_sim3/res_${k}_sim5_bm3_ft_odd.npz ] || ev "$FT" odd "$k"
done

echo "[P4] upper-bound run (train=all, 30ep)"
if [ ! -f "$FTALL" ]; then
  PARITY=all EVALP=all OUT_DIR=checkpoints/zc_finetune_all bash scripts/finetune_zc.sh >> logs_pipeline.log 2>&1
fi
while pgrep -f "[t]rain.py" >/dev/null; do sleep 120; done
[ -f "$FTALL" ] || { echo "PIPELINE_FAIL no $FTALL"; exit 1; }

echo "[P5] upper-bound backtest (official all + ft all, same subset)"
for k in 062224 081533 0923; do
  # official+all 时脚本 tag 为空 -> 文件名 res_<k>_sim5_bm3.npz（归档名不同，不覆盖）
  [ -f data/zc_sim3/res_${k}_sim5_bm3.npz ] || ev "" all "$k"
  [ -f data/zc_sim3/res_${k}_sim5_bm3_ft_all.npz ] || ev "$FTALL" all "$k"
done
cp results/forgetting_*.json "${SC[0]}/" 2>/dev/null
echo "[P6] report"
$PY /data/code/reLocation-all/relopt/scripts/gen_reloc3r_finetune_report.py >> logs_pipeline.log 2>&1 && cp results/forgetting_*.json "${SC[0]}/" 2>/dev/null
echo PIPELINE_DONE

#!/bin/bash
# 训练挂死看门狗（只杀不起，起由 finetune_pipeline.sh 负责，避免双启动竞态）。
# 依据: reloc3r 训练为 epoch 级心跳(--output_dir 下 log.txt, ~17min/次)。
# 心跳 45min 无更新 => 判定挂死（本机实测 16:55 X11 显示切换触发主线程自旋/GPU 0%，
# 进程活着 4.5h 不产出）。杀后由 pipeline 的等待循环重新拉起（train.py 自动 resume last ckpt）。
cd /data/code/reLocation-all/reloc3r
STALE=2700
while true; do
  sleep 300
  for p in $(pgrep -f "[t]rain.py"); do
    od=$(tr '\0' '\n' </proc/$p/cmdline 2>/dev/null | grep -A1 -- "--output_dir" | tail -1)
    # 心跳=tfevents(SummaryWriter 每 print_freq iter 写一次, 比 epoch 级的 log.txt 平滑;
    # 曾用 log.txt 且单 epoch ~30min 时误杀正常训练 2026-10-12)
    lp=$(ls -t "$od"/events.out.tfevents.* 2>/dev/null | head -1)
    [ -n "$lp" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$lp" 2>/dev/null || date +%s) ))
    if [ "$age" -gt "$STALE" ]; then
      echo "$(date +%F_%H:%M) watchdog: $p stale=${age}s log=$lp -> SIGKILL" >> logs_watchdog.log
      kill -9 "$p" 2>/dev/null
    fi
  done
done

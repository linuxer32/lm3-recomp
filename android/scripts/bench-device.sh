#!/bin/sh
# Measure a running build's frame rate on a device.
#
# The game library prints `[recomp-vulkan] presented <n> frames (<draws>
# draws, <skipped> skipped)` every 128 presents; the frame rate is that step
# over the time between two lines. The window is 1/128 of what it prints, so
# a short run still reads accurately.
#
# Usage: android/scripts/bench-device.sh [seconds] [package]
#   ADB=/path/to/adb to use another adb.
#
# The activity is restarted first. Press nothing: the loading screen and the
# title screen both present frames, so name the phase in the output.
set -e

ADB=${ADB:-adb}
SECONDS_TO_RUN=${1:-120}
PACKAGE=${2:-dev.switchrecomp.android}
LOG=${TMPDIR:-/tmp}/recomp-bench.log

"$ADB" logcat -c
"$ADB" shell am force-stop "$PACKAGE"
"$ADB" shell am start -n "$PACKAGE/.RecompActivity" >/dev/null
echo "launched $PACKAGE; sampling ${SECONDS_TO_RUN}s"
sleep "$SECONDS_TO_RUN"

"$ADB" logcat -d -v time -s SwitchRecompGame >"$LOG"
awk '
  $0 ~ /presented [0-9]+ frames/ {
    n = ""
    for (i = 1; i <= NF; i++) if ($i == "presented") n = $(i + 1)
    if (n == "") next
    split($2, clock, ":")
    at = clock[1] * 3600 + clock[2] * 60 + clock[3]
    if (previous_at != "") {
      span = at - previous_at
      step = n - previous_n
      if (span > 0 && step > 0) {
        printf "%s  %6.2f FPS  (%s frames in %.2fs)\n", $2, step / span, step, span
      }
    }
    previous_at = at
    previous_n = n
  }
' "$LOG"

echo
echo "frames shown (guest) and draws last reported:"
grep -E "frame [0-9]+ shown" "$LOG" | tail -3
grep -E "presented [0-9]+ frames" "$LOG" | tail -1
echo "log: $LOG"

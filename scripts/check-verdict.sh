# shellcheck shell=bash
# Sourced by scripts/linux/check.sh and scripts/windows/check.sh.
#
# check_verdict <status file>: print the file and succeed only when the run
# reached `done` and every step recorded 0. A run that broke before its steps
# (a remote shell error, a lost connection) never writes `done` and must not
# read as a pass. Lines may end in CR when the host is Windows.
check_verdict() {
  local status="$1" line passed=1 finished=0
  if [ ! -f "$status" ]; then
    echo "no status file: the run did not report" >&2
    return 1
  fi
  cat "$status"
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in
      '') ;;
      done) finished=1 ;;
      *=0) ;;
      *) passed=0 ;;
    esac
  done < "$status"
  [ "$passed" = 1 ] && [ "$finished" = 1 ]
}

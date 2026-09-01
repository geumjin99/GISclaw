#!/usr/bin/env bash
# brief.sh — 새 세션을 시작할 때 가장 먼저 실행한다.
# 프로젝트의 현재 상태를 한 화면으로 출력한다. LLM 호출 없음, 순수 파일 읽기.
# 없는 항목은 조용히 건너뛴다.
#   사용: bash tools/brief.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BOLD=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; YEL=$'\e[33m'; GRN=$'\e[32m'; OFF=$'\e[0m'
hr() { printf '%s\n' "────────────────────────────────────────────────────────────"; }
has() { [ -e "$1" ]; }

TITLE=$(grep -m1 '^# ' PROJECT.md 2>/dev/null | sed 's/^# *//') || TITLE=""
echo "${BOLD}${TITLE:-$(basename "$PWD")}${OFF}   $(date '+%Y-%m-%d %H:%M')"
hr

# ── 1. 일정 카운트다운 ───────────────────────────────────────
if has tools/milestones.txt; then
  echo "${BOLD}■ 주요 일정${OFF}"
  today=$(date +%s)
  while IFS='|' read -r d label; do
    case "$d" in ''|\#*) continue ;; esac
    t=$(date -d "$d" +%s 2>/dev/null) || continue
    days=$(( (t - today) / 86400 ))
    if   [ "$days" -lt 0 ];  then mark="${DIM}지남${OFF}"
    elif [ "$days" -le 7 ];  then mark="${RED}D-${days}${OFF}"
    elif [ "$days" -le 30 ]; then mark="${YEL}D-${days}${OFF}"
    else mark="D-${days}"; fi
    printf '  %-12s %-8s %s\n' "$d" "$mark" "$label"
  done < tools/milestones.txt
  hr
fi

# ── 2. 현재 단계 + 블로커 ────────────────────────────────────
if has PROJECT.md; then
  stage=$(grep -m1 '현재 단계\|Current stage\|当前阶段' PROJECT.md)
  [ -n "$stage" ] && { echo "${BOLD}■ 현재 단계${OFF}"; echo "  ${stage}"; echo; }

  blockers=$(grep -E '^\| *\*{0,2}B[0-9]' PROJECT.md | sed 's/|/ /g; s/\*\*//g; s/  */ /g' | cut -c1-145)
  if [ -n "$blockers" ]; then
    echo "${BOLD}■ 블로커${OFF}"
    echo "$blockers" | sed 's/^/  • /'
    hr
  fi
fi

# ── 3. 다음 액션 ─────────────────────────────────────────────
if has PROJECT.md; then
  todo=$(grep '^- \[ \]' PROJECT.md | sed 's/^- \[ \]/  ☐/')
  if [ -n "$todo" ]; then
    echo "${BOLD}■ 다음 액션 (미완료)${OFF}"
    echo "$todo"
    echo "  ${DIM}(완료 $(grep -c '^- \[x\]' PROJECT.md)건)${OFF}"
    hr
  fi
fi

# ── 4. git ──────────────────────────────────────────────────
if git rev-parse --git-dir >/dev/null 2>&1; then
  echo "${BOLD}■ 최근 커밋${OFF}"
  git log --oneline -6 2>/dev/null | sed 's/^/  /'
  echo
  dirty=$(git status --porcelain 2>/dev/null | wc -l)
  if [ "$dirty" -gt 0 ]; then
    echo "  ${YEL}⚠ 커밋되지 않은 변경 ${dirty}건${OFF}"
    git status --porcelain | head -10 | sed 's/^/    /'
  else
    echo "  ${GRN}✓ working tree clean${OFF}"
  fi

  # PROJECT.md 가 최근 커밋들보다 오래 방치되었는지 경고
  if has PROJECT.md; then
    pm_last=$(git log -1 --format=%ct -- PROJECT.md 2>/dev/null || echo 0)
    head_last=$(git log -1 --format=%ct 2>/dev/null || echo 0)
    if [ "$pm_last" -gt 0 ] && [ $((head_last - pm_last)) -gt 604800 ]; then
      echo "  ${YEL}⚠ PROJECT.md 가 마지막 커밋보다 7일 이상 오래됐다 — 상태가 낡았을 수 있음${OFF}"
    fi
  fi
  # push 가드가 있는데 활성화되지 않았으면 경고
  if [ -d .githooks ] && [ "$(git config --get core.hooksPath 2>/dev/null)" != ".githooks" ]; then
    echo "  ${RED}⚠ .githooks/ 가 있는데 활성화되지 않았다 — 'git config core.hooksPath .githooks' 실행 필요${OFF}"
  fi
  hr
else
  echo "  ${RED}⚠ git 저장소가 아니다. 이력이 남지 않는다 — 'git init' 권장${OFF}"; hr
fi

# ── 5. 최근 손댄 문서 ────────────────────────────────────────
recent=$(find docs refs notes -name '*.md' -mtime -7 -printf '%TY-%Tm-%Td  %p\n' 2>/dev/null | sort -r | head -8)
if [ -n "$recent" ]; then
  echo "${BOLD}■ 최근 수정된 문서 (7일)${OFF}"
  echo "$recent" | sed 's/^/  /'
  hr
fi

# ── 6. 결정 기록 (ADR) ───────────────────────────────────────
adr=$(ls docs/04_decisions/ADR-*.md docs/decisions/ADR-*.md 2>/dev/null)
if [ -n "$adr" ]; then
  echo "${BOLD}■ 결정 기록 (ADR)${OFF}"
  for f in $adr; do
    st=$(grep -m1 -i '^- \*\*상태\*\*\|^- \*\*状态\*\*\|^- \*\*status\*\*\|^\*\*상태\*\*' "$f" | sed 's/.*[:：] *//; s/ *—.*//')
    printf '  %-46s %s\n' "$(basename "$f")" "${st:-?}"
  done
  hr
fi

# ── 6b. 버린 길의 기록 (lessons) ─────────────────────────────
lsn=$(ls docs/07_lessons/LSN-*.md 2>/dev/null | wc -l)
if [ "$lsn" -gt 0 ]; then
  echo "${BOLD}■ 버린 길의 기록 (docs/07_lessons)${OFF}   ${DIM}총 ${lsn}건${OFF}"
  ls -t docs/07_lessons/LSN-*.md 2>/dev/null | head -3 | while read -r f; do
    printf '  %s\n' "$(basename "$f" .md)"
  done
  hr
fi

# ── 7. CCPM (설치된 경우만) ──────────────────────────────────
if [ -x .claude/skills/ccpm/references/scripts/status.sh ]; then
  echo "${BOLD}■ CCPM${OFF}"
  bash .claude/skills/ccpm/references/scripts/status.sh 2>/dev/null \
    | sed -n '/PRDs:/,$p' | grep -E ':' | sed 's/^/  /'
  hr
fi

echo "${DIM}상세는 PROJECT.md · 이력은 git log${OFF}"

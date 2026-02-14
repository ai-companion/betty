"""Betty Agent — continuous session observer.

Heuristic observation engine with optional LLM-powered narrative and drift detection.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING

import litellm
import openai

from .agent_models import AgentObservation, FileChange, SessionReport
from .analyzer import GoalExtractor
from .cache import AgentCache
from .metrics import SessionMetrics, compute_session_metrics, _is_error_turn

if TYPE_CHECKING:
    from .config import AgentConfig, LLMConfig
    from .models import Session, Turn

logger = logging.getLogger(__name__)

# Suppress litellm verbose logging
litellm.suppress_debug_info = True

NARRATIVE_SYSTEM_PROMPT = (
    "You are a supervisor observing an AI coding assistant's session in real-time. "
    "Given the session goal, recent observations, and current metrics, write a 2-3 sentence "
    "situation report describing what the assistant is currently doing and how it's progressing. "
    "Be specific about files, tools, and actions. Use present tense. "
    "Return ONLY the narrative text, no JSON, no markdown."
)

DRIFT_SYSTEM_PROMPT = (
    "You are a supervisor monitoring an AI coding assistant for goal drift. "
    "Compare the session's original goal with recent activity and determine if the assistant "
    "is still on track.\n\n"
    "Return valid JSON with exactly these fields:\n"
    '- "on_track": boolean, true if still aligned with goal\n'
    '- "drift_description": string, brief description of any drift (empty if on track)\n'
    '- "progress_assessment": one of "on_track", "slow", "stalled", "spinning", "off_track"\n\n'
    "Return ONLY valid JSON, no markdown fences, no extra text."
)

GOAL_SYSTEM_PROMPT = (
    "You are a supervisor observing an AI coding assistant's session. "
    "Based on the conversation so far, determine the user's current goal for this session.\n\n"
    "Guidelines:\n"
    "- Focus on the USER's intent — what do they want the assistant to accomplish?\n"
    "- If the first message is a greeting (like 'hi', 'hello'), look at subsequent messages for the real goal.\n"
    "- If the user has shifted focus to a new task, the goal should reflect the CURRENT focus, not the original.\n"
    "- Be specific and concise: 1 sentence, under 80 characters if possible.\n"
    "- Describe the goal as an action (e.g., 'Fix the login page CSS' not 'The user wants to fix...').\n\n"
    "Return ONLY the goal text, no JSON, no quotes, no markdown, no explanation."
)


class Agent:
    """Continuous observer that tracks session progress via heuristics + optional LLM.

    Thread-safe: all report state is protected by ``_lock``.
    """

    def __init__(self, config: AgentConfig, llm_config: LLMConfig | None = None) -> None:
        self._config = config
        self._llm_config = llm_config
        self._reports: dict[str, SessionReport] = {}
        self._lock = threading.Lock()
        # Per-session counters for milestone tracking
        self._tool_counts: dict[str, int] = {}
        self._user_turn_counts: dict[str, int] = {}
        # Per-session last-turn timestamp for stall detection
        self._last_turn_ts: dict[str, datetime] = {}
        # Per-session turn count for LLM update gating
        self._turns_since_update: dict[str, int] = {}
        # Per-session user turn count for goal re-evaluation
        self._user_turn_count_at_goal: dict[str, int] = {}
        # Reuse the existing multi-source goal extractor from the analyzer
        self._goal_extractor = GoalExtractor()
        # Persistent cache for reports
        self._cache = AgentCache(max_observations=config.max_observations)
        # Track which sessions have been loaded from cache
        self._loaded_from_cache: set[str] = set()
        # LLM executor (1 worker to avoid overloading)
        self._executor: ThreadPoolExecutor | None = None
        if llm_config:
            self._executor = ThreadPoolExecutor(max_workers=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_turn(self, turn: Turn, session: Session) -> None:
        """Process a new turn — run heuristic checks and emit observations."""
        sid = session.session_id

        with self._lock:
            if sid not in self._reports:
                # Try to restore from cache
                report = self._load_from_cache(sid)
                if report is None:
                    report = SessionReport(session_id=sid)
                self._reports[sid] = report

        # Collect observations from each detector
        observations: list[AgentObservation] = []

        obs = self._check_goal(turn, session)
        if obs:
            observations.append(obs)

        obs = self._check_stall(turn, session)
        if obs:
            observations.append(obs)

        obs = self._check_error_spike(turn, session)
        if obs:
            observations.append(obs)

        obs = self._check_retry_loop(turn, session)
        if obs:
            observations.append(obs)

        obs = self._check_file_changes(turn, session)
        if obs:
            observations.append(obs)

        obs = self._check_milestone(turn, session)
        if obs:
            observations.append(obs)

        # Track file changes
        self._track_file_change(turn, session)

        # Update report
        metrics = compute_session_metrics(session)
        with self._lock:
            report = self._reports[sid]
            for o in observations:
                report.observations.append(o)
                # Trim to max observations
                if len(report.observations) > self._config.max_observations:
                    report.observations = report.observations[-self._config.max_observations:]
            report.metrics = metrics
            report.updated_at = datetime.now()

        # Heuristic progress assessment (overridden by LLM if available)
        self._assess_progress(sid, metrics, session)

        # Persist to cache periodically
        with self._lock:
            report = self._reports.get(sid)
            if report:
                self._save_to_cache(sid, report)

        # Track last turn timestamp
        self._last_turn_ts[sid] = turn.timestamp

        # Track turns since last LLM update
        self._turns_since_update[sid] = self._turns_since_update.get(sid, 0) + 1

        # Maybe trigger LLM update (batched, not per-turn)
        self._maybe_update_narrative(sid, session)

    def get_report(self, session_id: str) -> SessionReport | None:
        """Get the current report for a session."""
        with self._lock:
            report = self._reports.get(session_id)
            if report is None:
                return None
            return report

    def shutdown(self) -> None:
        """Clean up resources and flush cache."""
        # Save all reports to cache
        with self._lock:
            for sid, report in self._reports.items():
                self._save_to_cache(sid, report)
        self._cache.flush()
        if self._executor:
            self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def _load_from_cache(self, session_id: str) -> SessionReport | None:
        """Load a report from disk cache. Returns None if not cached."""
        if session_id in self._loaded_from_cache:
            return None

        self._loaded_from_cache.add(session_id)
        cached = self._cache.get(session_id)
        if not cached:
            return None

        try:
            report = SessionReport(session_id=session_id)
            # Don't restore goal from cache — GoalExtractor will re-derive
            # it from the full session history on the next turn
            report.narrative = cached.get("narrative", "")
            report.progress_assessment = cached.get("progress_assessment", "on_track")

            # Restore observations
            for obs_data in cached.get("observations", []):
                report.observations.append(AgentObservation(
                    turn_number=obs_data.get("turn_number", 0),
                    timestamp=datetime.fromisoformat(obs_data["timestamp"]),
                    observation_type=obs_data.get("observation_type", "info"),
                    content=obs_data.get("content", ""),
                    severity=obs_data.get("severity", "info"),
                    metadata=obs_data.get("metadata", {}),
                ))

            updated = cached.get("updated_at")
            if updated:
                report.updated_at = datetime.fromisoformat(updated)

            # Restore file changes
            for fc_data in cached.get("file_changes", []):
                report.file_changes.append(FileChange(
                    file_path=fc_data.get("file_path", ""),
                    operation=fc_data.get("operation", "read"),
                    turn_number=fc_data.get("turn_number", 0),
                    timestamp=datetime.fromisoformat(fc_data["timestamp"]),
                    lines_added=fc_data.get("lines_added", 0),
                    lines_removed=fc_data.get("lines_removed", 0),
                ))

            logger.debug(f"Restored {len(report.observations)} observations from cache for {session_id}")
            return report
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Failed to restore report from cache: {e}")
            return None

    def _save_to_cache(self, session_id: str, report: SessionReport) -> None:
        """Serialize and save a report to disk cache."""
        data = {
            "goal": report.goal,
            "current_objective": report.current_objective,
            "narrative": report.narrative,
            "progress_assessment": report.progress_assessment,
            "updated_at": report.updated_at.isoformat(),
            "observations": [
                {
                    "turn_number": obs.turn_number,
                    "timestamp": obs.timestamp.isoformat(),
                    "observation_type": obs.observation_type,
                    "content": obs.content,
                    "severity": obs.severity,
                    "metadata": obs.metadata,
                }
                for obs in report.observations
            ],
            "file_changes": [
                {
                    "file_path": fc.file_path,
                    "operation": fc.operation,
                    "turn_number": fc.turn_number,
                    "timestamp": fc.timestamp.isoformat(),
                    "lines_added": fc.lines_added,
                    "lines_removed": fc.lines_removed,
                }
                for fc in report.file_changes
            ],
        }
        self._cache.set(session_id, data)

    # ------------------------------------------------------------------
    # Progress assessment
    # ------------------------------------------------------------------

    def _assess_progress(self, session_id: str, metrics: SessionMetrics, session: Session) -> None:
        """Heuristic progress assessment from spin detection signals.

        Sets progress_assessment on the report. LLM drift detection (PR 6)
        can override this when available.
        """
        # Compute a composite score: positive signals push toward on_track,
        # negative signals push toward stalled/spinning.
        negative_signals = 0
        positive_signals = 0

        # Negative: high repetitive tool score
        if metrics.repetitive_tool_score > 0.7:
            negative_signals += 2
        elif metrics.repetitive_tool_score > 0.5:
            negative_signals += 1

        # Negative: error-retry loops
        if metrics.error_retry_count >= 3:
            negative_signals += 2
        elif metrics.error_retry_count >= 1:
            negative_signals += 1

        # Negative: high error rate
        if metrics.error_rate > 0.5:
            negative_signals += 2
        elif metrics.error_rate > 0.3:
            negative_signals += 1

        # Negative: output shrinking
        if metrics.output_shrinking:
            negative_signals += 1

        # Negative: trailing retries
        if metrics.retry_count >= 4:
            negative_signals += 2
        elif metrics.retry_count >= 2:
            negative_signals += 1

        # Negative: stall
        if metrics.seconds_since_last_turn > 300:  # 5 min
            negative_signals += 2
        elif metrics.seconds_since_last_turn > 120:  # 2 min
            negative_signals += 1

        # Positive: diverse tools
        if metrics.tool_diversity > 0.5:
            positive_signals += 1

        # Positive: files being touched
        if len(metrics.files_touched) >= 3:
            positive_signals += 1

        # Positive: good velocity
        if metrics.recent_velocity > 1.0:
            positive_signals += 1

        # Determine assessment
        net = negative_signals - positive_signals

        if net >= 5:
            assessment = "off_track"
        elif net >= 3:
            if metrics.repetitive_tool_score > 0.6 or metrics.error_retry_count >= 2:
                assessment = "spinning"
            else:
                assessment = "stalled"
        elif net >= 1:
            assessment = "slow"
        else:
            assessment = "on_track"

        with self._lock:
            report = self._reports.get(session_id)
            if report:
                # Only update if LLM hasn't set a drift-based assessment recently
                # (LLM updates reset turns_since_update, so if we just got an LLM update
                # the turns_since count is low)
                report.progress_assessment = assessment

    # ------------------------------------------------------------------
    # LLM-powered narrative and drift detection
    # ------------------------------------------------------------------

    def _maybe_update_narrative(self, session_id: str, session: Session) -> None:
        """Gate LLM updates to every N turns."""
        if not self._executor or not self._llm_config:
            return

        turns_since = self._turns_since_update.get(session_id, 0)
        if turns_since < self._config.update_interval:
            return

        # Reset counter
        self._turns_since_update[session_id] = 0

        # Build context snapshot (copy data under lock to avoid races)
        with self._lock:
            report = self._reports.get(session_id)
            if not report:
                return
            recent_obs = [o.content for o in report.observations[-10:]]
            metrics = report.metrics

        # Use GoalExtractor for the richest goal context
        goal = self._goal_extractor.extract(session)

        # Submit async LLM calls
        self._executor.submit(self._update_narrative_async, session_id, goal, recent_obs, metrics, session)
        self._executor.submit(self._check_goal_drift_async, session_id, goal, recent_obs, session)

    def _update_narrative_async(
        self,
        session_id: str,
        goal: str | None,
        recent_obs: list[str],
        metrics: SessionMetrics | None,
        session: Session,
    ) -> None:
        """LLM call for situation report narrative."""
        try:
            prompt = self._build_narrative_prompt(goal, recent_obs, metrics, session)
            narrative = self._call_llm(prompt, NARRATIVE_SYSTEM_PROMPT)

            with self._lock:
                report = self._reports.get(session_id)
                if report:
                    report.narrative = narrative
                    report.updated_at = datetime.now()
        except Exception as e:
            logger.debug(f"Narrative LLM call failed: {e}")

    def _check_goal_drift_async(
        self,
        session_id: str,
        goal: str | None,
        recent_obs: list[str],
        session: Session,
    ) -> None:
        """LLM call to assess goal drift."""
        if not goal:
            return

        try:
            prompt = self._build_drift_prompt(goal, recent_obs, session)
            response = self._call_llm(prompt, DRIFT_SYSTEM_PROMPT)

            data = self._parse_drift_response(response)
            if data is None:
                return

            with self._lock:
                report = self._reports.get(session_id)
                if not report:
                    return

                report.progress_assessment = data.get("progress_assessment", "on_track")

                if not data.get("on_track", True) and data.get("drift_description"):
                    report.observations.append(AgentObservation(
                        turn_number=len(session.turns),
                        timestamp=datetime.now(),
                        observation_type="goal_drift",
                        content=data["drift_description"],
                        severity="warning",
                        metadata={"assessment": report.progress_assessment},
                    ))
                    # Trim observations
                    if len(report.observations) > self._config.max_observations:
                        report.observations = report.observations[-self._config.max_observations:]
                    report.updated_at = datetime.now()

        except Exception as e:
            logger.debug(f"Drift LLM call failed: {e}")

    def _build_narrative_prompt(
        self,
        goal: str | None,
        recent_obs: list[str],
        metrics: SessionMetrics | None,
        session: Session,
    ) -> str:
        """Build prompt for narrative generation."""
        parts = []
        if goal:
            parts.append(f"Session goal: {goal}")

        # Recent turns summary (last 5 turns, condensed)
        recent_turns = session.turns[-5:]
        turn_lines = []
        for t in recent_turns:
            preview = t.content_preview[:100]
            turn_lines.append(f"  T{t.turn_number} [{t.role}]{f' ({t.tool_name})' if t.tool_name else ''}: {preview}")
        if turn_lines:
            parts.append("Recent turns:\n" + "\n".join(turn_lines))

        if recent_obs:
            parts.append("Recent observations:\n" + "\n".join(f"  - {o}" for o in recent_obs[-5:]))

        if metrics:
            parts.append(
                f"Metrics: {metrics.turn_velocity:.1f} turns/min, "
                f"error rate {metrics.error_rate:.0%}, "
                f"{len(metrics.files_touched)} files touched"
            )

        return "\n\n".join(parts)

    def _build_drift_prompt(
        self,
        goal: str,
        recent_obs: list[str],
        session: Session,
    ) -> str:
        """Build prompt for drift assessment."""
        parts = [f"Original goal: {goal}"]

        # Recent turns summary
        recent_turns = session.turns[-10:]
        turn_lines = []
        for t in recent_turns:
            preview = t.content_preview[:80]
            turn_lines.append(f"  T{t.turn_number} [{t.role}]{f' ({t.tool_name})' if t.tool_name else ''}: {preview}")
        if turn_lines:
            parts.append("Recent activity:\n" + "\n".join(turn_lines))

        if recent_obs:
            parts.append("Agent observations:\n" + "\n".join(f"  - {o}" for o in recent_obs[-5:]))

        return "\n\n".join(parts)

    def _parse_drift_response(self, response: str) -> dict | None:
        """Parse JSON response from drift check LLM."""
        import re

        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", response)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            match = re.search(r"\{[^}]+\}", cleaned)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Validate required fields
        valid_assessments = {"on_track", "slow", "stalled", "spinning", "off_track"}
        assessment = data.get("progress_assessment", "on_track")
        if assessment not in valid_assessments:
            data["progress_assessment"] = "on_track"

        return data

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Call LLM using the same routing as Summarizer.

        Routes: claude-code/* → subprocess, api_base → openai SDK, else → litellm
        """
        if not self._llm_config:
            raise RuntimeError("No LLM config available")

        model = self._llm_config.model
        api_base = self._llm_config.api_base
        api_key = self._llm_config.api_key

        if model.startswith("claude-code/"):
            return self._call_claude_code(prompt, system_prompt, model)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if api_base:
            client = openai.OpenAI(
                base_url=api_base.rstrip("/"),
                api_key=api_key or "no-key-required",
            )
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=300,
                temperature=0.3,
            )
        else:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.3,
            }
            if api_key:
                kwargs["api_key"] = api_key
            response = litellm.completion(**kwargs)

        return response.choices[0].message.content.strip()

    def _call_claude_code(self, prompt: str, system_prompt: str, model: str) -> str:
        """Call claude CLI in single-prompt mode."""
        claude_model = model.split("/", 1)[1] if "/" in model else model

        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence", "--model", claude_model,
             "--disable-slash-commands", "--tools", "", "--setting-sources", "",
             "--system-prompt", system_prompt],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"claude exited with code {result.returncode}")
        return result.stdout.strip()

    # ------------------------------------------------------------------
    # Heuristic detectors
    # ------------------------------------------------------------------

    def _check_goal(self, turn: Turn, session: Session) -> AgentObservation | None:
        """Set initial session goal and trigger LLM refinement.

        On the first pass: sets goal from the first user message as a heuristic
        fallback. Then kicks off an async LLM call to determine the real goal
        from conversation context.

        On subsequent user turns: triggers LLM re-evaluation so the goal can
        evolve as the session progresses.
        """
        sid = session.session_id
        with self._lock:
            report = self._reports.get(sid)
            if not report:
                return None

        # Count current user turns for tracking
        user_turn_count = sum(1 for t in session.turns if t.role == "user")

        # --- Heuristic initial goal (set once, no LLM needed) ---
        with self._lock:
            has_goal = report.goal is not None

        if not has_goal:
            # Find first user message for a quick heuristic goal
            first_user_content = None
            for t in session.turns:
                if t.role == "user":
                    first_user_content = t.content_full[:200] if t.content_full else t.content_preview[:200]
                    break

            if first_user_content and first_user_content != "[No user message found]":
                with self._lock:
                    if report.goal is None:  # Double-check under lock
                        report.goal = first_user_content
                        report.current_objective = first_user_content

                # Immediately trigger LLM goal determination
                if self._executor and self._llm_config:
                    self._executor.submit(self._determine_goal_async, sid, session)
                    self._user_turn_count_at_goal[sid] = user_turn_count

                return AgentObservation(
                    turn_number=turn.turn_number,
                    timestamp=turn.timestamp,
                    observation_type="goal_set",
                    content=f"Goal: {first_user_content}",
                    severity="info",
                    metadata={"goal": first_user_content, "source": "user_message"},
                )
            return None

        # --- LLM goal re-evaluation on new user turns ---
        if turn.role == "user" and self._executor and self._llm_config:
            last_count = self._user_turn_count_at_goal.get(sid, 0)
            if user_turn_count > last_count:
                self._user_turn_count_at_goal[sid] = user_turn_count
                self._executor.submit(self._determine_goal_async, sid, session)

        return None

    def _determine_goal_async(self, session_id: str, session: Session) -> None:
        """LLM call to determine or refine the session goal from conversation context."""
        try:
            prompt = self._build_goal_prompt(session)
            new_goal = self._call_llm(prompt, GOAL_SYSTEM_PROMPT)

            if not new_goal or len(new_goal) < 3:
                return

            # Clean up: strip quotes, periods at end
            new_goal = new_goal.strip().strip('"\'').rstrip('.')

            with self._lock:
                report = self._reports.get(session_id)
                if not report:
                    return

                old_goal = report.goal
                report.goal = new_goal
                report.current_objective = new_goal
                report.updated_at = datetime.now()

                # Log goal update as observation if it changed meaningfully
                if old_goal and old_goal != new_goal:
                    report.observations.append(AgentObservation(
                        turn_number=len(session.turns),
                        timestamp=datetime.now(),
                        observation_type="goal_set",
                        content=f"Goal updated: {new_goal}",
                        severity="info",
                        metadata={"goal": new_goal, "previous_goal": old_goal, "source": "llm"},
                    ))
                    if len(report.observations) > self._config.max_observations:
                        report.observations = report.observations[-self._config.max_observations:]

        except Exception as e:
            logger.debug(f"Goal determination LLM call failed: {e}")

    def _build_goal_prompt(self, session: Session) -> str:
        """Build prompt for LLM goal determination from conversation context."""
        parts = []

        # Include all user messages (they define the goal arc)
        user_messages = []
        for t in session.turns:
            if t.role == "user":
                content = t.content_full[:300] if t.content_full else t.content_preview[:300]
                user_messages.append(f"  User (T{t.turn_number}): {content}")
        if user_messages:
            parts.append("User messages in this session:\n" + "\n".join(user_messages))

        # Include recent assistant messages for context on what's happening
        recent_assistant = []
        for t in session.turns[-10:]:
            if t.role == "assistant":
                content = t.content_preview[:150]
                recent_assistant.append(f"  Assistant (T{t.turn_number}): {content}")
        if recent_assistant:
            parts.append("Recent assistant activity:\n" + "\n".join(recent_assistant))

        # Include active tasks if any
        active_tasks = [
            f"  [{t.status}] {t.subject}"
            for t in session.tasks.values()
            if not t.is_deleted
        ]
        if active_tasks:
            parts.append("Active tasks:\n" + "\n".join(active_tasks[:5]))

        return "\n\n".join(parts)

    def _check_error_spike(self, turn: Turn, session: Session) -> AgentObservation | None:
        """Detect error rate > 40% in last 10 tool turns."""
        if turn.role != "tool":
            return None

        tool_turns = [t for t in session.turns if t.role == "tool"]
        recent = tool_turns[-10:]
        if len(recent) < 3:
            return None

        error_count = sum(1 for t in recent if _is_error_turn(t))
        error_rate = error_count / len(recent)

        if error_rate > 0.4:
            return AgentObservation(
                turn_number=turn.turn_number,
                timestamp=turn.timestamp,
                observation_type="error_spike",
                content=f"Error rate {error_rate:.0%} in last {len(recent)} tool calls ({error_count} errors)",
                severity="warning",
                metadata={"error_rate": error_rate, "error_count": error_count, "window": len(recent)},
            )
        return None

    def _check_retry_loop(self, turn: Turn, session: Session) -> AgentObservation | None:
        """Detect same tool called 3+ times consecutively."""
        if turn.role != "tool" or not turn.tool_name:
            return None

        tool_turns = [t for t in session.turns if t.role == "tool"]
        if len(tool_turns) < 3:
            return None

        # Count consecutive same-tool calls at the end
        last_tool = tool_turns[-1].tool_name
        count = 0
        for t in reversed(tool_turns):
            if t.tool_name == last_tool:
                count += 1
            else:
                break

        if count >= 3:
            return AgentObservation(
                turn_number=turn.turn_number,
                timestamp=turn.timestamp,
                observation_type="retry_loop",
                content=f"{last_tool} called {count}x consecutively",
                severity="warning",
                metadata={"tool_name": last_tool, "count": count},
            )
        return None

    def _check_stall(self, turn: Turn, session: Session) -> AgentObservation | None:
        """Detect gap > 2 min since last turn (checked when a new turn arrives)."""
        sid = session.session_id
        prev_ts = self._last_turn_ts.get(sid)
        if prev_ts is None:
            return None

        gap = (turn.timestamp - prev_ts).total_seconds()
        if gap > 120:  # 2 minutes
            return AgentObservation(
                turn_number=turn.turn_number,
                timestamp=turn.timestamp,
                observation_type="stall",
                content=f"No activity for {gap / 60:.1f} min",
                severity="info",
                metadata={"gap_seconds": gap},
            )
        return None

    def _check_file_changes(self, turn: Turn, session: Session) -> AgentObservation | None:
        """Track Write/Edit tool paths."""
        if turn.role != "tool" or turn.tool_name not in ("Write", "Edit"):
            return None

        # Extract file path from content_preview
        preview = turn.content_preview.strip()
        if not preview:
            return None
        path = preview.split(" (")[0].strip()
        if not path:
            return None

        op = "wrote" if turn.tool_name == "Write" else "edited"
        return AgentObservation(
            turn_number=turn.turn_number,
            timestamp=turn.timestamp,
            observation_type="file_change",
            content=f"{op.capitalize()} {path}",
            severity="info",
            metadata={"file_path": path, "operation": turn.tool_name.lower()},
        )

    def _track_file_change(self, turn: Turn, session: Session) -> None:
        """Track detailed file operations (Read/Write/Edit) in the report."""
        if turn.role != "tool" or not turn.tool_name:
            return

        preview = turn.content_preview.strip()
        if not preview:
            return

        import re

        if turn.tool_name == "Read":
            path = preview.strip()
            if not path or not path.startswith("/"):
                return
            op = "read"
            lines_added = 0
            lines_removed = 0
        elif turn.tool_name == "Write":
            path = preview.split(" (")[0].strip()
            if not path:
                return
            op = "write"
            # Try to extract line count: "path (N lines)"
            m = re.search(r"\((\d+) lines?\)", preview)
            lines_added = int(m.group(1)) if m else 0
            lines_removed = 0
        elif turn.tool_name == "Edit":
            path = preview.split(" (")[0].strip()
            if not path:
                return
            op = "edit"
            # Try to extract: "path (-N, +M)"
            m = re.search(r"\(-(\d+),\s*\+(\d+)\)", preview)
            if m:
                lines_removed = int(m.group(1))
                lines_added = int(m.group(2))
            else:
                lines_added = 0
                lines_removed = 0
        else:
            return

        change = FileChange(
            file_path=path,
            operation=op,
            turn_number=turn.turn_number,
            timestamp=turn.timestamp,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )

        sid = session.session_id
        with self._lock:
            report = self._reports.get(sid)
            if report:
                report.file_changes.append(change)

    def _check_milestone(self, turn: Turn, session: Session) -> AgentObservation | None:
        """Emit milestone every 10th tool call or 5th user turn."""
        sid = session.session_id

        if turn.role == "tool":
            self._tool_counts[sid] = self._tool_counts.get(sid, 0) + 1
            count = self._tool_counts[sid]
            if count > 0 and count % 10 == 0:
                return AgentObservation(
                    turn_number=turn.turn_number,
                    timestamp=turn.timestamp,
                    observation_type="milestone",
                    content=f"Milestone: {count} tool calls",
                    severity="info",
                    metadata={"tool_count": count},
                )

        elif turn.role == "user":
            self._user_turn_counts[sid] = self._user_turn_counts.get(sid, 0) + 1
            count = self._user_turn_counts[sid]
            if count > 0 and count % 5 == 0:
                return AgentObservation(
                    turn_number=turn.turn_number,
                    timestamp=turn.timestamp,
                    observation_type="milestone",
                    content=f"Milestone: {count} user messages",
                    severity="info",
                    metadata={"user_turn_count": count},
                )

        return None

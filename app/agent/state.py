"""AURA Agent State Management.

Defines structured state representation and lifecycle statuses for the AURA Agent.
Tracks the user's objective, active plan, execution history, tool outputs,
and final response across reasoning steps.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(str, Enum):
    """Lifecycle statuses for the AURA Agent."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentState:
    """Structured container holding all state for an agent session/goal.

    Attributes:
        user_goal: The original high-level goal or instruction from the user.
        current_step: 1-based index indicating the current execution step.
        plan: Ordered list of planned action dictionaries.
        completed_actions: List of actions successfully or attempted executed.
        observations: Sequential textual or structured observations recorded.
        retrieved_context: Knowledge or context retrieved (e.g. from RAG or memory).
        tool_results: Mapping or storage of individual tool execution outputs.
        final_response: The compiled, user-facing final answer.
        status: The current status of the agent (idle, planning, executing, completed, failed).
        error: Optional error description string if an action or execution failed.
    """

    user_goal: str = ""
    current_step: int = 0
    plan: List[Dict[str, Any]] = field(default_factory=list)
    completed_actions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    retrieved_context: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    final_response: Optional[str] = None
    status: AgentStatus = AgentStatus.IDLE
    error: Optional[str] = None
    activities: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure status is an AgentStatus enum instance."""
        if isinstance(self.status, str):
            try:
                self.status = AgentStatus(self.status.lower())
            except ValueError:
                self.status = AgentStatus.IDLE

    def reset(self) -> None:
        """Reset all state attributes to their initial idle values."""
        self.user_goal = ""
        self.current_step = 0
        self.plan.clear()
        self.completed_actions.clear()
        self.observations.clear()
        self.retrieved_context.clear()
        self.tool_results.clear()
        self.final_response = None
        self.status = AgentStatus.IDLE
        self.error = None
        self.activities.clear()

    def add_observation(self, observation: str) -> None:
        """Append an observation to the sequential history."""
        self.observations.append(observation)

    def add_completed_action(self, action: Dict[str, Any]) -> None:
        """Record an executed action."""
        self.completed_actions.append(action)

    def add_activity(self, activity_type: str, label: str, status: str = "completed") -> None:
        """Record a user-safe high-level activity update.

        Guarantees that no internal prompts, secrets, reasoning tokens, or stack traces
        are stored.
        """
        # Ensure status is valid
        clean_status = "failed" if status == "failed" else "completed"
        self.activities.append({
            "type": str(activity_type).strip().lower(),
            "label": str(label).strip(),
            "status": clean_status,
        })

    def get_activities(self) -> List[Dict[str, str]]:
        """Return safe high-level activity records for the current state.

        If activities were already recorded during execution, returns them.
        Otherwise, safely derives them from the plan, completed actions, and execution status
        without exposing any hidden reasoning, internal prompts, or secrets.
        """
        if self.activities:
            return list(self.activities)

        derived: List[Dict[str, str]] = []

        # 1. Planning stage (if a plan was generated or goal processed)
        if self.plan or self.completed_actions or self.final_response or self.user_goal:
            plan_status = "failed" if (self.status == AgentStatus.FAILED and not self.completed_actions and not self.final_response) else "completed"
            derived.append({
                "type": "planning",
                "label": "Planning task",
                "status": plan_status,
            })

        # 2. Executed actions
        for act_dict in self.completed_actions:
            raw_act = (act_dict.get("action") or act_dict.get("name") or "action").lower().strip()
            is_success = act_dict.get("success", True)
            act_status = "completed" if is_success else "failed"

            if raw_act == "calculator":
                derived.append({"type": "tool", "label": "Using Calculator", "status": act_status})
            elif raw_act in ("retrieve", "retrieval", "rag"):
                derived.append({"type": "retrieval", "label": "Retrieving documents", "status": act_status})
            elif raw_act == "weather":
                derived.append({"type": "tool", "label": "Using Weather", "status": act_status})
            elif raw_act == "search":
                derived.append({"type": "tool", "label": "Searching", "status": act_status})
            elif raw_act in ("files", "read_file", "file"):
                derived.append({"type": "tool", "label": "Reading file", "status": act_status})
            elif raw_act == "respond":
                derived.append({"type": "response", "label": "Generating response", "status": act_status})
            else:
                derived.append({"type": "tool", "label": f"Using {raw_act.capitalize()}", "status": act_status})

        # 3. Response generation stage
        if self.final_response and not any(a.get("type") == "response" for a in derived):
            derived.append({
                "type": "response",
                "label": "Generating response",
                "status": "completed",
            })

        # If overall execution failed and no specific step was marked failed
        if self.status == AgentStatus.FAILED and derived and not any(a.get("status") == "failed" for a in derived):
            derived[-1]["status"] = "failed"

        return derived

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a standard Python dictionary."""
        data = asdict(self)
        data["status"] = self.status.value if isinstance(self.status, AgentStatus) else str(self.status)
        data["activities"] = self.get_activities()
        return data

    @property
    def goal(self) -> str:
        """Alias for user_goal."""
        return self.user_goal

    @goal.setter
    def goal(self, value: str) -> None:
        self.user_goal = value

    @property
    def steps_count(self) -> int:
        """Number of steps in the active plan."""
        return len(self.plan)

    def __getitem__(self, item: str) -> Any:
        """Provide dictionary-like subscript access for backwards compatibility."""
        if item == "goal":
            return self.user_goal
        if item == "steps_count":
            return len(self.plan)
        if hasattr(self, item):
            val = getattr(self, item)
            if item == "status" and isinstance(val, AgentStatus):
                return val.value
            return val
        raise KeyError(f"AgentState has no field '{item}'")

    def __contains__(self, item: str) -> bool:
        """Check if an attribute or key exists on the state."""
        if item in ("goal", "steps_count"):
            return True
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        """Safe dictionary-like get access."""
        try:
            return self[item]
        except KeyError:
            return default


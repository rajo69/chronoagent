"""AgentRegistry: capability map and agent lookup by task type.

The registry maps task-type strings to concrete :class:`~chronoagent.agents.base.BaseAgent`
subclasses and provides factory helpers so the pipeline graph can resolve an
agent class (or a pre-built instance) from a plain string.

Supported task types
--------------------
``"plan"``
    :class:`~chronoagent.agents.planner.PlannerAgent` — decomposes PR diffs
    into :class:`~chronoagent.agents.planner.ReviewSubtask` lists.

``"security_review"``
    :class:`~chronoagent.agents.security_reviewer.SecurityReviewerAgent` — identifies
    security vulnerabilities with CWE mappings.

``"style_review"``
    :class:`~chronoagent.agents.style_reviewer.StyleReviewerAgent` — checks code
    quality, naming, and complexity.

``"summarize"``
    :class:`~chronoagent.agents.summarizer.SummarizerAgent` — synthesizes all
    findings into a :class:`~chronoagent.agents.summarizer.ReviewReport`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chronoagent.agents.base import BaseAgent
from chronoagent.agents.planner import PlannerAgent
from chronoagent.agents.security_reviewer import SecurityReviewerAgent
from chronoagent.agents.style_reviewer import StyleReviewerAgent
from chronoagent.agents.summarizer import SummarizerAgent

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Internal capability map: task_type -> agent class
# ---------------------------------------------------------------------------

_CAPABILITY_MAP: dict[str, type[BaseAgent]] = {
    "plan": PlannerAgent,
    "security_review": SecurityReviewerAgent,
    "style_review": StyleReviewerAgent,
    "summarize": SummarizerAgent,
}


class UnknownTaskTypeError(KeyError):
    """Raised when a task type has no registered agent.

    Args:
        task_type: The unrecognised task-type string.
    """

    def __init__(self, task_type: str) -> None:
        super().__init__(
            f"No agent registered for task_type={task_type!r}. "
            f"Known types: {sorted(_CAPABILITY_MAP)}"
        )
        self.task_type = task_type


class AgentRegistry:
    """Registry that maps task-type strings to agent classes.

    The registry wraps :data:`_CAPABILITY_MAP` and provides convenience methods
    for querying it.  It is intentionally stateless — all state lives in the
    module-level map so a single import gives full access.

    Typical use::

        registry = AgentRegistry()
        agent_cls = registry.get_class("security_review")
        # build an instance with your backend + collection
        agent = agent_cls(agent_id="sec-1", backend=..., collection=...)
    """

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_class(self, task_type: str) -> type[BaseAgent]:
        """Return the agent class registered for *task_type*.

        Args:
            task_type: One of ``"plan"``, ``"security_review"``,
                ``"style_review"``, or ``"summarize"``.

        Returns:
            The concrete :class:`~chronoagent.agents.base.BaseAgent` subclass.

        Raises:
            UnknownTaskTypeError: If *task_type* is not in the capability map.
        """
        try:
            return _CAPABILITY_MAP[task_type]
        except KeyError:
            raise UnknownTaskTypeError(task_type)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def supported_task_types(self) -> list[str]:
        """Return all registered task-type strings, sorted alphabetically.

        Returns:
            Sorted list of registered task-type strings.
        """
        return sorted(_CAPABILITY_MAP)

    def capabilities(self) -> dict[str, type[BaseAgent]]:
        """Return a snapshot of the full capability map.

        Returns:
            Dict mapping task-type string → agent class.  The returned dict
            is a shallow copy; mutating it does not affect the registry.
        """
        return dict(_CAPABILITY_MAP)

    def has(self, task_type: str) -> bool:
        """Return ``True`` if *task_type* has a registered agent.

        Args:
            task_type: Task-type string to check.

        Returns:
            ``True`` if registered, ``False`` otherwise.
        """
        return task_type in _CAPABILITY_MAP

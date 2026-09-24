"""AURA Agent Core Orchestration.

Defines the central AuraAgent class that integrates AgentState,
ConversationMemory, LLMBrain, Planner, and lifecycle methods for autonomous goal processing.
In Phase 10 Step 25, AuraAgent plans and consults the LLM with conversational context,
while tool execution remains deferred.
"""

import logging
from typing import Any, Dict, List, Optional

from app.agent.state import AgentState, AgentStatus
from app.agent.planner import Planner
from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.memory.conversation import ConversationMemory
from app.brain.llm import LLMBrain

logger = logging.getLogger("AURA.Agent")

# Alias for backwards compatibility
PlanningInfo = AgentState


class AuraAgent:
    """Core autonomous agent for AURA.

    Coordinates goal intake, conversation memory context, plan synthesis,
    LLM reasoning, and state lifecycle.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        memory: Optional[ConversationMemory] = None,
        planner: Optional[Planner] = None,
        state: Optional[AgentState] = None,
        retriever: Optional[Any] = None,
        tools: Optional[Any] = None,
        max_steps: int = 10,
    ) -> None:
        """Initialize the AuraAgent with memory, LLM, planner, state, retriever, tools, and max_steps.

        Args:
            llm: LLMBrain or LLM provider instance. Defaults to LLMBrain(enable_tools=False).
            memory: ConversationMemory instance. Defaults to ConversationMemory().
            planner: Planner instance. Defaults to Planner().
            state: AgentState instance. Defaults to AgentState().
            retriever: LocalRetriever or compatible vector retriever instance.
            tools: ToolRegistry instance for executable tools.
            max_steps: Maximum step execution limit to prevent runaway execution loops.
        """
        # Backward compatibility if positional arguments were passed differently
        if isinstance(llm, Planner):
            planner = llm
            llm = None
        elif isinstance(llm, ConversationMemory):
            memory = llm
            llm = None

        self.memory: ConversationMemory = memory if memory is not None else ConversationMemory()
        self.llm: Any = llm if llm is not None else LLMBrain(enable_tools=False)
        self.planner: Planner = planner or Planner()
        self.state: AgentState = state or AgentState()
        self.max_steps: int = max_steps
        self.action_handlers: Dict[str, Any] = {}

        # Initialize tools
        if tools is not None:
            self.tools = tools
        elif hasattr(self.llm, "tool_registry") and self.llm.tool_registry is not None:
            self.tools = self.llm.tool_registry
        else:
            try:
                from app.tools.registry import create_default_registry
                self.tools = create_default_registry()
            except Exception as e:
                logger.warning(f"Could not initialize default ToolRegistry: {e}")
                self.tools = None

        # Initialize retriever
        if retriever is not None:
            self.retriever = retriever
        elif hasattr(self.llm, "retriever") and self.llm.retriever is not None:
            self.retriever = self.llm.retriever
        else:
            try:
                from app.rag.retriever import LocalRetriever
                self.retriever = LocalRetriever()
            except Exception as e:
                logger.warning(f"Could not initialize LocalRetriever: {e}")
                self.retriever = None

        self.initialize()

    def initialize(self) -> "AuraAgent":
        """Initialize and reset the agent state to IDLE ready for new goals."""
        self.state.reset()
        logger.info("AuraAgent initialized with IDLE status.")
        return self

    def register_action_handler(self, action_name: str, handler: Any) -> None:
        """Register a custom action handler callable.

        Args:
            action_name: Action name string (case-insensitive).
            handler: Callable taking (step_dict, agent_state) and returning result.
        """
        self.action_handlers[(action_name or "").strip().lower()] = handler

    def create_plan(
        self,
        goal: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Create a structured plan for the given goal or current user goal.

        Args:
            goal: Optional goal string. If provided, updates state.user_goal.
            context: Optional conversation context/history to assist planning.

        Returns:
            List of planned action step dictionaries.
        """
        if goal is not None:
            self.state.user_goal = goal

        self.state.status = AgentStatus.PLANNING
        plan = self.planner.create_plan(self.state.user_goal, context=context)
        self.state.plan = plan
        return plan

    def _extract_math_expression(
        self,
        text: str,
        previous_result: Optional[Any] = None,
    ) -> Optional[str]:
        """Extract or resolve a mathematical expression from text, supporting previous result substitution."""
        if not text:
            return None
        import re
        clean = text.strip()

        # If previous_result is provided, substitute {previous}, ans, result, etc.
        if previous_result is not None:
            clean = re.sub(r"\{(?:previous|result|ans)\}", str(previous_result), clean, flags=re.IGNORECASE)
            clean = re.sub(r"\b(?:previous|ans|result)\b", str(previous_result), clean, flags=re.IGNORECASE)

        # Normalize words and symbols
        clean = clean.replace("×", "*").replace("÷", "/")
        clean = re.sub(r"\btimes\b", "*", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bdivided\s+by\b", "/", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bplus\b", "+", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\bminus\b", "-", clean, flags=re.IGNORECASE)

        # Check for function call syntax like sqrt(144)
        func_match = re.search(r"\b(sqrt|sin|cos|tan|log|log10|abs|round|pow|ceil|floor)\s*\([0-9\.\,\s\+\-\*\/]+\)", clean, re.IGNORECASE)
        if func_match:
            return func_match.group(0).strip()

        # Check for standard arithmetic expressions (e.g. 25 * 40, 10 + 5, (10 + 2) * 3)
        arith_match = re.search(r"(\(?\s*\d+(?:\.\d+)?\s*\)?\s*(?:[\+\-\*\/\%\^]|\*\*)\s*\(?\s*\d+(?:\.\d+)?\s*\)?(?:\s*(?:[\+\-\*\/\%\^]|\*\*)\s*\(?\s*\d+(?:\.\d+)?\s*\)?)*)", clean)
        if arith_match:
            expr = arith_match.group(1).strip()
            expr = re.sub(r"\^", "**", expr)
            return expr

        return None

    def execute_plan(self) -> AgentState:
        """Execute the stored plan sequentially using the observation loop.

        Structure:
        Goal -> Plan -> Action -> Result -> Observation -> Next Action

        For every action:
        1. Read the current AgentState.
        2. Execute the action.
        3. Record the result.
        4. Create an observation from the result.
        5. Store the observation in AgentState.
        6. Move to the next action.
        7. Make previous results available to subsequent steps.

        Returns:
            The updated AgentState.

        Raises:
            NotImplementedError: If plan contains tools disabled in Phase 10 Step 28.
        """
        if not self.state.plan:
            logger.warning("No plan available to execute.")
            return self.state

        # Pre-check for disabled tools to preserve backward compatibility
        for step in self.state.plan:
            action = (step.get("action") or "").lower().strip()
            if action in ("weather", "search", "files"):
                raise NotImplementedError(
                    f"Tool execution is disabled in Phase 10 Step 28. "
                    f"Action '{action}' cannot be executed yet."
                )

        self.state.status = AgentStatus.EXECUTING
        last_action_result: Optional[Any] = None

        # Guard against plan exceeding max_steps
        if len(self.state.plan) > self.max_steps:
            err_msg = f"Plan length ({len(self.state.plan)}) exceeds maximum step limit of {self.max_steps} steps."
            logger.error(err_msg)
            self.state.error = err_msg
            self.state.add_observation(f"Execution halted: {err_msg}")
            self.state.status = AgentStatus.FAILED
            self.state.final_response = f"Execution halted: maximum step limit exceeded ({self.max_steps} steps)."
            return self.state

        step_counter = 0

        for step in self.state.plan:
            step_counter += 1
            if step_counter > self.max_steps:
                err_msg = f"Reached maximum step limit of {self.max_steps} steps."
                logger.error(err_msg)
                self.state.error = err_msg
                self.state.add_observation(f"Execution halted: {err_msg}")
                self.state.status = AgentStatus.FAILED
                self.state.final_response = f"Execution halted: maximum step limit of {self.max_steps} steps reached."
                return self.state

            step_num = step.get("step", step_counter)
            action = (step.get("action") or "").lower().strip()
            self.state.current_step = step_num
            logger.info(f"Executing step {step_num}: action='{action}'")

            # ------------------------------------------------------------------
            # Custom Action Handler
            # ------------------------------------------------------------------
            if action in self.action_handlers:
                try:
                    handler = self.action_handlers[action]
                    res = handler(step, self.state)
                    success = res.get("success", True) if isinstance(res, dict) else True
                    if success:
                        val = res.get("result", res) if isinstance(res, dict) else res
                        last_action_result = val
                        self.state.tool_results[action] = res if isinstance(res, dict) else {"success": True, "result": val}
                        obs = f"Action '{action}' completed successfully: {val}"
                        self.state.add_observation(obs)
                        self.state.add_completed_action({**step, "success": True, "result": val})
                        self.state.add_activity("tool", f"Using {action.capitalize()}", "completed")
                    else:
                        err = res.get("error", f"Action '{action}' failed.") if isinstance(res, dict) else f"Action '{action}' failed."
                        self.state.error = err
                        self.state.tool_results[action] = res if isinstance(res, dict) else {"success": False, "error": err}
                        obs = f"Action '{action}' failed: {err}"
                        self.state.add_observation(obs)
                        self.state.add_completed_action({**step, "success": False, "error": err})
                        self.state.add_activity("tool", f"Using {action.capitalize()}", "failed")
                        self.state.status = AgentStatus.FAILED
                        self.state.final_response = f"I encountered an error executing {action}: {err}"
                        return self.state
                except Exception as exc:
                    logger.error(f"Error executing custom action '{action}': {exc}")
                    err = str(exc)
                    self.state.error = err
                    self.state.tool_results[action] = {"success": False, "error": err}
                    obs = f"Action '{action}' failed with error: {err}"
                    self.state.add_observation(obs)
                    self.state.add_completed_action({**step, "success": False, "error": err})
                    self.state.add_activity("tool", f"Using {action.capitalize()}", "failed")
                    self.state.status = AgentStatus.FAILED
                    self.state.final_response = f"I encountered an error executing {action}: {err}"
                    return self.state
                continue

            # ------------------------------------------------------------------
            # Action: CALCULATOR
            # ------------------------------------------------------------------
            if action == "calculator":
                # Determine expression from step parameters or goal
                raw_expr = (
                    step.get("expression")
                    or step.get("parameters", {}).get("expression")
                    or step.get("params", {}).get("expression")
                )
                if not raw_expr:
                    expr = self._extract_math_expression(self.state.user_goal, previous_result=last_action_result)
                else:
                    expr = self._extract_math_expression(raw_expr, previous_result=last_action_result)

                if not expr:
                    err = "No valid mathematical expression found to calculate."
                    self.state.error = err
                    self.state.tool_results["calculator"] = {"success": False, "result": None, "error": err}
                    obs = f"Calculator action failed: {err}"
                    self.state.add_observation(obs)
                    self.state.add_completed_action({**step, "success": False, "error": err})
                    self.state.add_activity("tool", "Using Calculator", "failed")
                    self.state.status = AgentStatus.FAILED
                    self.state.final_response = f"I encountered an error during calculation: {err}"
                    return self.state

                # Execute calculation
                try:
                    if self.tools is not None and "calculator" in self.tools:
                        calc_result = self.tools.execute("calculator", expression=expr)
                    else:
                        from app.tools.calculator import calculate
                        calc_result = calculate(expr)
                except Exception as exc:
                    calc_result = {"success": False, "result": None, "error": str(exc)}

                if calc_result.get("success", False):
                    val = calc_result.get("result")
                    last_action_result = val
                    self.state.tool_results["calculator"] = calc_result
                    obs = f"Calculated {expr} = {val}."
                    self.state.add_observation(obs)
                    self.state.add_completed_action({**step, "success": True, "result": val, "expression": expr})
                    self.state.add_activity("tool", "Using Calculator", "completed")
                else:
                    err = calc_result.get("error") or "Calculation failed."
                    self.state.error = err
                    self.state.tool_results["calculator"] = calc_result
                    obs = f"Calculator action failed: {err}"
                    self.state.add_observation(obs)
                    self.state.add_completed_action({**step, "success": False, "error": err, "expression": expr})
                    self.state.add_activity("tool", "Using Calculator", "failed")
                    self.state.status = AgentStatus.FAILED
                    self.state.final_response = f"I encountered an error during calculation: {err}"
                    return self.state

            # ------------------------------------------------------------------
            # Action: RETRIEVE
            # ------------------------------------------------------------------
            elif action == "retrieve":
                query = step.get("query") or self.state.user_goal
                chunks: List[Dict[str, Any]] = []
                retrieval_failed = False

                if self.retriever is not None:
                    try:
                        chunks = self.retriever.retrieve(query)
                    except Exception as exc:
                        logger.warning(f"RAG retrieval error: {exc}")
                        retrieval_failed = True
                        chunks = []
                        self.state.error = str(exc)
                        self.state.add_observation(f"Retrieval failed with error: {exc}")
                else:
                    self.state.add_observation("No retriever configured; retrieval skipped.")

                self.state.retrieved_context = chunks
                sources = []
                for c in chunks:
                    meta = dict(c.get("metadata", {}))
                    sources.append({
                        "filename": meta.get("filename", "unknown"),
                        "page": meta.get("page"),
                        "score": c.get("score"),
                        "chunk_id": c.get("id") or meta.get("chunk_id"),
                    })

                self.state.tool_results["retrieve"] = {
                    "success": not retrieval_failed,
                    "chunks": chunks,
                    "sources": sources,
                    "count": len(chunks),
                }

                if chunks:
                    last_action_result = chunks
                    self.state.add_observation(
                        f"Retrieved {len(chunks)} relevant document chunk(s) from knowledge base."
                    )
                else:
                    if not retrieval_failed and not any("Retrieval failed" in obs for obs in self.state.observations):
                        self.state.add_observation("No relevant document chunks found in knowledge base.")

                self.state.add_completed_action({
                    **step,
                    "success": not retrieval_failed,
                    "chunks_count": len(chunks),
                })
                self.state.add_activity(
                    "retrieval",
                    "Retrieving documents",
                    "completed" if not retrieval_failed else "failed",
                )

            # ------------------------------------------------------------------
            # Action: RESPOND
            # ------------------------------------------------------------------
            elif action == "respond":
                conv_history = self.memory.get_context()
                # Disambiguate RAG document chunks from conversational memory history
                rag_context = None
                if self.state.retrieved_context:
                    is_rag_chunks = any(
                        isinstance(c, dict) and ("text" in c or "metadata" in c)
                        for c in self.state.retrieved_context
                    ) and not any(
                        isinstance(c, dict) and "role" in c
                        for c in self.state.retrieved_context
                    )
                    if is_rag_chunks:
                        rag_context = self.state.retrieved_context

                response = self._call_llm(
                    goal=self.state.user_goal,
                    context=conv_history,
                    rag_context=rag_context,
                    tool_results=self.state.tool_results,
                    observations=self.state.observations,
                )

                self.state.final_response = response
                self.state.add_observation("Synthesized final response based on execution results.")
                self.state.add_completed_action({**step, "success": True, "response": response})
                self.state.add_activity("response", "Generating response", "completed")
                self.state.status = AgentStatus.COMPLETED

            # ------------------------------------------------------------------
            # Unknown / Unsupported Action
            # ------------------------------------------------------------------
            else:
                err = f"Unknown or unsupported action '{action}'."
                logger.error(err)
                self.state.error = err
                self.state.tool_results[action] = {"success": False, "error": err}
                obs = f"Action '{action}' failed: {err}"
                self.state.add_observation(obs)
                self.state.add_completed_action({**step, "success": False, "error": err})
                self.state.add_activity("tool", f"Using {action.capitalize()}", "failed")
                self.state.status = AgentStatus.FAILED
                self.state.final_response = f"I encountered an unsupported action: {action}"
                return self.state

        if self.state.status == AgentStatus.EXECUTING:
            self.state.status = AgentStatus.COMPLETED

        return self.state

    def display_plan(self, plan: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format and display the generated plan to logs and output.

        Args:
            plan: Optional plan to display; defaults to self.state.plan.

        Returns:
            The formatted plan display string.
        """
        active_plan = plan if plan is not None else self.state.plan
        lines = [
            "=" * 50,
            "              AURA AGENT PLAN",
            "=" * 50,
            f"Goal: {self.state.user_goal}",
            "-" * 50,
            "Steps:",
        ]
        for step in active_plan:
            num = step.get("step", "?")
            act = step.get("action", "unknown")
            reason = step.get("reason", "")
            lines.append(f"  {num}. [{act.upper()}] {reason}")
        lines.append("=" * 50)

        output = "\n".join(lines)
        print(output)
        logger.info(f"\n{output}")
        return output

    def _call_llm(
        self,
        goal: str,
        context: Optional[List[Dict[str, Any]]] = None,
        rag_context: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[Dict[str, Any]] = None,
        observations: Optional[List[str]] = None,
    ) -> str:
        """Query the LLM component with user goal, conversation history, RAG context, tool results, and observations.

        Args:
            goal: User goal or prompt.
            context: List of recent conversation turns from memory.
            rag_context: Optional list of retrieved document chunks.
            tool_results: Optional dictionary of executed tool outputs.
            observations: Optional list of observations recorded during execution.

        Returns:
            The generated response string.
        """
        try:
            if hasattr(self.llm, "ask"):
                try:
                    return self.llm.ask(
                        user_message=goal,
                        context=context,
                        rag_context=rag_context,
                        tool_results=tool_results,
                        observations=observations,
                    )
                except TypeError:
                    if rag_context is not None:
                        try:
                            return self.llm.ask(user_message=goal, context=context, rag_context=rag_context)
                        except TypeError:
                            return self.llm.ask(user_message=goal, context=context, use_rag=True)
                    return self.llm.ask(user_message=goal, context=context)
            elif hasattr(self.llm, "generate"):
                from app.brain.prompts import build_llm_messages
                messages = build_llm_messages(
                    user_message=goal,
                    context=context,
                    system_prompt=AGENT_SYSTEM_PROMPT,
                    rag_chunks=rag_context,
                )
                return self.llm.generate(messages=messages)
            elif callable(self.llm):
                return str(self.llm(goal))
            else:
                return str(self.llm)
        except Exception as exc:
            logger.error(f"Error calling LLM in AuraAgent: {exc}")
            return f"I encountered an error while reasoning about your goal: {exc}"

    def process_goal(self, goal: str) -> AgentState:
        """Receive a user goal, retrieve memory, plan, execute sequentially, update state and memory.

        Steps:
        1. Store the goal in AgentState.
        2. Retrieve relevant conversation history from existing memory.
        3. Give the goal and relevant history to the planner.
        4. Store the generated plan in AgentState.
        5. Execute the plan via execute_plan() observation loop.
        6. Save the interaction to conversation memory.
        7. Return the AgentState.

        Args:
            goal: The user's request or objective.

        Returns:
            The updated AgentState.
        """
        logger.info(f"Processing goal: '{goal}'")

        # 1. Store the goal in AgentState
        self.state.user_goal = goal
        self.state.status = AgentStatus.PLANNING

        # 2. Retrieve relevant conversation history from existing memory
        relevant_history = self.memory.get_context()

        # 3. Give the goal and relevant history to the planner
        plan = self.create_plan(goal=goal, context=relevant_history)
        self.state.add_activity("planning", "Planning task", "completed")

        # 4. Store the generated plan in AgentState
        self.state.plan = plan
        self.display_plan(plan)

        # Keep retrieved_context aligned for conversational fallback goals
        if len(plan) == 1 and plan[0].get("action") == "respond":
            self.state.retrieved_context = relevant_history

        # Check if plan contains disabled tools:
        has_disabled_tool = any(
            (s.get("action") or "").lower().strip() in ("weather", "search", "files")
            for s in plan
        )
        if has_disabled_tool:
            # For backward compatibility with conversational fallback for unenabled tools:
            self.state.retrieved_context = relevant_history
            resp_step = [s for s in plan if s.get("action") == "respond"]
            self.state.current_step = resp_step[0].get("step", 1) if resp_step else 1
            response = self._call_llm(goal=goal, context=relevant_history)
            self.state.final_response = response
            self.state.add_activity("response", "Generating response", "completed")
            self.state.status = AgentStatus.COMPLETED
        else:
            # 5. Execute the plan via the observation loop
            self.execute_plan()

        # 6. Save interaction to conversation memory if successful response generated
        if self.state.final_response:
            self.memory.add_user_message(goal)
            self.memory.add_assistant_message(self.state.final_response)

        # 7. Return the AgentState
        return self.state


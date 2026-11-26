"""OpenAI and LangChain compatible chat agent with function-calling loop."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ecommerce_chat.agent.tools import ToolRegistry
from ecommerce_chat.config import DEFAULT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from ecommerce_chat.sessions import ChatMessage


class AgentResponse(BaseModel):
    """Structured response produced by an AI chat agent."""

    content: str
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)
    escalate_to_human: bool = False


@runtime_checkable
class BaseChatAgent(Protocol):
    """Protocol implemented by LLM agents compatible with the chat router."""

    async def generate_reply(
        self,
        history: list[ChatMessage],
        user_message: str,
        tools: ToolRegistry,
    ) -> AgentResponse:
        """Generate an assistant reply given session history and bound tools."""
        ...


def _history_to_openai_messages(
    system_prompt: str,
    history: list[ChatMessage],
    user_message: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if msg.role in ("customer", "user"):
            messages.append({"role": "user", "content": msg.content})
        elif msg.role in ("assistant", "agent"):
            messages.append({"role": "assistant", "content": msg.content})
    messages.append({"role": "user", "content": user_message})
    return messages


class OpenAIChatAgent:
    """Async OpenAI Chat Completions agent with iterative tool calling."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tool_iterations: int = 5,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tool_iterations = max_tool_iterations
        self._client = client or AsyncOpenAI(api_key=api_key)

    async def generate_reply(
        self,
        history: list[ChatMessage],
        user_message: str,
        tools: ToolRegistry,
    ) -> AgentResponse:
        messages = _history_to_openai_messages(
            self.system_prompt, history, user_message
        )
        tool_schemas = tools.openai_tool_schemas()
        tool_calls_made: list[dict[str, Any]] = []

        for _ in range(self.max_tool_iterations):
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                tools=tool_schemas,  # type: ignore[arg-type]
                tool_choice="auto",
            )
            choice = response.choices[0].message

            if not choice.tool_calls:
                return AgentResponse(
                    content=choice.content or "",
                    tool_calls_made=tool_calls_made,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.tool_calls
                    ],
                }
            )

            for tc in choice.tool_calls:
                raw_args = tc.function.arguments or "{}"
                args = json.loads(raw_args)
                output = await tools.execute(tc.function.name, args)
                tool_calls_made.append(
                    {"tool": tc.function.name, "arguments": args, "output": output}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

        # Final fallback call without tools if iteration cap is reached
        final_response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
        )
        return AgentResponse(
            content=final_response.choices[0].message.content or "",
            tool_calls_made=tool_calls_made,
        )


class LangChainChatAgent:
    """Optional LangChain-compatible chat agent using bound StructuredTools."""

    def __init__(
        self,
        chat_model: Any | None = None,
        api_key: str | None = None,
        model_name: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tool_iterations: int = 5,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_tool_iterations = max_tool_iterations

        if chat_model is not None:
            self._model = chat_model
        else:
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise ImportError(
                    "LangChainChatAgent requires 'langchain-openai'. "
                    "Install with: pip install 'ecommerce-chat-agent[langchain]'"
                ) from exc
            self._model = ChatOpenAI(model=model_name, api_key=api_key)

    async def generate_reply(
        self,
        history: list[ChatMessage],
        user_message: str,
        tools: ToolRegistry,
    ) -> AgentResponse:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        lc_messages: list[Any] = [SystemMessage(content=self.system_prompt)]
        for msg in history:
            if msg.role in ("customer", "user"):
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role in ("assistant", "agent"):
                lc_messages.append(AIMessage(content=msg.content))
        lc_messages.append(HumanMessage(content=user_message))

        lc_tools = tools.langchain_tools()
        bound_model = self._model.bind_tools(lc_tools)
        tool_calls_made: list[dict[str, Any]] = []

        for _ in range(self.max_tool_iterations):
            ai_msg = await bound_model.ainvoke(lc_messages)
            lc_messages.append(ai_msg)

            pending_calls = getattr(ai_msg, "tool_calls", None) or []
            if not pending_calls:
                return AgentResponse(
                    content=str(ai_msg.content),
                    tool_calls_made=tool_calls_made,
                )

            for call in pending_calls:
                name = call["name"]
                args = call.get("args", {})
                output = await tools.execute(name, args)
                tool_calls_made.append(
                    {"tool": name, "arguments": args, "output": output}
                )
                lc_messages.append(
                    ToolMessage(content=output, tool_call_id=call["id"])
                )

        final_msg = await self._model.ainvoke(lc_messages)
        return AgentResponse(
            content=str(final_msg.content),
            tool_calls_made=tool_calls_made,
        )

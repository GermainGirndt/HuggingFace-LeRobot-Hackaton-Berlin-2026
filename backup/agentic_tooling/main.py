import json
import os
from typing import Any, Callable, Literal, TypedDict, cast

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("Agent Tooling Module Started.")

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

openai_client = OpenAI()


ToolName = Literal[
    "put_pen_into_glass",
    "play_ticktack_toe",
    "refuse_invalid_input",
]


class SelectedToolCall(TypedDict):
    name: ToolName
    arguments: dict[str, Any]


class RobotArm:
    def put_pen_into_glass(self) -> str:
        print("Tool Executed: put_pen_into_glass")
        return "success"

    def play_ticktack_toe(self) -> str:
        print("Tool Executed: play_ticktack_toe")
        return "success"

    def refuse_invalid_input(self) -> str:
        print("Tool Executed: refuse_invalid_input")
        return "success"


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "put_pen_into_glass",
        "description": "Use when the user asks to put a pen into a glass.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "play_ticktack_toe",
        "description": "Use when the user asks to play tic-tac-toe or a simple game.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "refuse_invalid_input",
        "description": (
            "Use when the user request is unclear, nonsensical, unsupported, "
            "or asks for multiple actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


VALID_TOOL_NAMES: set[str] = {
    "put_pen_into_glass",
    "play_ticktack_toe",
    "refuse_invalid_input",
}


def call_best_tool(input_text: str) -> SelectedToolCall:
    open_ai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

    print("call_best_tool input:", input_text)

    response = openai_client.responses.create(
        model=open_ai_model,
        instructions="""
You are a robot arm receiving a spoken string input from the user.

Choose exactly one function to call.

Rules:
- If the user asks to put a pen into a glass, call put_pen_into_glass.
- If the user asks to play tic-tac-toe or a simple game, call play_ticktack_toe.
- If the input is unclear, nonsensical, unsupported, or asks for multiple actions, call refuse_invalid_input.
- Never answer with plain text.
- Never call more than one function.
        """.strip(),
        input=input_text,
        tools=TOOLS,
        tool_choice="required",
        parallel_tool_calls=False,
        temperature=0,
    )

    function_calls = [
        item for item in response.output
        if item.type == "function_call"
    ]

    if len(function_calls) != 1:
        raise ValueError(
            f"Expected exactly one function call, got {len(function_calls)}: "
            f"{response.output}"
        )

    function_call = function_calls[0]

    if function_call.name not in VALID_TOOL_NAMES:
        raise ValueError(f"Unexpected tool name: {function_call.name}")

    arguments = json.loads(function_call.arguments or "{}")

    return {
        "name": cast(ToolName, function_call.name),
        "arguments": arguments,
    }


def execute_tool(robot_arm: RobotArm, tool_call: SelectedToolCall) -> str:
    actions: dict[ToolName, Callable[[], str]] = {
        "put_pen_into_glass": robot_arm.put_pen_into_glass,
        "play_ticktack_toe": robot_arm.play_ticktack_toe,
        "refuse_invalid_input": robot_arm.refuse_invalid_input,
    }

    action = actions[tool_call["name"]]
    return action()


if __name__ == "__main__":
    print("Agentic Tooling Module Main Function Started.")

    robot_arm = RobotArm()

    tool_call = call_best_tool("I would like to jump rope.")
    print(f"Selected tool call: {tool_call}")

    result = execute_tool(robot_arm, tool_call)
    print(f"Execution result: {result}")

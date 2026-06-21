import json
import os
from typing import Any, Callable, Literal, TypedDict, cast

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file.
# Expected variables:
# - OPENAI_API_KEY
# - OPENAI_MODEL (optional)
load_dotenv()

print("Agent Tooling Module Started.")

# Fail fast if no OpenAI API key is configured.
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Shared OpenAI client used for all API requests.
openai_client = OpenAI()


# List of all supported robot actions.
# The LLM must always select exactly one of these.
ToolName = Literal[
    "put_pen_into_glass",
    "play_ticktack_toe",
    "refuse_invalid_input",
]


# Represents the tool call returned by the LLM.
#
# Example:
# {
#     "name": "put_pen_into_glass",
#     "arguments": {}
# }
class SelectedToolCall(TypedDict):
    name: ToolName
    arguments: dict[str, Any]


class RobotArm:
    """
    Robot control layer.

    In the hackathon version these methods only print messages.
    Later these methods can be connected to actual robot actions
    (LeRobot SO-101 commands, motion planning, gripper control, etc.).
    """

    def put_pen_into_glass(self) -> str:
        print("Tool Executed: put_pen_into_glass")
        return "success"

    def play_ticktack_toe(self) -> str:
        print("Tool Executed: play_ticktack_toe")
        return "success"

    def refuse_invalid_input(self) -> str:
        print("Tool Executed: refuse_invalid_input")
        return "success"


# Function definitions exposed to the OpenAI model.
#
# These definitions are used by the model to decide which robot action
# should be executed based on the user's spoken command.
#
# Since none of the actions currently require parameters,
# the JSON schema is empty.
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


# Additional safety layer.
#
# Even if the model somehow returns an unexpected function name,
# execution will be rejected before any robot action is performed.
VALID_TOOL_NAMES: set[str] = {
    "put_pen_into_glass",
    "play_ticktack_toe",
    "refuse_invalid_input",
}


def call_best_tool(input_text: str) -> SelectedToolCall:
    """
    Uses the OpenAI model to select exactly one robot action.

    Flow:
        Spoken user input
                ↓
           Whisper STT
                ↓
          input_text
                ↓
        OpenAI Function Calling
                ↓
         SelectedToolCall

    Returns:
        {
            "name": <selected tool>,
            "arguments": {}
        }
    """

    # Allow model selection through environment variables.
    # Defaults to GPT-4o.
    open_ai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

    print("call_best_tool input:", input_text)

    # Ask the model to choose exactly one tool.
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
        # Force function calling.
        tool_choice="required",
        # Ensure only one tool can be called.
        parallel_tool_calls=False,
        # Deterministic selection.
        temperature=0,
    )

    # Extract all function call outputs from the model response.
    function_calls = [
        item for item in response.output
        if item.type == "function_call"
    ]

    # Safety check:
    # The model must return exactly one tool call.
    if len(function_calls) != 1:
        raise ValueError(
            f"Expected exactly one function call, got {len(function_calls)}: "
            f"{response.output}"
        )

    function_call = function_calls[0]

    # Validate the returned function name.
    if function_call.name not in VALID_TOOL_NAMES:
        raise ValueError(f"Unexpected tool name: {function_call.name}")

    # Parse JSON arguments returned by the model.
    arguments = json.loads(function_call.arguments or "{}")

    return {
        "name": cast(ToolName, function_call.name),
        "arguments": arguments,
    }


def execute_tool(robot_arm: RobotArm, tool_call: SelectedToolCall) -> str:
    """
    Executes the tool selected by the LLM.

    This function acts as a dispatcher between
    the LLM-selected tool name and the actual robot method.
    """

    # Mapping between tool names and robot methods.
    actions: dict[ToolName, Callable[[], str]] = {
        "put_pen_into_glass": robot_arm.put_pen_into_glass,
        "play_ticktack_toe": robot_arm.play_ticktack_toe,
        "refuse_invalid_input": robot_arm.refuse_invalid_input,
    }

    # Resolve the selected action.
    action = actions[tool_call["name"]]

    # Execute the action and return its result.
    return action()


if __name__ == "__main__":
    """
    Demo execution.

    Example pipeline:

        User speaks
              ↓
        Whisper STT
              ↓
        "I would like to jump rope."
              ↓
        call_best_tool()
              ↓
        refuse_invalid_input
              ↓
        execute_tool()
              ↓
        Robot action
    """

    print("Agentic Tooling Module Main Function Started.")

    robot_arm = RobotArm()

    # Example user command.
    tool_call = call_best_tool("I would like to jump rope.")
    print(f"Selected tool call: {tool_call}")

    # Execute the selected robot action.
    result = execute_tool(robot_arm, tool_call)
    print(f"Execution result: {result}")

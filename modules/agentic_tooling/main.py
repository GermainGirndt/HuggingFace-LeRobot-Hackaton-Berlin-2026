import json
import os
from typing import Any, Literal, TypedDict

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

print("Agent Tooling Module Started.")

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")


openai_client = OpenAI()


class FormMessage(TypedDict):
    role: Literal["Technical", "Legal"]
    formContent: dict[str, Any]


class RequestBody(TypedDict):
    action: Literal["CHECK_QUALITY", "SUBMIT_FOR_NEXT_STEP"]
    history: list[FormMessage]


class RobotArm:
    def __init__(self) -> None:
        return

    def put_pen_into_glass(self) -> None:
        print(f"Tool Executed: put_pen_into_glass")

    def play_ticktack_toe(self) -> None:
        #  history: list[FormMessage]
        print(f"Tool Executed: play_ticktack_toe")

    def refuse_invalid_input(self) -> None:
        print(f"Tool Executed: refuse_invalid_input")


class ToolCall(TypedDict):
    tool_name: Literal[
        "put_pen_into_glass",
        "play_ticktack_toe",
        "refuse_invalid_input",
    ]


def read_api_output(response: Any) -> str:
    if "choices" in response and len(response["choices"]) > 0:
        choice = response["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
    print("API response:", response)
    print(
        f"Expected 'choices' with 'message' and 'content' in the API response, but got: {response}")
    raise ValueError("Invalid API response format.")


def call_best_tool(
    input_text: str
) -> ToolCall:
    open_ai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

    print("call_best_tool input:", input_text)

    request: dict[str, Any] = {
        "model": open_ai_model,
        "instructions": """
You are a robot arm, receiving an spoken string input from the user.

Choose exactly the best tool based on the input string. The available tools are: put_pen_into_glass, play_ticktack_toe and, in case of an input which does not make sense, refuse_invalid_input.

### Example 1
Input: "I want to play a game."
Output: {"tool_name": "play_ticktack_toe"}

### Example 2
Input: "I want to put a pen into a glass."
Output: {"tool_name": "put_pen_into_glass"}

### Example 3
Input: "I want to put a pen into a glass and then play a game."
Output: {"tool_name": "refuse_invalid_input"}

### Example 4
Input: "Why won't we go the the disco?"
Output: {"tool_name": "refuse_invalid_input"}
            """.strip(),
        "input": input_text,
        "tools": [
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
                "description": "Use when the user request is unclear, nonsensical, unsupported, or asks for multiple actions.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ],
    }

    response = openai_client.responses.create(**request)
    return json.loads(read_api_output(response))


if __name__ == "__main__":
    print(f"Agentic Tooling Module Main Function Started.")
    # Example usage
    robot_arm = RobotArm()
    best_tool = call_best_tool("I want to put a pen into a glass.")
    print(f"Best tool to use: {best_tool}")

    # TODO: abstract
    if best_tool["tool_name"] == "put_pen_into_glass":
        robot_arm.put_pen_into_glass()
    elif best_tool["tool_name"] == "play_ticktack_toe":
        robot_arm.play_ticktack_toe()
    elif best_tool["tool_name"] == "refuse_invalid_input":
        robot_arm.refuse_invalid_input()

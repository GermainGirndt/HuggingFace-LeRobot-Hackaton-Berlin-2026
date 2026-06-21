#pip install -U openai-whisper==20250625
import os
from dotenv import load_dotenv

load_dotenv()
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")


import whisper
import sys
import tempfile
import json
import traceback
import os

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import ssl
import certifi

import json

from typing import Any, Callable, Literal, TypedDict, cast

#from dotenv import load_dotenv
from openai import OpenAI

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
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
class SelectedToolCall(TypedDict):
    name: ToolName
    arguments: dict[str, Any]

class RobotArm:
    def __init__(self):
        print("Robot Arm Module Started.")

    def put_pen_into_glass(self) -> str:
        print("Tool Executed: put_pen_into_glass")
        return "success"

    def play_ticktack_toe(self) -> str:
        print("Tool Executed: play_ticktack_toe")
        return "success"

    def refuse_invalid_input(self) -> str:
        print("Tool Executed: refuse_invalid_input")
        return "success"


class Hackathon:
    def __init__(self):
        self.sample_rate = 44100
        self.frames = []
        self.recording = self.input_voice_raw()
        write("output.wav", self.sample_rate, self.recording)
        self.word_string = self.Voice2String()
        self.String2ToolCalling()




    def call_best_tool(self,input_text: str) -> SelectedToolCall:
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

    def execute_tool(self,robot_arm: RobotArm, tool_call: SelectedToolCall) -> str:
        actions: dict[ToolName, Callable[[], str]] = {
            "put_pen_into_glass": robot_arm.put_pen_into_glass,
            "play_ticktack_toe": robot_arm.play_ticktack_toe,
            "refuse_invalid_input": robot_arm.refuse_invalid_input,
        }

        action = actions[tool_call["name"]]
        return action()

    def String2ToolCalling(self):
        print("Agentic Tooling Module Main Function Started.")

        self.robot_arm = RobotArm()

        # tool_call = call_best_tool("Maybe some ticktacktoe would be good.")
        tool_call = self.call_best_tool(self.word_string)
        print(f"Selected tool call: {tool_call}")

        result = self.execute_tool(self.robot_arm, tool_call)
        print(f"Execution result: {result}")

    def Voice2String(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            print(f"[1] Temp-Datei erstellt: {tmp.name}")

            with open("output.wav", "rb") as f:
                audio_bytes = f.read()
                tmp.write(audio_bytes)
                tmp.flush()
                tmp.close()
                model = whisper.load_model("small")
                result = model.transcribe(tmp.name, language="en", verbose=False)
                segments = [
                    {"start": round(seg["start"], 2),  # type: ignore
                     "end": round(seg["end"], 2),  # type: ignore
                     "text": seg["text"].strip()}  # type: ignore
                    for seg in result["segments"]
                ]
                print(json.dumps({
                    "text": result["text"].strip(),  # type: ignore
                    "segments": segments
                }))
                output_string = result["text"].strip()

        except Exception as e:
            print(f"FEHLER: {e}")
            traceback.print_exc()

        finally:
            import os
            if os.path.exists(tmp.name):
                 os.unlink(tmp.name)
        return output_string

    def callback(self,indata, frame_count, time_info, status):
        self.frames.append(indata.copy())

    def input_voice_raw(self):
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16", callback=self.callback)
        with stream:
            input("Recording... press Enter to stop.\n")

        recording = np.concatenate(self.frames, axis=0)
        return recording

Hackathon = Hackathon()

#audio_bytes = sys.stdin.buffer.read()

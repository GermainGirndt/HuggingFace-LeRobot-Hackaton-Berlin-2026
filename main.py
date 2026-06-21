# pip install -U openai openai-whisper sounddevice scipy numpy python-dotenv certifi

from __future__ import annotations
import subprocess
import json
import os
import ssl
import tempfile
import traceback
from typing import Any, Callable, Literal, TypedDict, cast

import certifi
import numpy as np
import sounddevice as sd
import whisper
from dotenv import load_dotenv
from openai import OpenAI
from scipy.io.wavfile import write


# ---------- Setup ----------

load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

openai_client = OpenAI()


# ---------- Types ----------

ToolName = Literal[
    "pick_up_the_green_sharperner",
    "put_pen_into_glass",
    "play_tic_tac_toe",
    "refuse_invalid_input",
]


class SelectedToolCall(TypedDict):
    name: ToolName
    arguments: dict[str, Any]


# ---------- Tool Definitions for OpenAI ----------

ROBOT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "pick_up_the_green_sharperner",
        "description": "Use when the user asks to pick up the green sharperner.",
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
        "name": "play_tic_tac_toe",
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
            "Use when the user request is unclear, unsupported, nonsensical, "
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
    "pick_up_the_green_sharperner",
    "put_pen_into_glass",
    "play_tic_tac_toe",
    "refuse_invalid_input",
}


# ---------- Robot Control Layer ----------

class RobotArm:
    def __init__(self) -> None:
        print("Robot arm module started.")

    def pick_up_the_green_sharperner(self) -> str:
        print("Executing robot skill: pick_up_the_green_sharperner")

        print("Starting external process to control the robot arm...")
        process = subprocess.Popen(
            "my_custom_command",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        assert process.stdout is not None

        print("Robot arm output:")
        for line in process.stdout:
            print(line, end="")

        print("Waiting for the robot arm process to complete...")
        return_code = process.wait()

        if return_code == 0:
            print("Robot arm skill completed successfully.")
            return "success"

        print(f"Robot arm skill failed with exit code {return_code}.")
        return "failure"

    def put_pen_into_glass(self) -> str:
        print("Executing robot skill: put_pen_into_glass")
        # TODO: Add SO101 control code here.
        return "success"

    def play_tic_tac_toe(self) -> str:
        print("Executing robot skill: play_tic_tac_toe")
        # TODO: Add SO101 control code here.
        return "success"

    def refuse_invalid_input(self) -> str:
        print("Executing fallback: refuse_invalid_input")
        return "unsupported_or_unclear_request"


# ---------- Voice Recording ----------

class VoiceRecorder:
    def __init__(self, sample_rate: int = 44_100) -> None:
        self.sample_rate = sample_rate
        self.frames: list[np.ndarray] = []

    def _callback(self, indata: np.ndarray, frame_count: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"Recording status: {status}")
        self.frames.append(indata.copy())

    def record_until_enter(self) -> np.ndarray:
        print("Recording... press Enter to stop.")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=self._callback,
        ):
            input()

        if not self.frames:
            raise RuntimeError("No audio was recorded.")

        return np.concatenate(self.frames, axis=0)

    def save_wav(self, recording: np.ndarray, path: str) -> None:
        write(path, self.sample_rate, recording)


# ---------- Speech-to-Text ----------

class SpeechToText:
    def __init__(self, model_name: str = "small") -> None:
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)

    def transcribe(self, wav_path: str) -> str:
        result = self.model.transcribe(
            wav_path,
            language="en",
            verbose=False,
        )

        text = str(result["text"]).strip()
        print(f"Transcribed text: {text}")
        return text


# ---------- Agentic Tool Selection ----------

class ToolSelector:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    def select_tool(self, user_text: str) -> SelectedToolCall:
        print(f"Selecting tool for input: {user_text}")

        response = openai_client.responses.create(
            model=self.model,
            instructions="""
You are controlling a Hugging Face SO101 robot arm from spoken user input.

Choose exactly one function to call.

Rules:
- If the user asks to put a pen into a glass, call put_pen_into_glass.
- If the user asks to play tic-tac-toe or a simple game, call play_tic_tac_toe.
- If the input is unclear, unsupported, nonsensical, or asks for multiple actions, call refuse_invalid_input.
- Never answer with plain text.
- Never call more than one function.
            """.strip(),
            input=user_text,
            tools=ROBOT_TOOLS,
            tool_choice="required",
            parallel_tool_calls=False,
            temperature=0,
        )

        function_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        if len(function_calls) != 1:
            raise RuntimeError(
                f"Expected exactly one function call, got {len(function_calls)}."
            )

        function_call = function_calls[0]

        if function_call.name not in VALID_TOOL_NAMES:
            raise RuntimeError(f"Unexpected tool name: {function_call.name}")

        arguments = json.loads(function_call.arguments or "{}")

        return {
            "name": cast(ToolName, function_call.name),
            "arguments": arguments,
        }


# ---------- Execution Layer ----------

class RobotSkillExecutor:
    def __init__(self, robot_arm: RobotArm) -> None:
        self.robot_arm = robot_arm

        self.actions: dict[ToolName, Callable[[], str]] = {
            "pick_up_the_green_sharperner": self.robot_arm.pick_up_the_green_sharperner,
            "put_pen_into_glass": self.robot_arm.put_pen_into_glass,
            "play_tic_tac_toe": self.robot_arm.play_tic_tac_toe,
            "refuse_invalid_input": self.robot_arm.refuse_invalid_input,
        }

    def execute(self, tool_call: SelectedToolCall) -> str:
        tool_name = tool_call["name"]
        print(f"Selected tool: {tool_name}")

        action = self.actions[tool_name]
        return action()


# ---------- Main Pipeline ----------

class VoiceControlledRobotPipeline:
    def __init__(self) -> None:
        self.recorder = VoiceRecorder()
        self.transcriber = SpeechToText(model_name="small")
        self.tool_selector = ToolSelector()
        self.robot_arm = RobotArm()
        self.executor = RobotSkillExecutor(self.robot_arm)

    def run(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            wav_path = temp_audio.name

        try:
            recording = self.recorder.record_until_enter()
            self.recorder.save_wav(recording, wav_path)

            user_text = self.transcriber.transcribe(wav_path)
            tool_call = self.tool_selector.select_tool(user_text)

            result = self.executor.execute(tool_call)
            print(f"Execution result: {result}")

        except Exception as error:
            print(f"Error: {error}")
            traceback.print_exc()

        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)


if __name__ == "__main__":
    pipeline = VoiceControlledRobotPipeline()
    pipeline.run()

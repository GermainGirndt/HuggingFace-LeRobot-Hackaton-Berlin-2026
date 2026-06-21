# Hackaton Berlin 2026 – Hugging Face's LeRobot

# Jarvis, a Voice-Controlled Agentic Robotics with LeRobot SO-101

<img width="1920" height="1076" alt="title" src="https://github.com/user-attachments/assets/c5a96a09-80cf-487c-a26f-ee6e03dac92a" />

The Jarvis project demonstrates an agentic robotics pipeline that allows users to control a robot arm using natural voice commands.

The system captures spoken input through a microphone, converts speech into text using OpenAI Whisper, and then uses an LLM-based tool selection mechanism to determine the most appropriate robot action. The selected action is executed through a dedicated robot control interface, enabling intuitive voice-driven interaction with the Hugging Face LeRobot SO-101 platform.

### Group 6 Members

Salem Folz
Germain Girndt
Rakesh Suthar
Johanna Girndt
Mohsin Ali Mirza
Saad Rasheed

## Architecture

```text
Voice Input
     ↓
OpenAI Whisper (Speech-to-Text)
     ↓
Text Command
     ↓
GPT Function Calling
     ↓
Tool Selection
     ↓
Robot Action
     ↓
LeRobot SO-101
```

The project is designed around the concept of **skills**. Each robot capability is implemented as a separate tool that can be selected and executed by the language model. This architecture allows new robot skills to be added without changing the overall interaction flow.

## Requirements

### Python

The recommended Python version is:

```text
Python 3.10.14
```

Using other Python versions may work, but Python 3.10.14 is the version used during development and testing.

### FFmpeg

OpenAI Whisper requires FFmpeg to process audio files.

Verify that FFmpeg is installed:

```bash
ffmpeg -version
```

Installation examples:

- macOS:

```bash
brew install ffmpeg
```

- Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

- Windows:

Download and install FFmpeg from the official website and ensure it is available in your system PATH.

### Python Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root directory:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
```

Only `OPENAI_API_KEY` is required. The model defaults to `gpt-4o` if `OPENAI_MODEL` is not specified.

## Quick Start

1. Install the ffmpeg library for your OS.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the application:

```bash
# Tested with python 3.10.14
python main.py
```

The application will listen for a voice command, transcribe it using Whisper, select the most appropriate robot skill through OpenAI function calling, and execute the corresponding robot action.

## Disclaimer

The current implementation focuses on demonstrating the agentic control pipeline. The example robot skills included in this repository are intentionally simple and are intended to serve as a foundation for future robotic capabilities and experimentation.

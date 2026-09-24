# AURA — AI Unified Response Assistant

A modular, production-oriented AI voice assistant built in Python. Designed with a clear separation of concerns across Speech Processing, Query Understanding, LLM Orchestration, Dual-tier Memory, Tool Execution, and Retrieval-Augmented Generation (RAG).

> **Current Status**: **Phase 7 (Voice System)** completed. Decoupled Speech-to-Text (`app/speech/stt.py`) with ambient noise adjustment, configurable timeouts, and error handling; offline Text-to-Speech (`app/speech/tts.py`) via system synthesis engines; interactive voice interaction loop (`app/speech/voice_loop.py`) connecting microphone speech -> STT -> Intent Classification -> Memory -> RAG / Tools -> LLM Brain -> Memory -> TTS -> Speaker; and 165 unit and integration tests verified.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features & Roadmap](#features--roadmap)
3. [Architecture Diagram](#architecture-diagram)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Environment Variables](#environment-variables)
7. [Running the Project](#running-the-project)
8. [Testing](#testing)
9. [How the AI Pipeline Works](#how-the-ai-pipeline-works)
10. [Technologies Used](#technologies-used)
11. [Future Improvements](#future-improvements)

---

## 1. Introduction

**AURA** (AI Unified Response Assistant) is designed to bridge natural voice interaction with powerful generative AI models and local tool execution. Built from the ground up for clean software engineering and pedagogical clarity, AURA avoids black-box frameworks and excessive abstractions, favoring modular components with typed interfaces, unit tests, and sandboxed security.

---

## 2. Features & Roadmap

- [x] **Phase 1 — Foundation**: Clean directory structure, modular packages, logging, environment management, and test runners.
- [x] **Phase 2 — Intent Classification**: Offline lightweight intent recognition with preprocessing, vocabulary mapping, and fallback routing.
- [x] **Phase 3 — Brain (LLM Orchestration)**: Provider-agnostic LLM interface (Mock, OpenAI, Gemini) with prompt templating and conversation memory.
- [x] **Phase 4 — Memory Subsystem**: Sliding-window short-term conversational buffer alongside long-term semantic vector memory.
- [x] **Phase 5 — Tools Integration**: Sandboxed safe filesystem operations, arithmetic calculator, weather API, and web search.
- [x] **Phase 6 — Retrieval-Augmented Generation (RAG)**: Local document loader, text chunker, embedding generation, and cosine similarity retrieval.
- [x] **Phase 7 — Voice (STT / TTS)**: Decoupled speech-to-text listener and text-to-speech audio synthesis with ambient noise calibration and graceful fallback.
- [ ] **Phase 8 — End-to-End Autonomous Pipeline**: Full multimodal pipeline hardening and continuous listening refinements.

---

## 3. Architecture Diagram

```
                 User Voice
                     │
                     ▼
           [ Speech-to-Text (STT) ]
                     │
                     ▼
       [ Intent / Query Understanding ]
                     │
                     ▼
            ┌─────────────────┐
            │   AURA Brain    │
            └────────┬────────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
 [ LLM Client ]   [ Memory ]       [ Tools ]
 (Mock / Cloud)   ├─ Short-term    ├─ Calculator
                  └─ Vector Store  ├─ Weather
                                   ├─ Web Search
                                   └─ Safe Files
    │                │                │
    └────────────────┼────────────────┘
                     │
                     ▼
        [ Retrieval-Augmented Context ]
                     │
                     ▼
              Final Response
                     │
                     ▼
           [ Text-to-Speech (TTS) ]
                     │
                     ▼
                 Voice Output
```

---

## 4. Project Structure

```
aura-ai/
│
├── app/
│   ├── main.py                     # Entry point & coordinator
│   │
│   ├── speech/                     # Speech I/O
│   │   ├── __init__.py
│   │   ├── stt.py                  # Speech-to-Text interface
│   │   └── tts.py                  # Text-to-Speech interface
│   │
│   ├── brain/                      # Decision engine & LLM
│   │   ├── __init__.py
│   │   ├── llm.py                  # Abstract LLM client
│   │   ├── intent.py               # Intent classifier
│   │   └── prompts.py              # System prompts & templates
│   │
│   ├── memory/                     # Context & recall
│   │   ├── __init__.py
│   │   ├── conversation.py         # Short-term buffer
│   │   └── vector_memory.py        # Long-term semantic store
│   │
│   ├── tools/                      # Executable tools
│   │   ├── __init__.py
│   │   ├── calculator.py           # Safe arithmetic evaluator
│   │   ├── weather.py              # Weather lookup
│   │   ├── search.py               # Web search integration
│   │   └── files.py                # Sandboxed file manager
│   │
│   └── rag/                        # Document grounding
│       ├── __init__.py
│       ├── loader.py               # Document reader & chunker
│       ├── embeddings.py           # Dense vector generator
│       └── retriever.py            # Cosine similarity retriever
│
├── models/                         # Model weights and artifacts (.gitkeep)
│
├── data/                           # Datasets & sandboxed storage (.gitkeep)
│
├── tests/                          # Automated test suites
│   ├── __init__.py
│   ├── test_foundation.py          # Module import verification
│   ├── test_intent.py              # Intent classification tests
│   ├── test_memory.py              # Conversation memory tests
│   └── test_tools.py               # Tool safety & execution tests
│
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion rules
├── requirements.txt                # Project dependencies
└── README.md                       # Documentation
```

---

## 5. Installation

### Prerequisites
- **Python 3.11+** installed on your system.
- Git (optional, for version control).

### Setup Instructions

1. **Clone or navigate into the repository directory:**
   ```bash
   cd aura-ai-voice
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Environment Variables

AURA uses a `.env` file to manage configuration and secrets securely.

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Adjust the variables as needed:

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `AURA` | Name of the application |
| `APP_ENV` | `development` | Environment (`development`, `production`) |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LLM_PROVIDER` | `mock` | Active LLM backend (`mock`, `openai`, `gemini`) |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key |
| `GEMINI_API_KEY` | *(empty)* | Google Gemini API key |
| `SAFE_DATA_DIR` | `./data` | Sandboxed root folder for filesystem tool |
| `WEATHER_API_KEY` | *(empty)* | OpenWeather / Weather API key |
| `SERPAPI_API_KEY` | *(empty)* | SerpAPI key for search operations |
| `STT_PROVIDER` | `local` | STT provider (`local` / `google`, `mock`) |
| `STT_TIMEOUT` | `5` | Microphone timeout waiting for speech (seconds) |
| `STT_PHRASE_TIME_LIMIT` | `10` | Maximum recording length per speech turn (seconds) |
| `STT_ENERGY_THRESHOLD` | `300` | Energy threshold baseline for noise calibration |
| `TTS_PROVIDER` | `pyttsx3` | TTS provider (`pyttsx3`, `mock`) |
| `TTS_VOICE_RATE` | `180` | Spoken rate in words per minute |
| `TTS_VOLUME` | `1.0` | Output volume (0.0 to 1.0) |
| `TTS_VOICE_ID` | *(empty)* | Specific voice ID or name substring |
| `AURA_MODE` | `voice` | Mode (`voice` for microphone + TTS, `text` for console) |

---

## 7. Running the Project

### Interactive Voice Mode (Default)
Run the application directly to speak with AURA via your microphone and hear responses spoken through your speakers:

```powershell
python -m app.main
```

Example interaction:
```text
========================================================
           AURA Phase 7: Interactive Voice Mode         
========================================================
  * Speak into your microphone after the prompt appears.
  * Say 'goodbye', 'exit', or press Ctrl+C to stop.
========================================================

[Listening...] Speak into your microphone...

You: What is artificial intelligence?
AURA: Artificial intelligence is the field of computer science dedicated to creating systems capable of performing tasks that typically require human intelligence, such as visual perception, decision-making, and natural language understanding.

[Listening...] Speak into your microphone...

You: Goodbye
AURA: Goodbye! Have a wonderful day!

--- AURA Voice Session Ended ---
```

### Interactive Text Console Mode
If you do not have a microphone or prefer text:
```powershell
python -m app.main --text
```

### Automated Pipeline Demo
To run the automated multi-turn pipeline demo (Tools, RAG, and Memory):
```powershell
python -m app.main --demo
```

---

## 8. Testing

AURA uses `pytest` for automated unit and integration tests. All audio hardware and external services are cleanly mocked, allowing the entire suite to run offline without a microphone or network connection.

Run the test suite from the project root:

```powershell
pytest -v
```

All 165 tests verify:
- Speech-to-Text (transcription, silence, timeout, recognition errors, microphone hardware errors).
- Text-to-Speech (speech playback, empty text handling, provider and audio driver failure handling).
- End-to-end Voice Loop integration with Intent Classification, Memory, RAG, and LLM Brain.
- Dual-tier conversation memory buffering, tool sandboxing, and RAG vector store retrieval.

All foundation tests verify:
- Package integrity and module importability.
- Short-term conversation memory buffering and trimming.
- Defensive initialization of tools and intent stubs.

---

## 9. How the AI Pipeline Works

1. **Audio Capture**: The user speaks; `speech.stt` captures the microphone stream and performs acoustic modeling and decoding into text.
2. **Intent & Routing**: `brain.intent` classifies the query (e.g., smalltalk, calculation, web lookup, file inspection).
3. **Context Assembly**: `memory.conversation` provides short-term conversation context; `rag.retriever` fetches relevant reference documents if needed.
4. **Brain Reasoning**: `brain.llm` receives the prompt, assembled context, and system persona to generate a structured or conversational response.
5. **Tool Execution**: If a tool call is required (e.g., computing an expression or reading a file in `./data`), the tool runs within a strictly validated sandbox.
6. **Voice Synthesis**: The final response text is sent to `speech.tts`, which synthesizes spoken audio and streams it to the user.

---

## 10. Technologies Used

- **Language**: Python 3.11+
- **Configuration & Secrets**: `python-dotenv`
- **Testing**: `pytest`
- **HTTP / APIs**: `requests`
- **Future ML / Speech Libraries**: PyTorch, SpeechRecognition, pyttsx3, NumPy (to be introduced in their respective phases)

---

## 11. Future Improvements

- PyTorch-based neural intent classification model.
- Streaming responses for low-latency voice synthesis.
- Local vector database (FAISS / ChromaDB / SQLite-VSS) for scalable memory.
- Wake-word detection engine (e.g., "Hey Aura").

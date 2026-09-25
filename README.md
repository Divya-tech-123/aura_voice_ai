# 🤖 AURA — AI Unified Response Assistant

> **A modular AI voice assistant built with Python, combining Speech Recognition, LLM orchestration, RAG, memory, tools, and Text-to-Speech into one intelligent pipeline.**

AURA is a learning-focused AI assistant designed to demonstrate how multiple AI components can work together in a real application.

Instead of treating AI as a single API call, AURA separates the system into independent modules for **speech processing, intent understanding, LLM orchestration, memory, tools, and Retrieval-Augmented Generation (RAG).**

---

## ✨ Project Overview

AURA allows users to interact with an AI assistant through **voice or text**.

The assistant can:

* 🎙️ Accept voice input
* 📝 Process text queries
* 🧠 Understand user intent
* 🤖 Communicate with an LLM
* 📚 Retrieve information from documents using RAG
* 💾 Maintain conversational memory
* 🧮 Execute tools such as calculations
* 🌦️ Retrieve weather information
* 🔎 Perform web searches
* 📁 Work with files inside a controlled sandbox
* 🔊 Convert responses back into speech

The main goal of the project was to understand **how these individual AI concepts connect together to form a complete AI assistant.**

---

# 🧠 What I Learned

Building AURA helped me move from learning individual AI concepts to integrating them into one system.

### Core concepts explored

**Artificial Intelligence**

↓

**LLMs**

↓

**Intent Classification**

↓

**Memory**

↓

**Embeddings & Vector Search**

↓

**RAG**

↓

**Tools / Agent Workflows**

↓

**Speech Recognition**

↓

**Text-to-Speech**

↓

**Complete AI Assistant**

---

# 🏗️ Architecture

```text
                    👤 USER
                       │
             ┌─────────┴─────────┐
             │                   │
          🎙️ Voice             📝 Text
             │                   │
             ▼                   │
       Speech-to-Text             │
             │                   │
             └─────────┬─────────┘
                       ▼
              🧠 Intent / Query
                 Understanding
                       │
                       ▼
              ┌─────────────────┐
              │    AURA BRAIN   │
              │  LLM + Routing  │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   🤖 LLM          💾 Memory       🛠️ Tools
                    │               │
             ┌──────┴──────┐    ┌───┴────────┐
             │             │    │            │
         Short-term   Vector Memory  Calculator
                                     Weather
                                     Web Search
                                     Safe Files
                       │
                       ▼
                 📚 RAG Pipeline
                       │
             Documents → Chunks
                       │
                  Embeddings
                       │
                 Vector Search
                       │
                       ▼
              Relevant Context
                       │
                       ▼
                 🤖 LLM Response
                       │
                       ▼
               🔊 Text-to-Speech
                       │
                       ▼
                   👤 USER
```

---

# 🔄 How AURA Works

### 1. 🎙️ Input

The user can communicate with AURA using:

* Voice
* Text

For voice interaction, the microphone input is processed by the Speech-to-Text module.

### 2. 🧠 Intent Understanding

AURA analyzes the query and determines what type of operation is required.

Examples:

```text
Small Talk
Calculation
Weather
Web Search
File Operation
Document Question
General AI Question
```

### 3. 💾 Memory

AURA maintains conversational context using two levels of memory:

**Short-Term Memory**

Stores recent conversation context.

**Long-Term Vector Memory**

Stores semantic information that can be retrieved when relevant.

### 4. 📚 RAG

For document-related questions, AURA can retrieve relevant information from indexed documents.

The process is:

```text
Documents
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Store
   ↓
Similarity Search
   ↓
Relevant Context
   ↓
LLM
   ↓
Answer
```

### 5. 🛠️ Tools

Depending on the query, AURA can route tasks to tools such as:

* Calculator
* Weather lookup
* Web search
* Sandboxed file operations

### 6. 🤖 LLM Response

The LLM receives the user's query together with the required context, memory, retrieved information, and tool results.

### 7. 🔊 Voice Output

The final response can be converted into speech using Text-to-Speech.

---

# 🚀 Features

| Feature                  | Description                                        |
| ------------------------ | -------------------------------------------------- |
| 🎙️ Speech-to-Text       | Converts microphone speech into text               |
| 🔊 Text-to-Speech        | Converts AI responses into spoken audio            |
| 🧠 Intent Classification | Routes queries to appropriate functionality        |
| 🤖 LLM Integration       | Provider-agnostic LLM architecture                 |
| 💾 Conversation Memory   | Maintains conversational context                   |
| 📚 RAG                   | Answers questions using retrieved document context |
| 🧮 Calculator            | Performs arithmetic operations                     |
| 🌦️ Weather              | Retrieves weather information                      |
| 🔎 Web Search            | Performs web search operations                     |
| 📁 Safe Files            | Controlled filesystem operations                   |
| 🧪 Testing               | Automated unit and integration testing             |

---

# 📂 Project Structure

```text
aura-ai-voice/
│
├── app/
│   │
│   ├── main.py
│   │
│   ├── speech/
│   │   ├── __init__.py
│   │   ├── stt.py
│   │   └── tts.py
│   │
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── intent.py
│   │   └── prompts.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── conversation.py
│   │   └── vector_memory.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   ├── weather.py
│   │   ├── search.py
│   │   └── files.py
│   │
│   └── rag/
│       ├── __init__.py
│       ├── loader.py
│       ├── embeddings.py
│       └── retriever.py
│
├── models/
├── data/
├── tests/
│   ├── __init__.py
│   ├── test_foundation.py
│   ├── test_intent.py
│   ├── test_memory.py
│   └── test_tools.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language

* Python 3.11+

### AI / LLM

* LLM provider abstraction
* OpenAI
* Google Gemini
* Prompt engineering
* Intent classification

### RAG

* Document loading
* Text chunking
* Embeddings
* Vector storage
* Cosine similarity retrieval

### Voice

* SpeechRecognition
* Speech-to-Text
* pyttsx3
* Text-to-Speech

### Development

* Python
* pytest
* requests
* python-dotenv

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/aura-ai-voice.git

cd aura-ai-voice
```

Replace `YOUR-USERNAME` with your GitHub username.

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file based on `.env.example`.

Example:

```env
APP_NAME=AURA
APP_ENV=development
LOG_LEVEL=INFO

LLM_PROVIDER=mock

OPENAI_API_KEY=
GEMINI_API_KEY=

SAFE_DATA_DIR=./data

WEATHER_API_KEY=
SERPAPI_API_KEY=

STT_PROVIDER=local
STT_TIMEOUT=5
STT_PHRASE_TIME_LIMIT=10
STT_ENERGY_THRESHOLD=300

TTS_PROVIDER=pyttsx3
TTS_VOICE_RATE=180
TTS_VOLUME=1.0

AURA_MODE=voice
```

⚠️ **Never upload your real API keys to GitHub.**

Use `.env` locally and keep it inside `.gitignore`.

---

# ▶️ Running AURA

## 🎙️ Voice Mode

Start AURA using your microphone:

```bash
python -m app.main
```

Example:

```text
========================================================
             AURA Interactive Voice Mode
========================================================

[Listening...]

You: What is artificial intelligence?

AURA: Artificial intelligence is the field of computer
science dedicated to creating systems capable of performing
tasks that typically require human intelligence.

[Listening...]

You: Goodbye

AURA: Goodbye! Have a wonderful day!

========================================================
             AURA Voice Session Ended
========================================================
```

---

# 📝 Text Mode

If you don't have a microphone or want to test using text:

```bash
python -m app.main --text
```

---

# 🧪 Demo Mode

Run the automated demonstration of the main pipeline:

```bash
python -m app.main --demo
```

This demonstrates functionality involving:

* Tools
* RAG
* Memory
* AI pipeline integration

---

# 🧪 Testing

AURA uses **pytest** for automated testing.

Run:

```bash
pytest -v
```

The test suite covers areas including:

* Speech-to-Text
* Text-to-Speech
* Voice loop integration
* Intent classification
* Conversation memory
* Vector memory
* Tool execution
* Tool sandboxing
* RAG retrieval
* Module integrity

Audio hardware and external services can be mocked so that core tests can run without requiring a microphone or live external service.

---

# 📊 Development Roadmap

### ✅ Phase 1 — Foundation

* Modular project structure
* Logging
* Environment management
* Test infrastructure

### ✅ Phase 2 — Intent Classification

* Query preprocessing
* Intent recognition
* Fallback routing

### ✅ Phase 3 — LLM Brain

* Provider abstraction
* Mock LLM
* OpenAI integration
* Gemini integration
* Prompt templates

### ✅ Phase 4 — Memory

* Short-term conversation memory
* Long-term semantic memory
* Vector-based retrieval

### ✅ Phase 5 — Tools

* Calculator
* Weather
* Web search
* Safe file operations

### ✅ Phase 6 — RAG

* Document loading
* Text chunking
* Embeddings
* Vector retrieval

### ✅ Phase 7 — Voice

* Speech-to-Text
* Text-to-Speech
* Microphone handling
* Voice interaction loop
* Error handling

### 🔄 Phase 8 — Advanced Improvements

Planned improvements include:

* More robust autonomous workflows
* Continuous listening improvements
* Better voice interaction
* Streaming responses
* Improved deployment
* Enhanced UI experience

---

# 🧩 AI Pipeline

The complete processing pipeline can be summarized as:

```text
User
 │
 ▼
Voice / Text Input
 │
 ▼
Speech-to-Text
 │
 ▼
Intent Classification
 │
 ▼
Context Assembly
 │
 ├── Conversation Memory
 │
 ├── Vector Memory
 │
 ├── RAG Retrieval
 │
 └── Tools
       │
       ▼
    LLM Brain
       │
       ▼
   AI Response
       │
       ▼
Text-to-Speech
       │
       ▼
Voice Output
```

---

# 🎯 Why I Built AURA

The purpose of AURA was not simply to create another chatbot.

I built it to understand how different AI concepts work together inside a real application.

Through this project, I explored:

* How voice interfaces communicate with AI systems
* How LLMs can be integrated into modular applications
* How RAG provides external knowledge to an LLM
* How embeddings enable semantic retrieval
* How memory maintains conversational context
* How tools extend an AI assistant's capabilities
* How software testing applies to AI applications
* How real-world debugging affects AI projects

---

# 🐛 Challenges During Development

Building AURA involved several practical engineering challenges.

### 🎙️ Microphone & Voice Processing

Handling microphone input requires dealing with:

* Ambient noise
* Timeouts
* Silent input
* Speech recognition errors
* Hardware/device issues

### 🔗 API Integration

Different external services require:

* API configuration
* Error handling
* Provider abstraction
* Fallback behavior

### 📚 RAG

Building document-based question answering required understanding:

```text
Documents
→ Chunking
→ Embeddings
→ Vector Representation
→ Similarity Search
→ Relevant Context
→ LLM
```

### 🚀 Deployment

The local application and its deployment environment have different requirements, especially when working with:

* Microphone access
* Audio hardware
* Python dependencies
* Serverless environments
* Frontend/backend integration

Deployment and UI integration are therefore an ongoing improvement area for the project.

---

# 🔮 Future Improvements

Potential future improvements include:

* 🧠 Neural intent classification
* ⚡ Streaming AI responses
* 🎙️ Better real-time voice interaction
* 🔊 Lower-latency speech synthesis
* 🗄️ Scalable vector database
* 👋 Wake-word detection such as "Hey AURA"
* 🌐 Improved web deployment
* 🎨 Dedicated production-quality UI
* 🔐 Stronger security and permission controls

---

# 📸 Project Demo

> Add screenshots or a short demo video here.

Recommended screenshots:

```text
1. AURA Home / Interface
2. Text interaction
3. Voice interaction
4. RAG document question
5. Memory demonstration
6. Project architecture
```

Example:

```markdown
![AURA Interface](assets/aura-interface.png)
```


# 👩‍💻 Project

**AURA — AI Unified Response Assistant**

Built as a hands-on learning project to explore AI systems, LLM applications, RAG, memory, tools, and voice interaction.

---

# ⭐ If You Find This Project Interesting

Feel free to explore the repository, experiment with the modules, and learn from the implementation.

If you are also learning AI, I'd love to connect and learn together.

---

## 📌 Learning by Building

> **Don't just learn AI concepts. Build systems with them.**

AURA represents one step in my journey toward understanding and building practical AI applications.

---

### 📄 License

Add your preferred license here, for example:

```text
MIT License
```

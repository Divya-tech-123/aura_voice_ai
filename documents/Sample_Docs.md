# AURA System Architecture Reference

## Overview
AURA (AI Unified Response Assistant) is a modular voice and decision assistant built with decoupled components:
- Speech Recognition (STT) and Synthesis (TTS)
- Intent Classification Engine (Neural & Preprocessing)
- LLM Brain Orchestration (Pluggable providers: Mock, OpenAI, Gemini)
- Dual-tier Memory (Short-term conversational buffer and long-term semantic store)
- Sandboxed Tools (Safe AST Calculator, Weather, Web Search, Sandboxed Files)
- Retrieval-Augmented Generation (RAG pipeline for grounding responses in user documents)

## RAG Design Principles
AURA\'s RAG system adheres to strict pedagogical principles:
1. Deterministic Local Embedding: Offline feature hashing unit vectors ensure privacy and reproducible evaluations.
2. Word-Boundary Chunking: Overlapping sliding windows preserve cross-boundary semantics.
3. Explicit Source Attribution: Answers strictly attribute verified facts to the source filename and page/chunk.

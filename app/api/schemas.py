"""Data schemas for AURA REST API.

Defines Pydantic models with input validation and constraints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Schema for incoming chat messages sent to AURA."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user's query or instruction for AURA.",
        examples=["What is artificial intelligence?"],
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Ensure message is not empty and not solely whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or solely whitespace.")
        return cleaned


class ActivityItem(BaseModel):
    """Schema for a single high-level, user-safe agent activity entry."""

    type: str = Field(
        ...,
        description="Type of agent activity (e.g. 'planning', 'tool', 'retrieval', 'response').",
        examples=["planning", "tool", "response"],
    )
    label: str = Field(
        ...,
        description="Concise, user-safe description of the activity.",
        examples=["Planning task", "Using Calculator", "Generating response"],
    )
    status: str = Field(
        default="completed",
        description="Execution status of the activity: 'completed', 'failed', or 'in_progress'.",
        examples=["completed", "failed"],
    )


class ChatResponse(BaseModel):
    """Schema for AURA's response to the user."""

    response: str = Field(
        ...,
        description="AURA's generated text response.",
    )
    activities: Optional[List[ActivityItem]] = Field(
        default_factory=list,
        description="Structured high-level agent activity steps.",
    )


class HealthResponse(BaseModel):
    """Schema for API health status."""

    status: str = "ok"
    version: Optional[str] = "0.11.0"
    app: str = "AURA"


class VoiceRequest(BaseModel):
    """Optional JSON payload for voice input containing base64 encoded audio."""

    audio: str = Field(
        ...,
        description="Base64 encoded audio bytes or data URI.",
    )
    format: Optional[str] = Field(
        default="audio/wav",
        description="MIME type or audio format (e.g. 'audio/wav', 'audio/webm').",
    )


class VoiceResponse(BaseModel):
    """Schema for AURA's voice response."""

    user_text: str = Field(
        ...,
        description="Transcribed user speech recognized by STT.",
    )
    response: str = Field(
        ...,
        description="AURA's generated text response from AuraAgent.",
    )
    audio: Optional[str] = Field(
        default=None,
        description="Synthesized speech audio as a base64 data URI (data:audio/wav;base64,...), or None if TTS failed.",
    )
    audio_format: Optional[str] = Field(
        default="audio/wav",
        description="MIME type of the returned audio.",
    )
    activities: Optional[List[ActivityItem]] = Field(
        default_factory=list,
        description="Structured high-level agent activity steps.",
    )


class ErrorDetail(BaseModel):
    """Safe structured error model that does not expose internal stack traces."""

    error: str = Field(
        default="Unable to process the request.",
        description="Safe, human-readable error description.",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Safe detailed message for backward compatibility.",
    )

    def __init__(self, **data):
        if "detail" in data and "error" not in data:
            data["error"] = data["detail"]
        elif "error" in data and "detail" not in data:
            data["detail"] = data["error"]
        super().__init__(**data)


class DocumentUploadResponse(BaseModel):
    """Structured response returned after document upload and indexing."""

    success: bool = Field(..., description="Whether the document was successfully processed and indexed.")
    filename: str = Field(..., description="Sanitized base filename of the uploaded document.")
    message: str = Field(..., description="Human-readable status or result message.")
    chunks_indexed: Optional[int] = Field(default=None, description="Number of text chunks indexed.")


class DocumentItem(BaseModel):
    """Represents an uploaded document in AURA's knowledge base."""

    filename: str = Field(..., description="Sanitized base filename of the document.")
    status: str = Field(default="indexed", description="Current status of the document.")
    size: Optional[int] = Field(default=None, description="Document size in bytes.")


class DocumentListResponse(BaseModel):
    """List of documents currently in AURA's knowledge base."""

    documents: List[DocumentItem] = Field(default_factory=list, description="Collection of indexed documents.")



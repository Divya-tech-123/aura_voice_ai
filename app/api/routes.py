import os
import sys
import logging
import base64
import re
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request, status, File, UploadFile
from app.config import get_settings
from app.api.schemas import (
    ActivityItem,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    VoiceResponse,
    ErrorDetail,
    DocumentUploadResponse,
    DocumentItem,
    DocumentListResponse,
)
from app.api.dependencies import get_agent, get_stt, get_tts, get_documents_dir
from app.agent.agent import AuraAgent
from app.agent.state import AgentStatus
from app.rag.loader import index_file
from app.speech.stt import (
    BaseSpeechToText,
    NoSpeechDetectedError,
    SpeechRecognitionError,
    AudioDecodeError,
    SpeechServiceUnavailableError,
)
from app.speech.tts import (
    BaseTextToSpeech,
    TTSError,
)

logger = logging.getLogger("AURA.API.Routes")

router = APIRouter(prefix="/api")

# Default upload limits derived from centralized settings
MAX_AUDIO_SIZE = get_settings().max_audio_size
MAX_DOCUMENT_SIZE = get_settings().max_document_size
ALLOWED_DOC_EXTENSIONS = get_settings().allowed_doc_extensions

# Dangerous executable extensions to prevent disguised execution
DANGEROUS_EXTENSIONS = {
    ".exe", ".bat", ".cmd", ".sh", ".py", ".pyw", ".js", ".vbs", ".msi", ".ps1", ".php", ".phtml", ".dll", ".so"
}

# Allowed audio MIME types for security validation
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/webm",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp4",
    "audio/x-m4a",
    "application/octet-stream",
}


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Returns service health status and API information without exposing internal system details.",
)
def health_check() -> HealthResponse:
    """Verify that the AURA API backend is running and healthy."""
    return HealthResponse(status="ok", version="0.11.0", app="AURA")


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    responses={
        400: {"model": ErrorDetail, "description": "Invalid or empty message"},
        422: {"description": "Validation error"},
        500: {"model": ErrorDetail, "description": "Agent execution error"},
    },
    summary="Send a chat message to AURA Agent",
    description="Processes the message using the autonomous AuraAgent (memory, tools, planner, and RAG).",
)
def chat(
    request: ChatRequest,
    agent: AuraAgent = Depends(get_agent),
) -> ChatResponse:
    """Process a user chat message through the central AuraAgent.

    Validates message content, runs the cognitive loop, handles errors gracefully,
    and returns the agent's textual response.
    """
    clean_message = request.message.strip()
    if not clean_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty or contain only whitespace.",
        )

    try:
        # Execute the goal using the central AuraAgent
        state = agent.process_goal(clean_message)

        if state.status == AgentStatus.FAILED and not state.final_response:
            logger.error(f"AuraAgent failed without final response: {state.error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while processing your request.",
            )

        response_text = state.final_response or "I received your message but have no response to provide."
        raw_activities = state.get_activities() if hasattr(state, "get_activities") else getattr(state, "activities", [])
        activities = []
        for act in (raw_activities or []):
            if isinstance(act, dict):
                activities.append(ActivityItem(
                    type=act.get("type", "action"),
                    label=act.get("label", "Processing"),
                    status=act.get("status", "completed"),
                ))
            elif isinstance(act, ActivityItem):
                activities.append(act)

        return ChatResponse(response=response_text, activities=activities)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as exc:
        # Log internal error securely without leaking stack traces or sensitive env to user
        logger.exception(f"Unexpected error in chat endpoint: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request.",
        )


@router.post(
    "/voice",
    response_model=VoiceResponse,
    tags=["Voice"],
    responses={
        400: {"model": ErrorDetail, "description": "Invalid, empty, or unparseable audio"},
        413: {"model": ErrorDetail, "description": "Audio payload exceeds maximum size limit"},
        415: {"model": ErrorDetail, "description": "Unsupported audio format"},
        500: {"model": ErrorDetail, "description": "Internal agent or server execution error"},
    },
    summary="Send recorded voice audio to AURA",
    description="Processes audio through STT, generates cognitive response with AuraAgent, and synthesizes audio response via TTS.",
)
async def voice(
    request: Request,
    agent: AuraAgent = Depends(get_agent),
    stt: BaseSpeechToText = Depends(get_stt),
    tts: BaseTextToSpeech = Depends(get_tts),
) -> VoiceResponse:
    """End-to-end voice pipeline: Audio Input -> STT -> AuraAgent -> TTS -> Playable Audio.

    Accepts binary audio, multipart form-data, or base64 JSON payload.
    Validates audio format and size, converts speech to text, generates intelligent response,
    and returns textual and synthesized audio output.
    """
    content_type_header = request.headers.get("content-type", "").lower()
    audio_bytes: bytes = b""
    content_type: str = "audio/wav"

    # 1. Parse incoming audio according to Content-Type
    if "multipart/form-data" in content_type_header:
        try:
            form = await request.form()
            upload_file = form.get("audio") or form.get("file")
            if upload_file is None:
                for v in form.values():
                    if hasattr(v, "read"):
                        upload_file = v
                        break
            if upload_file is not None and hasattr(upload_file, "read"):
                audio_bytes = await upload_file.read()
                content_type = getattr(upload_file, "content_type", None) or "audio/wav"
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No audio file found in form data.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(f"Failed to parse multipart form data: {exc}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to parse multipart audio upload.",
            )
    elif "application/json" in content_type_header:
        try:
            body_json = await request.json()
            raw_b64 = body_json.get("audio", "")
            content_type = body_json.get("format", "audio/wav")
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            audio_bytes = base64.b64decode(raw_b64)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON payload or malformed base64 audio data.",
            )
    else:
        # Raw binary audio body
        audio_bytes = await request.body()
        content_type = content_type_header.split(";")[0].strip() or "audio/wav"

    # 2. Security & Format Validation
    base_content_type = content_type.split(";")[0].strip().lower()
    logger.info(
        f"[Voice API] Received voice request: Header='{content_type_header}', ResolvedType='{content_type}', BaseType='{base_content_type}', Size={len(audio_bytes)} bytes"
    )

    if base_content_type and base_content_type not in ALLOWED_AUDIO_TYPES:
        logger.warning(f"[Voice API] Rejected unsupported audio type: '{base_content_type}'")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format: '{base_content_type}'. Supported formats: audio/wav, audio/webm, audio/ogg.",
        )

    if not audio_bytes or len(audio_bytes.strip() if hasattr(audio_bytes, "strip") else audio_bytes) == 0:
        logger.warning("[Voice API] Audio payload is empty or missing (0 bytes received).")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Recorded audio is empty or missing (0 bytes received). Please speak into the microphone and try again.",
        )

    if len(audio_bytes) > MAX_AUDIO_SIZE:
        logger.warning(f"[Voice API] Audio payload exceeds maximum limit ({len(audio_bytes)} > {MAX_AUDIO_SIZE} bytes).")
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Audio payload exceeds maximum size limit of {MAX_AUDIO_SIZE // (1024 * 1024)} MB.",
        )

    # 3. Step 1: Speech-to-Text (STT) Transcription
    logger.info(f"[Voice API] Transcribing audio with STT engine ({len(audio_bytes)} bytes, format='{content_type}')...")
    try:
        user_text = stt.transcribe_audio_data(audio_bytes, content_type=content_type)
    except AudioDecodeError as decode_err:
        logger.warning(f"[Voice API] Audio decode failure: {decode_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio could not be decoded: {decode_err}",
        )
    except NoSpeechDetectedError as no_speech_err:
        logger.warning(f"[Voice API] No speech detected in audio: {no_speech_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not understand speech from audio: {no_speech_err}",
        )
    except SpeechServiceUnavailableError as service_err:
        logger.error(f"[Voice API] Speech recognition service error: {service_err}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Speech recognition service error: {service_err}",
        )
    except SpeechRecognitionError as stt_err:
        logger.warning(f"[Voice API] STT recognition failed: {stt_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not understand speech from audio. Please speak clearly and try again. ({stt_err})",
        )
    except Exception as stt_exc:
        logger.error(f"[Voice API] Unexpected error in STT subsystem: {stt_exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while transcribing speech.",
        )

    if not user_text or not user_text.strip():
        logger.warning("[Voice API] STT returned empty transcription.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not understand speech from audio: No speech detected in audio.",
        )

    user_text = user_text.strip()
    logger.info(f"[Voice API] Voice recognized utterance: '{user_text}'")

    # 4. Step 2: Autonomous Agent Processing
    try:
        state = agent.process_goal(user_text)

        if state.status == AgentStatus.FAILED and not state.final_response:
            logger.error(f"AuraAgent failed during voice request: {state.error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while processing your request.",
            )

        response_text = state.final_response or "I received your voice message but have no response to provide."
    except HTTPException:
        raise
    except Exception as agent_exc:
        logger.exception(f"Unexpected agent error during voice request: {agent_exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing your request.",
        )

    # 5. Step 3 & 4: Text-to-Speech (TTS) Synthesis
    audio_data_url: Optional[str] = None
    try:
        audio_bytes_out = tts.synthesize_to_bytes(response_text)
        if audio_bytes_out:
            b64_audio = base64.b64encode(audio_bytes_out).decode("utf-8")
            audio_data_url = f"data:audio/wav;base64,{b64_audio}"
    except (TTSError, Exception) as tts_err:
        # If TTS fails, preserve text response and log warning
        logger.warning(f"TTS synthesis failed during voice request: {tts_err}")
        audio_data_url = None

    # Build safe structured activities
    raw_activities = state.get_activities() if hasattr(state, "get_activities") else getattr(state, "activities", [])
    voice_activities = []
    for act in (raw_activities or []):
        if isinstance(act, dict):
            voice_activities.append(ActivityItem(
                type=act.get("type", "action"),
                label=act.get("label", "Processing"),
                status=act.get("status", "completed"),
            ))
        elif isinstance(act, ActivityItem):
            voice_activities.append(act)

    return VoiceResponse(
        user_text=user_text,
        response=response_text,
        audio=audio_data_url,
        audio_format="audio/wav",
        activities=voice_activities,
    )


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
    responses={
        400: {"model": ErrorDetail, "description": "Invalid file, unsupported type, empty file, or unsafe path"},
        413: {"model": ErrorDetail, "description": "Uploaded file exceeds maximum allowed size limit (10 MB)"},
        422: {"model": ErrorDetail, "description": "Document indexing failed or text could not be extracted"},
        500: {"model": ErrorDetail, "description": "Server processing error"},
    },
    summary="Upload and index a document for RAG",
    description="Validates, safely stores, and indexes a PDF, DOCX, or TXT document into AURA's vector store.",
)
async def upload_document(
    file: UploadFile = File(...),
    agent: AuraAgent = Depends(get_agent),
    doc_dir: Path = Depends(get_documents_dir),
) -> DocumentUploadResponse:
    """Safely upload and index a user document into AURA's knowledge base.

    Performs security validation:
    1. Rejects path traversal sequences (..), directory separators, and control characters.
    2. Enforces allowed extensions (.pdf, .docx, .txt).
    3. Prevents dangerous disguised executable extensions (e.g. .exe.txt).
    4. Enforces file size limits with streaming chunk reading.
    5. Ensures file is stored strictly within AURA's controlled documents directory without executable permissions.
    6. Feeds document into existing RAG indexer and updates the active AuraAgent retriever.
    7. Returns structured response without exposing filesystem paths.
    """
    raw_filename = file.filename or ""
    if not raw_filename or not raw_filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty.",
        )

    # 1. Path traversal and separator check
    if ".." in raw_filename or "/" in raw_filename or "\\" in raw_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: path traversal sequences or directory separators are not allowed.",
        )

    # 2. Control characters & null bytes check
    if any(ord(c) < 32 or ord(c) == 127 for c in raw_filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: control characters or null bytes are not allowed.",
        )

    # 3. Extract and sanitize basename
    basename = Path(raw_filename).name.strip()
    if not basename or basename.startswith("."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: hidden or empty filenames are not allowed.",
        )

    # 4. Check for illegal characters (allow letters, numbers, hyphens, underscores, dots, and spaces)
    if not re.match(r"^[a-zA-Z0-9_\-. ]+$", basename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: contains unsafe characters. Only alphanumeric characters, dashes, underscores, dots, and spaces are allowed.",
        )

    # 5. Validate file extension against allowed whitelist
    ext = Path(basename).suffix.lower()
    allowed_exts = getattr(sys.modules[__name__], "ALLOWED_DOC_EXTENSIONS", ALLOWED_DOC_EXTENSIONS)
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Supported formats are: PDF, DOCX, TXT.",
        )

    # 6. Check for disguised dangerous executable extensions (e.g. malicious.exe.txt)
    stem_lower = Path(basename).stem.lower()
    for dang_ext in DANGEROUS_EXTENSIONS:
        if stem_lower.endswith(dang_ext):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename: dangerous executable pattern detected.",
            )

    # 7. Verify resolved storage path is strictly inside controlled doc_dir
    dest_path = (doc_dir / basename).resolve()
    if dest_path.parent != doc_dir.resolve():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal attempt detected.",
        )

    # 8. Read content enforcing file size limit with streaming chunks
    effective_max_size = getattr(sys.modules[__name__], "MAX_DOCUMENT_SIZE", MAX_DOCUMENT_SIZE)
    chunk_size = 64 * 1024
    total_bytes = 0
    chunks_data = []

    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > effective_max_size:
                raise HTTPException(
                    status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                    detail=f"Uploaded file exceeds maximum allowed size of {effective_max_size} bytes.",
                )
            chunks_data.append(chunk)
    except HTTPException:
        raise
    except Exception as read_err:
        logger.error(f"Error reading uploaded file: {read_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        )

    content = b"".join(chunks_data)
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 9. Write to controlled documents directory as a static passive file (no execution)
    try:
        dest_path.write_bytes(content)
        try:
            os.chmod(dest_path, 0o644)
        except (AttributeError, OSError):
            pass
    except Exception as write_err:
        logger.error(f"Failed to write uploaded file to {dest_path}: {write_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded document.",
        )

    # 10. Index document using existing RAG pipeline
    try:
        # Ensure agent has an active retriever instance
        if agent.retriever is None:
            from app.rag.retriever import LocalRetriever
            agent.retriever = LocalRetriever()

        index_result = index_file(dest_path, retriever=agent.retriever)

        if index_result.get("chunks_indexed", 0) == 0:
            # Clean up unparseable/empty document from disk
            dest_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document indexing failed: could not extract readable text from document.",
            )

        logger.info(
            f"Successfully indexed document '{basename}': "
            f"{index_result.get('chunks_indexed')} chunk(s) indexed."
        )

        return DocumentUploadResponse(
            success=True,
            filename=basename,
            message="Document indexed successfully",
            chunks_indexed=index_result.get("chunks_indexed"),
        )

    except HTTPException:
        raise
    except Exception as exc:
        # Clean up file on unexpected indexing failure
        dest_path.unlink(missing_ok=True)
        logger.exception(f"Unexpected error during document indexing: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document indexing failed: {str(exc)}",
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["Documents"],
    summary="List indexed knowledge base documents",
    description="Returns the list of documents available in AURA's controlled knowledge base.",
)
def list_documents(
    doc_dir: Path = Depends(get_documents_dir),
) -> DocumentListResponse:
    """Retrieve metadata of uploaded and indexed documents without exposing internal server paths."""
    docs = []
    if doc_dir.exists() and doc_dir.is_dir():
        for entry in sorted(doc_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in ALLOWED_DOC_EXTENSIONS:
                try:
                    file_size = entry.stat().st_size
                except OSError:
                    file_size = None
                docs.append(DocumentItem(
                    filename=entry.name,
                    status="indexed",
                    size=file_size,
                ))
    return DocumentListResponse(documents=docs)



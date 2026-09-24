"""Safe sandboxed filesystem tool for AURA.

Enables safe file inspection, reading, and metadata querying strictly confined
within a designated sandbox directory, defensively preventing path traversal attacks.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Safe File Manager Engine
# ==============================================================================

class SafeFileManager:
    """Manages file operations confined strictly inside a sandbox root directory."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize safe file manager sandbox.

        Args:
            base_dir: Root directory for safe operations. Defaults to SAFE_DATA_DIR env var or './data'.
        """
        raw_base = base_dir or os.getenv("SAFE_DATA_DIR", "./data")
        self.base_dir: Path = Path(raw_base).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def resolve_safe_path(self, relative_path: str) -> Path:
        """Resolve a path and ensure it is safely located within base_dir.

        Strictly prevents path traversal attacks (e.g. `../../`, absolute root escapes).

        Args:
            relative_path: The relative file path inside the safe sandbox.

        Returns:
            Resolved absolute Path guaranteed to be inside base_dir.

        Raises:
            PermissionError: If the resolved path points outside base_dir.
            ValueError: If the relative path contains illegal characters.
        """
        clean_path = str(relative_path or "").strip()

        # Handle root / empty relative path
        if not clean_path or clean_path in (".", "./", ".\\"):
            return self.base_dir

        # Combine with base_dir and resolve canonical path
        # If relative_path is absolute, Path(base_dir) / abs_path switches to abs_path in Path semantics.
        # .resolve() then evaluates symlinks and removes `..` segments.
        candidate = (self.base_dir / clean_path).resolve()

        # Defend sandbox boundary
        try:
            is_inside = candidate.is_relative_to(self.base_dir)
        except AttributeError:
            # Fallback for Python < 3.9 if ever run
            try:
                candidate.relative_to(self.base_dir)
                is_inside = True
            except ValueError:
                is_inside = False

        if not is_inside:
            raise PermissionError(
                f"Security restriction: Access to path '{clean_path}' outside sandbox "
                f"'{self.base_dir}' is blocked."
            )

        return candidate

    def list_files(self, subpath: str = "") -> Dict[str, Any]:
        """List files and subdirectories located within the safe sandbox.

        Args:
            subpath: Optional subdirectory inside the sandbox.

        Returns:
            Dict with 'success' (bool), 'data' (dict/None), and 'error' (str/None).
        """
        try:
            target = self.resolve_safe_path(subpath)
        except PermissionError as perm_err:
            return {"success": False, "data": None, "error": str(perm_err)}
        except Exception as exc:
            return {"success": False, "data": None, "error": f"Invalid path: {exc}"}

        if not target.exists():
            return {
                "success": False,
                "data": None,
                "error": f"Directory does not exist: '{subpath}'",
            }

        if not target.is_dir():
            return {
                "success": False,
                "data": None,
                "error": f"Path '{subpath}' is a file, not a directory.",
            }

        try:
            items: List[Dict[str, Any]] = []
            for entry in sorted(target.iterdir()):
                rel = entry.relative_to(self.base_dir).as_posix()
                items.append({
                    "name": entry.name,
                    "path": rel,
                    "is_file": entry.is_file(),
                    "is_dir": entry.is_dir(),
                    "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                })

            rel_dir = target.relative_to(self.base_dir).as_posix()
            return {
                "success": True,
                "data": {
                    "directory": "." if rel_dir == "." else rel_dir,
                    "count": len(items),
                    "items": items,
                },
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to list directory contents: {exc}",
            }

    def read_file(self, relative_path: str, max_chars: int = 100_000) -> Dict[str, Any]:
        """Read text content from a file inside the safe sandbox.

        Args:
            relative_path: Path to the file inside sandbox.
            max_chars: Maximum character limit to prevent memory exhaustion.

        Returns:
            Dict with 'success' (bool), 'data' (dict/None), and 'error' (str/None).
        """
        try:
            target = self.resolve_safe_path(relative_path)
        except PermissionError as perm_err:
            return {"success": False, "data": None, "error": str(perm_err)}
        except Exception as exc:
            return {"success": False, "data": None, "error": f"Invalid path: {exc}"}

        if not target.exists():
            return {
                "success": False,
                "data": None,
                "error": f"File does not exist: '{relative_path}'",
            }

        if target.is_dir():
            return {
                "success": False,
                "data": None,
                "error": f"Target path '{relative_path}' is a directory, not a readable file.",
            }

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            is_truncated = False
            if len(content) > max_chars:
                content = content[:max_chars]
                is_truncated = True

            rel_path = target.relative_to(self.base_dir).as_posix()
            return {
                "success": True,
                "data": {
                    "path": rel_path,
                    "content": content,
                    "truncated": is_truncated,
                    "size_bytes": target.stat().st_size,
                },
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to read file '{relative_path}': {exc}",
            }

    def get_metadata(self, relative_path: str) -> Dict[str, Any]:
        """Retrieve file or directory metadata.

        Args:
            relative_path: Path inside the sandbox.

        Returns:
            Dict with 'success' (bool), 'data' (dict/None), and 'error' (str/None).
        """
        try:
            target = self.resolve_safe_path(relative_path)
        except PermissionError as perm_err:
            return {"success": False, "data": None, "error": str(perm_err)}
        except Exception as exc:
            return {"success": False, "data": None, "error": f"Invalid path: {exc}"}

        if not target.exists():
            return {
                "success": False,
                "data": None,
                "error": f"File or directory does not exist: '{relative_path}'",
            }

        try:
            stat_info = target.stat()
            rel_path = target.relative_to(self.base_dir).as_posix()
            return {
                "success": True,
                "data": {
                    "path": rel_path,
                    "is_file": target.is_file(),
                    "is_dir": target.is_dir(),
                    "size_bytes": stat_info.st_size,
                    "modified_time": stat_info.st_mtime,
                    "created_time": stat_info.st_ctime,
                },
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to retrieve metadata: {exc}",
            }

    def write_file(self, relative_path: str, content: str) -> Dict[str, Any]:
        """Safely write text content to a file inside the sandbox.

        Args:
            relative_path: File path inside sandbox.
            content: Text content to write.

        Returns:
            Dict with 'success' (bool), 'data' (dict/None), and 'error' (str/None).
        """
        try:
            target = self.resolve_safe_path(relative_path)
        except PermissionError as perm_err:
            return {"success": False, "data": None, "error": str(perm_err)}
        except Exception as exc:
            return {"success": False, "data": None, "error": f"Invalid path: {exc}"}

        if target.is_dir():
            return {
                "success": False,
                "data": None,
                "error": f"Target path '{relative_path}' is an existing directory.",
            }

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
            rel_path = target.relative_to(self.base_dir).as_posix()
            return {
                "success": True,
                "data": {
                    "path": rel_path,
                    "bytes_written": len(content.encode("utf-8")),
                },
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to write file '{relative_path}': {exc}",
            }


# ==============================================================================
# 2. Files Tool Interface
# ==============================================================================

class FileTool(BaseTool):
    """AURA executable tool for safe sandboxed filesystem operations."""

    name: str = "files"
    description: str = (
        "Safe sandboxed file operations. Restricts access strictly to the safe data directory. "
        "Supports actions: 'list_files', 'read_file', 'get_metadata', 'write_file'."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_files", "read_file", "get_metadata", "write_file"],
                "description": "File action to perform.",
            },
            "path": {
                "type": "string",
                "description": "Relative path inside the safe sandbox (e.g., 'notes.txt', 'data/').",
            },
            "content": {
                "type": "string",
                "description": "Text content to write (required only for write_file).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, manager: Optional[SafeFileManager] = None) -> None:
        self.manager = manager or SafeFileManager()

    def execute(
        self,
        action: str = "list_files",
        path: str = "",
        content: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a safe file operation."""
        clean_action = (action or "").strip().lower()

        if clean_action in ("list_files", "list", "dir", "ls"):
            return self.manager.list_files(subpath=path)
        elif clean_action in ("read_file", "read", "cat"):
            return self.manager.read_file(relative_path=path)
        elif clean_action in ("get_metadata", "metadata", "stat", "info"):
            return self.manager.get_metadata(relative_path=path)
        elif clean_action in ("write_file", "write", "save"):
            return self.manager.write_file(relative_path=path, content=content or "")
        else:
            return {
                "success": False,
                "data": None,
                "error": (
                    f"Unknown file action '{action}'. "
                    "Allowed actions: 'list_files', 'read_file', 'get_metadata', 'write_file'."
                ),
            }

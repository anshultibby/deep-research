"""Chat history logger for debugging and auditing."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class ChatLogger:
    """Simple JSON-based chat history logger."""
    
    def __init__(self, log_dir: str = "chat_logs"):
        """Initialize chat logger.
        
        Args:
            log_dir: Directory to store chat logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_session_file = None
        self.session_start_time = None
    
    def start_session(self) -> str:
        """Start a new chat session and create log file.
        
        Returns:
            Session ID
        """
        self.session_start_time = datetime.now()
        session_id = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        self.current_session_file = self.log_dir / f"chat_{session_id}.json"
        
        # Initialize log file with metadata
        self._write_log({
            "session_id": session_id,
            "started_at": self.session_start_time.isoformat(),
            "messages": [],
            "total_tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0
            }
        })
        
        return session_id
    
    def log_message(self, message: Dict[str, Any]) -> None:
        """Log a single message to current session.
        
        Args:
            message: Message dict with role, content, metadata, etc.
        """
        if not self.current_session_file:
            self.start_session()
        
        # Read current log
        with open(self.current_session_file, 'r') as f:
            log_data = json.load(f)
        
        # Add timestamp to message
        message_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            **message
        }
        
        # Append message
        log_data["messages"].append(message_with_timestamp)
        
        # Update token totals if message has token metadata
        if "metadata" in message and "tokens" in message["metadata"]:
            tokens = message["metadata"]["tokens"]
            log_data["total_tokens"]["prompt_tokens"] += tokens.get("prompt_tokens", 0)
            log_data["total_tokens"]["completion_tokens"] += tokens.get("completion_tokens", 0)
            log_data["total_tokens"]["reasoning_tokens"] += tokens.get("reasoning_tokens", 0)
            log_data["total_tokens"]["total_tokens"] += tokens.get("total_tokens", 0)
        
        # Write back
        self._write_log(log_data)
    
    def log_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Log multiple messages at once.
        
        Args:
            messages: List of message dicts
        """
        for message in messages:
            self.log_message(message)
    
    def _write_log(self, data: Dict[str, Any]) -> None:
        """Write log data to file.
        
        Args:
            data: Log data to write
        """
        with open(self.current_session_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_session_file(self) -> str:
        """Get current session log file path.
        
        Returns:
            Path to current session log file
        """
        return str(self.current_session_file) if self.current_session_file else None


# Global logger instance
_logger = None


def get_chat_logger() -> ChatLogger:
    """Get or create global chat logger instance."""
    global _logger
    if _logger is None:
        _logger = ChatLogger()
    return _logger


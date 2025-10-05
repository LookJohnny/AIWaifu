from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class ConversationTurn:
    role: str  # 'user' or 'assistant'
    text: str
    emote: Optional[str] = None
    ts: float = field(default_factory=time.time)


class ConversationMemory:
    """
    Lightweight in-memory conversation window + naive summarization.
    - Keeps the last `max_turns` turns for prompt context.
    - Every `summary_interval` pairs (user+assistant), builds a short summary.
    """

    def __init__(self, max_turns: int = 8, summary_interval: int = 4):
        self.max_turns = max_turns
        self.summary_interval = summary_interval
        self.turns: List[ConversationTurn] = []
        self.summary: str = ""
        self._pairs_since_summary = 0

    def add_user(self, text: str):
        if not text:
            return
        self._append(ConversationTurn(role="user", text=text))

    def add_assistant(self, text: str, emote: Optional[str] = None):
        if not text:
            return
        self._append(ConversationTurn(role="assistant", text=text, emote=emote))
        # count after assistant reply (complete one pair)
        self._pairs_since_summary += 1
        if self._pairs_since_summary >= self.summary_interval:
            self._pairs_since_summary = 0
            self._update_summary()

    def _append(self, turn: ConversationTurn):
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def _update_summary(self):
        # Build a very compact summary from the last window
        texts = []
        for t in self.turns:
            prefix = "U:" if t.role == "user" else "A:"
            piece = f"{prefix}{t.text.strip()}"
            if len(piece) > 120:
                piece = piece[:117] + "..."
            texts.append(piece)
        joined = " | ".join(texts)
        # Cap summary length
        self.summary = joined[:500]

    def get_context_block(self) -> str:
        lines: List[str] = []
        if self.summary:
            lines.append("[会话摘要]" + self.summary)
        if self.turns:
            lines.append("[最近对话]")
            for t in self.turns:
                if t.role == "user":
                    lines.append(f"用户: {t.text}")
                else:
                    lines.append(f"Ani: {t.text}")
        return "\n".join(lines)


"""
回放测试模式 — 不烧 token 验证框架逻辑
保存每轮 LLM 响应到 JSON → 修改框架代码 → 回放验证
"""

import json
from pathlib import Path
from typing import Any, Optional


class ReplayRecorder:
    """录制模式：拦截 LLM 调用，保存响应到文件."""

    def __init__(self, replay_dir: Path):
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.recordings: list[dict] = []
        self.enabled = True

    def record(self, system_prompt: str, user_prompt: str, response: str) -> None:
        if not self.enabled:
            return
        self.recordings.append({
            "system": system_prompt[:500],
            "user": user_prompt[:500],
            "response": response,
        })

    def save(self) -> Path:
        path = self.replay_dir / "responses.json"
        path.write_text(
            json.dumps(self.recordings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


class ReplayPlayer:
    """回放模式：从文件读取预录的 LLM 响应."""

    def __init__(self, replay_file: str):
        self.replay_file = Path(replay_file)
        self.responses: list[str] = []
        self.index = 0
        self.enabled = True

    def load(self) -> bool:
        if not self.replay_file.exists():
            return False
        data = json.loads(self.replay_file.read_text(encoding="utf-8"))
        self.responses = [r["response"] for r in data]
        self.index = 0
        return True

    def next_response(self) -> Optional[str]:
        if not self.enabled or self.index >= len(self.responses):
            return None
        resp = self.responses[self.index]
        self.index += 1
        return resp

    def remaining(self) -> int:
        return len(self.responses) - self.index


class ReplayProtocol:
    """
    回放协议包裹器 — 可在录制和回放模式间切换.
    用法:
        # 录制模式
        recorder = ReplayRecorder(Path("output/v11/round_1"))
        protocol = ReplayProtocol(protocol, recorder=recorder)

        # 回放模式
        player = ReplayPlayer("output/v11/round_1/responses.json")
        protocol = ReplayProtocol(protocol, player=player)
    """

    def __init__(
        self,
        real_protocol: Any,
        recorder: Optional[ReplayRecorder] = None,
        player: Optional[ReplayPlayer] = None,
    ):
        self.protocol = real_protocol
        self.recorder = recorder
        self.player = player

    @property
    def mode(self) -> str:
        if self.player and self.player.enabled:
            return "replay"
        if self.recorder and self.recorder.enabled:
            return "record"
        return "live"

    def call_llm(self, system_prompt: str, user_prompt: str, cost=None) -> str:
        # 回放模式
        if self.player and self.player.enabled:
            resp = self.player.next_response()
            if resp is None:
                raise RuntimeError(
                    f"Replay exhausted at index {self.player.index}. "
                    f"Recorded: {len(self.player.responses)} responses."
                )
            return resp

        # 真实调用
        resp = self.protocol._call_llm(system_prompt, user_prompt, cost)

        # 录制
        if self.recorder and self.recorder.enabled:
            self.recorder.record(system_prompt, user_prompt, resp)

        return resp

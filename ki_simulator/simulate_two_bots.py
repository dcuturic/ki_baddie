import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests


@dataclass
class BotConfig:
    name: str
    base_url: str
    endpoint: str
    role: str
    user_id: str


@dataclass
class ConversationConfig:
    topic: str
    strategy: str
    turns: int
    starter: str
    history_window: int
    delay_seconds: float
    timeout_seconds: int
    output_path: Optional[str]


STRATEGY_HINTS: Dict[str, str] = {
    "debate": "Vertrete deine Position klar, bringe pro Antwort genau ein neues Argument.",
    "interview": "Stelle häufig Fragen und gehe präzise auf die letzte Antwort ein.",
    "story": "Erweitere gemeinsam eine zusammenhängende Szene mit klarer Kontinuität.",
    "brainstorm": "Sammle konkrete Ideen, strukturiert und lösungsorientiert.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simuliert ein Gespräch zwischen zwei KI-Bots über HTTP-Requests."
    )

    parser.add_argument("--config", type=str, help="Pfad zu einer JSON-Config")

    parser.add_argument("--bot-a-url", type=str, default="http://localhost:5001")
    parser.add_argument("--bot-b-url", type=str, default="http://localhost:5001")
    parser.add_argument("--bot-a-name", type=str, default="BotA")
    parser.add_argument("--bot-b-name", type=str, default="BotB")
    parser.add_argument("--bot-a-role", type=str, default="ruhiger analytischer Gesprächspartner")
    parser.add_argument("--bot-b-role", type=str, default="kreativer impulsiver Gesprächspartner")
    parser.add_argument("--bot-a-user", type=str, default="sim_bot_a")
    parser.add_argument("--bot-b-user", type=str, default="sim_bot_b")
    parser.add_argument("--endpoint", type=str, default="/chat")

    parser.add_argument("--topic", type=str, default="Wie bauen wir einen guten Stream-Abend auf?")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=list(STRATEGY_HINTS.keys()),
        default="debate",
    )
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--starter", type=str, choices=["a", "b"], default="a")
    parser.add_argument("--history-window", type=int, default=6)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", type=str, default="")

    return parser.parse_args()


def load_json_config(path: Optional[str]) -> Dict:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config nicht gefunden: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def endpoint_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def read_config(args: argparse.Namespace, file_cfg: Dict) -> (BotConfig, BotConfig, ConversationConfig):
    bot_a_cfg = file_cfg.get("bot_a", {})
    bot_b_cfg = file_cfg.get("bot_b", {})
    conv_cfg = file_cfg.get("conversation", {})

    endpoint = conv_cfg.get("endpoint", args.endpoint)

    bot_a = BotConfig(
        name=bot_a_cfg.get("name", args.bot_a_name),
        base_url=bot_a_cfg.get("url", args.bot_a_url),
        endpoint=bot_a_cfg.get("endpoint", endpoint),
        role=bot_a_cfg.get("role", args.bot_a_role),
        user_id=bot_a_cfg.get("user_id", args.bot_a_user),
    )

    bot_b = BotConfig(
        name=bot_b_cfg.get("name", args.bot_b_name),
        base_url=bot_b_cfg.get("url", args.bot_b_url),
        endpoint=bot_b_cfg.get("endpoint", endpoint),
        role=bot_b_cfg.get("role", args.bot_b_role),
        user_id=bot_b_cfg.get("user_id", args.bot_b_user),
    )

    conversation = ConversationConfig(
        topic=conv_cfg.get("topic", args.topic),
        strategy=conv_cfg.get("strategy", args.strategy),
        turns=int(conv_cfg.get("turns", args.turns)),
        starter=conv_cfg.get("starter", args.starter),
        history_window=int(conv_cfg.get("history_window", args.history_window)),
        delay_seconds=float(conv_cfg.get("delay_seconds", args.delay)),
        timeout_seconds=int(conv_cfg.get("timeout_seconds", args.timeout)),
        output_path=conv_cfg.get("output_path") or args.output or None,
    )

    if conversation.turns < 1:
        raise ValueError("turns muss >= 1 sein")
    if conversation.history_window < 1:
        raise ValueError("history-window muss >= 1 sein")

    return bot_a, bot_b, conversation


def build_prompt(
    speaker: BotConfig,
    listener: BotConfig,
    cfg: ConversationConfig,
    history: List[Dict[str, str]],
    last_message: str,
) -> str:
    recent = history[-cfg.history_window :]
    history_text = "\n".join(f"- {h['speaker']}: {h['text']}" for h in recent) if recent else "- (leer)"
    strategy_hint = STRATEGY_HINTS.get(cfg.strategy, STRATEGY_HINTS["debate"])

    return (
        "[SIMULATION-KONTEXT]\n"
        "Dies ist eine kontrollierte Bot-zu-Bot-Unterhaltung.\n"
        f"Deine Rolle: {speaker.role}\n"
        f"Gesprächspartner: {listener.name}\n"
        f"Thema: {cfg.topic}\n"
        f"Strategie: {cfg.strategy} - {strategy_hint}\n"
        "Regeln: Antworte in 1-3 Sätzen, direkt, konkret, ohne Meta-Erklärung.\n"
        "Beziehe dich auf die letzte Aussage und führe den Dialog sinnvoll fort.\n\n"
        f"Letzte Nachricht von {listener.name}: {last_message}\n\n"
        f"Letzte {len(recent)} Beiträge:\n{history_text}\n\n"
        "Jetzt antworte als nächster Gesprächsbeitrag."
    )


def send_chat(bot: BotConfig, username: str, text: str, timeout_seconds: int) -> str:
    url = endpoint_url(bot.base_url, bot.endpoint)
    payload = {"message": f"{username}:{text}"}

    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()

    data = response.json()
    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError(f"Ungültige Antwort von {bot.name} über {url}: {data}")

    return reply.strip()


def run_simulation(bot_a: BotConfig, bot_b: BotConfig, cfg: ConversationConfig) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    current = cfg.starter
    last_message = f"Startet ein Gespräch zum Thema: {cfg.topic}"

    print("=" * 70)
    print(f"Simulation gestartet: {bot_a.name} ↔ {bot_b.name}")
    print(f"Strategie: {cfg.strategy} | Turns: {cfg.turns}")
    print(f"Thema: {cfg.topic}")
    print("=" * 70)

    for turn in range(1, cfg.turns + 1):
        if current == "a":
            speaker = bot_a
            listener = bot_b
            username = bot_b.user_id
            current = "b"
        else:
            speaker = bot_b
            listener = bot_a
            username = bot_a.user_id
            current = "a"

        prompt = build_prompt(
            speaker=speaker,
            listener=listener,
            cfg=cfg,
            history=history,
            last_message=last_message,
        )

        reply = send_chat(
            bot=speaker,
            username=username,
            text=prompt,
            timeout_seconds=cfg.timeout_seconds,
        )

        entry = {
            "turn": str(turn),
            "speaker": speaker.name,
            "listener": listener.name,
            "text": reply,
        }
        history.append(entry)
        last_message = reply

        print(f"[{turn:02d}] {speaker.name}: {reply}")

        if cfg.delay_seconds > 0 and turn < cfg.turns:
            time.sleep(cfg.delay_seconds)

    return history


def write_output(path: str, bot_a: BotConfig, bot_b: BotConfig, cfg: ConversationConfig, history: List[Dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bot_a": bot_a.__dict__,
        "bot_b": bot_b.__dict__,
        "conversation": cfg.__dict__,
        "transcript": history,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    try:
        args = parse_args()
        file_cfg = load_json_config(args.config)
        bot_a, bot_b, conversation_cfg = read_config(args, file_cfg)

        transcript = run_simulation(bot_a, bot_b, conversation_cfg)

        if conversation_cfg.output_path:
            write_output(conversation_cfg.output_path, bot_a, bot_b, conversation_cfg, transcript)
            print(f"\nTranscript gespeichert: {conversation_cfg.output_path}")

        print("\nSimulation abgeschlossen.")
        return 0
    except requests.RequestException as exc:
        print(f"HTTP-Fehler: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

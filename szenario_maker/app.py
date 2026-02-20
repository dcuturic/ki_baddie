#!/usr/bin/env python3
"""
Szenario Maker - KI Multi-Chat Experiment Builder
Erstellt und steuert komplexe KI-Chat-Szenarien mit mehreren Teilnehmern,
Kommunikationsregeln, verbotenen Wörtern und Gewinnbedingungen.
"""

import json
import os
import sys
import io
import time
import uuid
import threading
import queue
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field

# ===== Windows UTF-8 fix =====
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import requests as http_requests
from flask import Flask, render_template, request, jsonify, Response

# ============================================================================
# Configuration
# ============================================================================

APP_ROOT = Path(__file__).parent
CONFIG_PATH = APP_ROOT / "config.json"
SCENARIOS_DIR = APP_ROOT / "scenarios"
TRANSCRIPTS_DIR = APP_ROOT / "transcripts"

def load_config() -> Dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

CONFIG = load_config()

SCENARIOS_DIR = APP_ROOT / CONFIG.get("scenarios_dir", "scenarios")
TRANSCRIPTS_DIR = APP_ROOT / CONFIG.get("transcripts_dir", "transcripts")
SCENARIOS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

DEFAULT_CHAT_URL = CONFIG.get("default_chat_url", "http://localhost:5001")
DEFAULT_CHAT_ENDPOINT = CONFIG.get("default_chat_endpoint", "/chat-free")
DEFAULT_TIMEOUT = CONFIG.get("default_timeout", 60)
DEFAULT_TTS_URL = CONFIG.get("tts_url", "http://localhost:5057")
DEFAULT_VROID_EMOTION_URL = CONFIG.get("vroid_emotion_url", "http://localhost:5004")
MANAGER_URL = CONFIG.get("manager_url", "http://localhost:8000")

# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['JSON_SORT_KEYS'] = False
app.json.ensure_ascii = False

# ============================================================================
# Active Experiments (running scenarios)
# ============================================================================

@dataclass
class ExperimentMessage:
    timestamp: str
    sender: str
    receiver: str
    channel_id: str
    text: str
    emotion: str = "neutral"
    forbidden_word_detected: Optional[str] = None

@dataclass
class ActiveExperiment:
    experiment_id: str
    scenario_name: str
    scenario: Dict
    status: str = "running"  # running, paused, stopped, finished
    messages: List[Dict] = field(default_factory=list)
    event_queue: queue.Queue = field(default_factory=queue.Queue)
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    win_result: Optional[str] = None
    started_at: Optional[str] = None
    round_count: int = 0

    def __post_init__(self):
        self.pause_event.set()  # not paused by default

ACTIVE_EXPERIMENTS: Dict[str, ActiveExperiment] = {}

# ============================================================================
# Scenario CRUD
# ============================================================================

def list_scenarios() -> List[Dict]:
    scenarios = []
    for f in sorted(SCENARIOS_DIR.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                scenarios.append({
                    "filename": f.stem,
                    "name": data.get("name", f.stem),
                    "description": data.get("description", ""),
                    "participant_count": len(data.get("participants", [])),
                    "channel_count": len(data.get("channels", []))
                })
        except Exception:
            pass
    return scenarios

def load_scenario(filename: str) -> Optional[Dict]:
    path = SCENARIOS_DIR / f"{filename}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_scenario(filename: str, data: Dict) -> None:
    path = SCENARIOS_DIR / f"{filename}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def delete_scenario(filename: str) -> bool:
    path = SCENARIOS_DIR / f"{filename}.json"
    if path.exists():
        path.unlink()
        return True
    return False

# ============================================================================
# Chat Communication with ki_chat
# ============================================================================

def send_chat_message(
    chat_url: str,
    endpoint: str,
    message: str,
    system_prompt: str = "",
    timeout: int = DEFAULT_TIMEOUT,
    username: str = "",
    use_memory: bool = False,
    history: list = None
) -> Dict:
    """Send message to ki_chat and return response.
    
    If use_memory=True (Gefühlspyramide aktiv), uses /chat endpoint
    with 'username: message' format so that memory banks, emotion tree
    and topic memory are active. Otherwise uses /chat-free.
    """
    url = f"{chat_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    if use_memory and endpoint.rstrip('/').endswith('/chat'):
        # /chat expects format: "username: message_text"
        payload = {
            "message": f"{username}: {message}" if username else message
        }
    else:
        # /chat-free expects message + optional system prompt + optional history
        payload = {
            "message": message,
            "system": system_prompt
        }
        if history:
            payload["history"] = history
    
    try:
        resp = http_requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return {
            "reply": data.get("reply", ""),
            "emotion": data.get("emotion", "neutral"),
            "success": True
        }
    except Exception as e:
        return {
            "reply": f"[FEHLER: {str(e)}]",
            "emotion": "neutral",
            "success": False
        }


def send_tts(text: str, emotion: str = "neutral", tts_url: str = "", voice: str = "") -> bool:
    """Send text to TTS service for voice output"""
    url = f"{(tts_url or DEFAULT_TTS_URL).rstrip('/')}/tts"
    payload = {
        "text": {"value": text, "emotion": emotion},
        "play_audio": True
    }
    if voice:
        payload["voice"] = voice
    try:
        resp = http_requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[TTS] Fehler: {e}")
        return False


def send_vroid_emotion(emotion: str, intensity: float = 1.0, vroid_url: str = "") -> bool:
    """Send emotion to VroidEmotion for avatar expression"""
    url = f"{(vroid_url or DEFAULT_VROID_EMOTION_URL).rstrip('/')}/emotion"
    payload = {
        "emotion": emotion,
        "intensity": min(1.0, max(0.0, intensity)),
        "fade": 0.35
    }
    try:
        resp = http_requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[VROID] Fehler: {e}")
        return False


def check_service_health(url: str, timeout: int = 5) -> Dict:
    """Check if a service is reachable and responding"""
    try:
        # Try /health first
        resp = http_requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        if resp.ok:
            return {"ok": True, "status": "healthy"}
    except Exception:
        pass
    
    try:
        # Fallback: try /chat-free with empty
        resp = http_requests.post(
            f"{url.rstrip('/')}/chat-free",
            json={"message": ""},
            timeout=timeout
        )
        if resp.ok:
            return {"ok": True, "status": "reachable"}
    except Exception as e:
        return {"ok": False, "status": f"nicht erreichbar: {e}"}
    
    return {"ok": False, "status": "keine Antwort"}


def switch_character(chat_url: str, character_name: str) -> bool:
    """Switch ki_chat to a different character before experiment starts"""
    url = f"{chat_url.rstrip('/')}/character/switch"
    try:
        resp = http_requests.post(url, json={"character": character_name}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return not data.get("error")
    except Exception as e:
        print(f"[CHARACTER SWITCH] Fehler: {e}")
        return False

# ============================================================================
# Experiment Engine
# ============================================================================

def build_participant_system_prompt(participant: Dict, scenario: Dict, target_name: str = "") -> str:
    """Build system prompt for a participant including scenario rules"""
    base_prompt = participant.get("system_prompt", "")
    forbidden = scenario.get("forbidden_words", {}).get(participant["id"], [])

    prompt_parts = []

    # Identity and scenario context
    scenario_name = scenario.get("name", "Experiment")
    speaker_name = participant.get("name", participant["id"])
    identity_block = f"[SZENARIO: {scenario_name}]\nDu bist {speaker_name}."
    if target_name:
        identity_block += f"\nDu sprichst mit {target_name}."
    prompt_parts.append(identity_block)

    if base_prompt:
        prompt_parts.append(base_prompt)

    if forbidden:
        words_str = ", ".join(f'"{w}"' for w in forbidden)
        prompt_parts.append(
            f"\n\n[EXPERIMENT-REGEL]: Du darfst folgende Wörter NICHT benutzen: {words_str}. "
            f"Vermeide diese Wörter unbedingt in deinen Antworten."
        )

    prompt_parts.append(
        "\n\nANTWORT-FORMAT: Antworte in 1-4 Sätzen. Jede Antwort endet exakt mit: \"|| <emotion>\"."
        "\nErlaubte Emotionen: surprise, angry, sorrow, fun, neutral, joy"
        "\n\nWICHTIG: Wiederhole NIEMALS Sätze oder Inhalte aus vorherigen Nachrichten wörtlich. "
        "Bringe das Gespräch immer weiter voran. Keine Zusammenfassungen des bisherigen Gesprächs."
    )

    return "\n".join(prompt_parts)


def build_context_message(
    speaker: Dict,
    target: Dict,
    channel: Dict,
    conversation_history: List[Dict],
    scenario: Dict,
    history_window: int = 8
) -> str:
    """Build the message to send to the KI including context.
    Returns only the current prompt — history is sent separately as alternating messages."""
    recent = conversation_history[-history_window:] if conversation_history else []

    initial_message = channel.get("initial_message", "")
    if initial_message and not recent:
        # First message with a task
        context = f"Aufgabe: {initial_message}\nBeginne jetzt das Gespräch."
    elif recent:
        last_msg = recent[-1]
        context = f"{last_msg['sender']}: {last_msg['text']}"
    else:
        context = f"Beginne das Gespräch."

    return context


def build_chat_history(
    speaker_id: str,
    conversation_history: List[Dict],
    history_window: int = 8
) -> List[Dict]:
    """Build alternating user/assistant message history from conversation.
    Messages FROM this speaker → assistant role (what the LLM said before)
    Messages FROM others → user role (what the other KI said)
    """
    recent = conversation_history[-(history_window):] if conversation_history else []
    # Don't include the very last message — that's sent as the current prompt
    if len(recent) > 1:
        recent = recent[:-1]
    else:
        return []

    history = []
    for msg in recent:
        if msg.get("sender_id") == speaker_id:
            history.append({"role": "assistant", "content": msg["text"]})
        else:
            history.append({"role": "user", "content": f"{msg['sender']}: {msg['text']}"})

    return history


def check_forbidden_words(text: str, participant_id: str, scenario: Dict) -> Optional[str]:
    """Check if text contains any forbidden words for this participant"""
    forbidden = scenario.get("forbidden_words", {}).get(participant_id, [])
    text_lower = text.lower()
    for word in forbidden:
        if word.lower() in text_lower:
            return word
    return None


def check_win_conditions(message: Dict, experiment: ActiveExperiment) -> Optional[str]:
    """Check if any win condition is met"""
    scenario = experiment.scenario
    conditions = scenario.get("win_conditions", [])

    for cond in conditions:
        cond_type = cond.get("type", "")

        if cond_type == "forbidden_word_said":
            if (message.get("forbidden_word_detected") and
                    message.get("sender_id") == cond.get("participant")):
                return cond.get("result", "Bedingung erfüllt!")

        elif cond_type == "max_rounds":
            if experiment.round_count >= cond.get("rounds", 999):
                return cond.get("result", f"Maximale Runden ({cond.get('rounds')}) erreicht!")

    return None


def parse_reply(raw: str) -> tuple:
    """Parse reply text and emotion from 'text || emotion' format"""
    if "||" in raw:
        parts = raw.rsplit("||", 1)
        text = parts[0].strip()
        emotion = parts[1].strip().lower()
        if emotion not in ("surprise", "angry", "sorrow", "fun", "neutral", "joy"):
            emotion = "neutral"
        return text, emotion
    return raw.strip(), "neutral"


def fire_channel(
    ch: Dict,
    experiment: ActiveExperiment,
    participants: Dict,
    scenario: Dict,
    settings: Dict,
    channel_histories: Dict,
    participant_memories: Dict,
) -> Optional[Dict]:
    """Fire a single channel: send message and process response.
    Returns the message record on success, None on failure.
    Sets experiment error events on failure."""
    ch_id = ch.get("id", f"{ch['from']}_{ch['to']}")
    sender_id = ch["from"]
    receiver_id = ch["to"]
    sender = participants.get(sender_id)
    receiver = participants.get(receiver_id)

    if not sender or not receiver:
        print(f"[FIRE] ERROR: sender_id={sender_id!r} receiver_id={receiver_id!r} not found in participants={list(participants.keys())}", flush=True)
        return None

    # Build system prompt and context
    system_prompt = build_participant_system_prompt(sender, scenario, target_name=receiver.get("name", ""))

    # Include shared memories in context
    shared_context = ""
    if participant_memories.get(sender_id):
        recent_memories = participant_memories[sender_id][-5:]
        memory_text = "\n".join(
            f"[Erinnerung] {m['sender']} zu {m['receiver']}: {m['text']}"
            for m in recent_memories
        )
        shared_context = f"\n\nDinge die du mitgehört/erfahren hast:\n{memory_text}"

    full_system = system_prompt + shared_context

    # Build message (current prompt only — history sent separately)
    ch_history = channel_histories.get(ch_id, [])
    message_text = build_context_message(
        sender, receiver, ch,
        ch_history,
        scenario
    )

    # Build proper alternating user/assistant history
    chat_history = build_chat_history(sender_id, ch_history, history_window=8)

    # Send to KI
    experiment.event_queue.put({
        "type": "sending",
        "sender": sender["name"],
        "receiver": receiver["name"],
        "channel": ch_id,
        "timestamp": datetime.now().isoformat()
    })

    # Determine endpoint based on features
    use_memory = sender.get("use_emotion_pyramid", False)
    actual_endpoint = sender.get("chat_endpoint", DEFAULT_CHAT_ENDPOINT)
    if use_memory:
        actual_endpoint = "/chat"  # Full memory + emotion tree

    result = send_chat_message(
        chat_url=sender.get("chat_url", DEFAULT_CHAT_URL),
        endpoint=actual_endpoint,
        message=message_text,
        system_prompt=full_system,
        timeout=settings.get("timeout_seconds", DEFAULT_TIMEOUT),
        username=sender.get("chat_username", f"sim_{sender_id}"),
        use_memory=use_memory,
        history=chat_history
    )

    if not result["success"]:
        print(f"[FIRE] Chat-Fehler: {result['reply']} (url={sender.get('chat_url')}, endpoint={actual_endpoint})", flush=True)
        return None  # caller handles error

    reply_text, emotion = parse_reply(result["reply"])
    if not reply_text:
        reply_text = result["reply"]

    # === TTS: Text-to-Speech ===
    if sender.get("enable_tts", False):
        tts_url = settings.get("tts_url", DEFAULT_TTS_URL)
        voice = sender.get("name", "").strip().lower()
        send_tts(reply_text, emotion, tts_url, voice=voice)
        experiment.event_queue.put({
            "type": "tts",
            "sender": sender["name"],
            "text": f"🔊 {sender['name']} spricht...",
            "timestamp": datetime.now().isoformat()
        })

    # === VroidEmotion: Avatar-Emotion ===
    if sender.get("enable_vroid_emotion", False):
        vroid_url = settings.get("vroid_emotion_url", DEFAULT_VROID_EMOTION_URL)
        send_vroid_emotion(emotion, 1.0, vroid_url)
        experiment.event_queue.put({
            "type": "emotion_set",
            "sender": sender["name"],
            "emotion": emotion,
            "text": f"😊 {sender['name']}: Emotion → {emotion}",
            "timestamp": datetime.now().isoformat()
        })

    # Check for forbidden words
    forbidden_hit = check_forbidden_words(reply_text, sender_id, scenario)

    # Create message record
    msg = {
        "turn": experiment.round_count + 1,
        "timestamp": datetime.now().isoformat(),
        "sender": sender["name"],
        "sender_id": sender_id,
        "receiver": receiver["name"],
        "receiver_id": receiver_id,
        "channel": ch_id,
        "text": reply_text,
        "emotion": emotion,
        "forbidden_word_detected": forbidden_hit
    }

    # Store in channel history
    if ch_id not in channel_histories:
        channel_histories[ch_id] = []
    channel_histories[ch_id].append(msg)

    # Store in experiment messages
    experiment.messages.append(msg)
    experiment.round_count += 1

    # Share memory with listeners
    for listener_id in ch.get("listeners", []):
        if listener_id in participant_memories:
            participant_memories[listener_id].append(msg)

    # Share memory with memory_shared_with
    for target_id in ch.get("memory_shared_with", []):
        if target_id in participant_memories:
            participant_memories[target_id].append(msg)

    # Also store in receiver memory (they heard it)
    if receiver_id in participant_memories:
        participant_memories[receiver_id].append(msg)

    # Emit event
    event_data = {
        "type": "message",
        "turn": msg["turn"],
        "sender": msg["sender"],
        "receiver": msg["receiver"],
        "channel": ch_id,
        "text": reply_text,
        "emotion": emotion,
        "timestamp": msg["timestamp"]
    }

    if forbidden_hit:
        event_data["forbidden_word"] = forbidden_hit
        event_data["type"] = "forbidden_word"
        experiment.event_queue.put({
            "type": "alert",
            "text": f"⚠️ {sender['name']} hat das verbotene Wort \"{forbidden_hit}\" benutzt!",
            "timestamp": datetime.now().isoformat()
        })

    experiment.event_queue.put(event_data)

    return msg


def run_experiment(experiment: ActiveExperiment):
    """Main experiment loop - runs in a separate thread"""
    scenario = experiment.scenario
    participants = {p["id"]: p for p in scenario.get("participants", [])}
    channels = scenario.get("channels", [])
    settings = scenario.get("settings", {})
    max_rounds = settings.get("max_rounds", 100)

    # Per-channel conversation history
    channel_histories: Dict[str, List[Dict]] = {}
    # Per-participant shared memory (what they've heard from other channels)
    participant_memories: Dict[str, List[Dict]] = {}

    for p_id in participants:
        participant_memories[p_id] = []

    # --- Pre-flight health checks ---
    checked_urls = set()
    for p_id, p in participants.items():
        chat_url = p.get("chat_url", DEFAULT_CHAT_URL)
        if chat_url not in checked_urls:
            checked_urls.add(chat_url)
            health = check_service_health(chat_url)
            if not health["ok"]:
                error_msg = f"❌ Service {chat_url} nicht erreichbar: {health['status']}"
                print(f"[INIT] {error_msg}")
                experiment.event_queue.put({
                    "type": "error",
                    "text": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                experiment.status = "stopped"
                experiment.event_queue.put({
                    "type": "system",
                    "text": "⏹️ Experiment abgebrochen - Service nicht erreichbar.",
                    "timestamp": datetime.now().isoformat()
                })
                return
            else:
                experiment.event_queue.put({
                    "type": "system",
                    "text": f"✅ {chat_url} erreichbar ({health['status']})",
                    "timestamp": datetime.now().isoformat()
                })

    # Switch characters for participants that have a character_name set
    for p_id, p in participants.items():
        if p.get("character_name"):
            chat_url = p.get("chat_url", DEFAULT_CHAT_URL)
            char_name = p["character_name"]
            ok = switch_character(chat_url, char_name)
            if ok:
                print(f"[INIT] Character '{char_name}' auf {chat_url} aktiviert")
                experiment.event_queue.put({
                    "type": "system",
                    "text": f"🎭 {p.get('name', p_id)}: Character '{char_name}' aktiviert",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                print(f"[INIT] WARNUNG: Character-Switch '{char_name}' auf {chat_url} fehlgeschlagen")
                experiment.event_queue.put({
                    "type": "error",
                    "text": f"⚠️ {p.get('name', p_id)}: Character-Switch '{char_name}' fehlgeschlagen",
                    "timestamp": datetime.now().isoformat()
                })

    for ch in channels:
        ch_id = ch.get("id", f"{ch['from']}_{ch['to']}")
        channel_histories[ch_id] = []

    experiment.started_at = datetime.now().isoformat()
    experiment.event_queue.put({
        "type": "system",
        "text": f"🚀 Experiment '{scenario.get('name', '')}' gestartet!",
        "timestamp": datetime.now().isoformat()
    })

    # Channel timing tracker
    channel_timers: Dict[str, float] = {}
    for ch in channels:
        ch_id = ch.get("id", f"{ch['from']}_{ch['to']}")
        channel_timers[ch_id] = 0  # ready immediately or with offset
        if ch.get("start_delay_seconds", 0) > 0:
            channel_timers[ch_id] = time.time() + ch["start_delay_seconds"]
        else:
            channel_timers[ch_id] = time.time()

    # --- Direct Dialog Mode ---
    # Detect bidirectional channel pairs and run them as ping-pong
    direct_dialog = settings.get("direct_dialog", False)
    dialog_pairs: Dict[str, Dict] = {}  # "a→b": {forward_ch, reverse_ch}

    if direct_dialog:
        # Build a lookup: (from, to) -> channel
        ch_lookup = {}
        for ch in channels:
            key = (ch["from"], ch["to"])
            ch_lookup[key] = ch

        paired_channels = set()
        for ch in channels:
            ch_id = ch.get("id", f"{ch['from']}_{ch['to']}")
            reverse_key = (ch["to"], ch["from"])
            if reverse_key in ch_lookup and ch_id not in paired_channels:
                reverse_ch = ch_lookup[reverse_key]
                reverse_id = reverse_ch.get("id", f"{reverse_ch['from']}_{reverse_ch['to']}")
                pair_key = f"{ch['from']}↔{ch['to']}"
                dialog_pairs[pair_key] = {
                    "forward": ch,
                    "reverse": reverse_ch,
                    "forward_id": ch_id,
                    "reverse_id": reverse_id,
                }
                paired_channels.add(ch_id)
                paired_channels.add(reverse_id)

        if dialog_pairs:
            experiment.event_queue.put({
                "type": "system",
                "text": f"💬 Direkt-Dialog-Modus: {len(dialog_pairs)} Gesprächspaar(e) erkannt",
                "timestamp": datetime.now().isoformat()
            })

    start_time = time.time()
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    while not experiment.stop_event.is_set() and experiment.round_count < max_rounds:
        # Pause support
        experiment.pause_event.wait()

        if experiment.stop_event.is_set():
            break

        # Zu viele Fehler hintereinander → Experiment stoppen
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            experiment.status = "stopped"
            experiment.event_queue.put({
                "type": "auto_stopped",
                "text": f"Experiment auto-gestoppt: {MAX_CONSECUTIVE_ERRORS} Fehler hintereinander. Prüfe ob Ollama/ki_chat erreichbar sind.",
                "timestamp": datetime.now().isoformat()
            })
            save_transcript(experiment)
            return

        any_channel_fired = False

        for ch in channels:
            if experiment.stop_event.is_set():
                break
            if experiment.round_count >= max_rounds:
                break

            ch_id = ch.get("id", f"{ch['from']}_{ch['to']}")
            interval = ch.get("interval_seconds", 10)
            now = time.time()

            # Check if channel is ready to fire
            if now < channel_timers.get(ch_id, 0):
                continue

            # --- Direct Dialog: Ping-Pong-Modus ---
            if direct_dialog:
                # Check if this channel is part of a dialog pair
                is_pair_channel = False
                for pair_key, pair in dialog_pairs.items():
                    if ch_id == pair["forward_id"] or ch_id == pair["reverse_id"]:
                        is_pair_channel = True
                        # Only fire from the forward channel's timer
                        if ch_id != pair["forward_id"]:
                            break  # skip reverse channel, it's fired by ping-pong

                        # Ping-Pong loop: alternate between forward and reverse
                        current_pair = [pair["forward"], pair["reverse"]]
                        pair_idx = 0  # start with forward

                        while not experiment.stop_event.is_set() and experiment.round_count < max_rounds:
                            experiment.pause_event.wait()
                            if experiment.stop_event.is_set():
                                break
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                break

                            active_ch = current_pair[pair_idx % 2]
                            active_ch_id = active_ch.get("id", f"{active_ch['from']}_{active_ch['to']}")

                            # Cross-share history: both channels share the same conversation
                            # Merge histories so both sides see the full dialog
                            fwd_id = pair["forward_id"]
                            rev_id = pair["reverse_id"]
                            merged_history = sorted(
                                channel_histories.get(fwd_id, []) + channel_histories.get(rev_id, []),
                                key=lambda m: m["turn"]
                            )
                            channel_histories[fwd_id] = merged_history
                            channel_histories[rev_id] = merged_history

                            msg = fire_channel(
                                active_ch, experiment, participants, scenario,
                                settings, channel_histories, participant_memories
                            )

                            if msg is None:
                                consecutive_errors += 1
                                sender = participants.get(active_ch["from"], {})
                                err_name = sender.get('name', active_ch.get('from', '?'))
                                print(f"[PINGPONG] Fehler bei {err_name} ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})", flush=True)
                                experiment.event_queue.put({
                                    "type": "error",
                                    "text": f"⚠️ Fehler bei {sender.get('name', '?')} ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})",
                                    "timestamp": datetime.now().isoformat()
                                })
                                time.sleep(5)
                                continue

                            consecutive_errors = 0

                            # Check win conditions
                            win = check_win_conditions(msg, experiment)
                            if win:
                                experiment.win_result = win
                                experiment.event_queue.put({
                                    "type": "win",
                                    "text": f"🏆 {win}",
                                    "timestamp": datetime.now().isoformat()
                                })
                                experiment.status = "finished"
                                save_transcript(experiment)
                                return

                            pair_idx += 1
                            # Small breathing delay between direct messages (1s)
                            time.sleep(1)

                        # After ping-pong ends, set timer far ahead so we don't re-enter
                        channel_timers[pair["forward_id"]] = time.time() + 999999
                        channel_timers[pair["reverse_id"]] = time.time() + 999999
                        any_channel_fired = True
                        break  # exit pair search

                if is_pair_channel:
                    continue  # skip to next channel

            # --- Normal interval-based channel ---
            msg = fire_channel(
                ch, experiment, participants, scenario,
                settings, channel_histories, participant_memories
            )

            if msg is None:
                sender = participants.get(ch["from"], {})
                consecutive_errors += 1
                experiment.event_queue.put({
                    "type": "error",
                    "text": f"⚠️ Fehler bei {sender.get('name', '?')} ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})",
                    "timestamp": datetime.now().isoformat()
                })
                # Longer delay after error
                channel_timers[ch_id] = now + interval + 10
                continue

            consecutive_errors = 0

            # Check win conditions
            win = check_win_conditions(msg, experiment)
            if win:
                experiment.win_result = win
                experiment.event_queue.put({
                    "type": "win",
                    "text": f"🏆 {win}",
                    "timestamp": datetime.now().isoformat()
                })
                experiment.status = "finished"
                save_transcript(experiment)
                return

            # Update timer for this channel
            channel_timers[ch_id] = now + interval
            any_channel_fired = True

        # Small sleep to prevent CPU spin
        if not any_channel_fired:
            time.sleep(0.5)

    # Experiment finished (max rounds or stopped)
    if experiment.stop_event.is_set():
        experiment.status = "stopped"
        experiment.event_queue.put({
            "type": "system",
            "text": "⏹️ Experiment wurde gestoppt.",
            "timestamp": datetime.now().isoformat()
        })
    else:
        experiment.status = "finished"
        experiment.event_queue.put({
            "type": "system",
            "text": f"✅ Experiment beendet nach {experiment.round_count} Runden.",
            "timestamp": datetime.now().isoformat()
        })

    save_transcript(experiment)


def save_transcript(experiment: ActiveExperiment):
    """Save experiment transcript to file"""
    transcript = {
        "experiment_id": experiment.experiment_id,
        "scenario_name": experiment.scenario_name,
        "started_at": experiment.started_at,
        "finished_at": datetime.now().isoformat(),
        "status": experiment.status,
        "total_rounds": experiment.round_count,
        "win_result": experiment.win_result,
        "messages": experiment.messages,
        "scenario": experiment.scenario
    }
    path = TRANSCRIPTS_DIR / f"{experiment.experiment_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

# ============================================================================
# Service Discovery
# ============================================================================

def discover_services() -> List[Dict]:
    """Discover running services by querying the manager and doing health checks."""
    services = []
    
    # --- Ask manager for status ---
    try:
        # Get all instances - response: {"success": true, "instances": [...]}
        inst_resp = http_requests.get(f"{MANAGER_URL}/api/instance/list", timeout=3)
        inst_data = inst_resp.json() if inst_resp.ok else {}
        instances = inst_data.get("instances", []) if isinstance(inst_data, dict) else []
        
        # Get status for each instance
        for inst in instances:
            # Instance objects have "filename" as the identifier, and optionally "name"
            inst_name = inst.get("filename") or inst.get("name", "") if isinstance(inst, dict) else str(inst)
            if not inst_name:
                continue
            try:
                status_resp = http_requests.get(
                    f"{MANAGER_URL}/api/status/all",
                    params={"instance": inst_name},
                    timeout=3
                )
                if not status_resp.ok:
                    continue
                status_data = status_resp.json()
                svc_statuses = status_data.get("services", {})
                
                for svc_id, svc_info in svc_statuses.items():
                    if svc_info.get("status") != "running":
                        continue
                    
                    # Get port from instance config
                    # Response: {"success": true, "config": {...}}
                    port = None
                    try:
                        cfg_resp = http_requests.get(
                            f"{MANAGER_URL}/api/instance/{inst_name}/config/service/{svc_id}",
                            timeout=3
                        )
                        if cfg_resp.ok:
                            cfg_data = cfg_resp.json()
                            cfg = cfg_data.get("config", cfg_data) if isinstance(cfg_data, dict) else {}
                            # Port is either at server.port or top-level port
                            port = cfg.get("server", {}).get("port") or cfg.get("port")
                    except Exception:
                        pass
                    
                    # Fallback: try reading port from the service's own config.json
                    if not port:
                        try:
                            cfg2_resp = http_requests.get(
                                f"{MANAGER_URL}/api/config/service/{svc_id}",
                                timeout=3
                            )
                            if cfg2_resp.ok:
                                cfg2_data = cfg2_resp.json()
                                cfg2 = cfg2_data.get("config", cfg2_data) if isinstance(cfg2_data, dict) else {}
                                port = cfg2.get("server", {}).get("port") or cfg2.get("port")
                        except Exception:
                            pass
                    
                    if port:
                        url = f"http://localhost:{port}"
                        services.append({
                            "service_id": svc_id,
                            "service_name": svc_info.get("service_name", svc_id),
                            "instance": inst_name,
                            "url": url,
                            "port": port,
                            "status": "online",
                            "pid": svc_info.get("pid"),
                            "uptime": svc_info.get("uptime_text", "")
                        })
            except Exception:
                continue
    except Exception as e:
        print(f"[DISCOVER] Manager nicht erreichbar: {e}")
        
        # Fallback: Direct health check on common ports
        FALLBACK_SERVICES = [
            {"service_id": "ki_chat", "service_name": "KI Chat", "port": 5001},
            {"service_id": "main_server", "service_name": "Main Server", "port": 5000},
            {"service_id": "text_to_speech", "service_name": "Text-to-Speech", "port": 5057},
            {"service_id": "vroid_emotion", "service_name": "VroidEmotion", "port": 5004},
            {"service_id": "vroid_poser", "service_name": "VroidPoser", "port": 5003},
            {"service_id": "web_avatar", "service_name": "Web Avatar", "port": 5006},
        ]
        for svc in FALLBACK_SERVICES:
            url = f"http://localhost:{svc['port']}"
            try:
                r = http_requests.get(f"{url}/health", timeout=2)
                if r.ok:
                    services.append({
                        "service_id": svc["service_id"],
                        "service_name": svc["service_name"],
                        "instance": "unknown",
                        "url": url,
                        "port": svc["port"],
                        "status": "online",
                        "pid": None,
                        "uptime": ""
                    })
            except Exception:
                pass
    
    return services


# ============================================================================
# API Routes
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html")

# ---------- Service Discovery ----------

@app.route("/api/discover-services", methods=["GET"])
def api_discover_services():
    """Discover online services (ki_chat, main_server, TTS, etc.)"""
    services = discover_services()
    return jsonify({"services": services})


@app.route("/api/characters", methods=["GET"])
def api_get_characters():
    """Fetch available characters from a ki_chat instance"""
    chat_url = request.args.get("chat_url", "").strip()
    if not chat_url:
        return jsonify({"characters": [], "error": "chat_url Parameter fehlt"}), 400
    
    url = f"{chat_url.rstrip('/')}/characters"
    try:
        resp = http_requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        characters = data.get("characters", [])
        current = data.get("current", "")
        return jsonify({"characters": characters, "current": current})
    except Exception as e:
        print(f"[CHARACTERS] Fehler beim Abrufen von {url}: {e}")
        return jsonify({"characters": [], "error": str(e)})

# ---------- Scenario CRUD ----------

@app.route("/api/scenarios", methods=["GET"])
def api_list_scenarios():
    return jsonify({"scenarios": list_scenarios()})

@app.route("/api/scenarios/<filename>", methods=["GET"])
def api_get_scenario(filename):
    data = load_scenario(filename)
    if data is None:
        return jsonify({"error": "Szenario nicht gefunden"}), 404
    return jsonify(data)

@app.route("/api/scenarios/<filename>", methods=["PUT"])
def api_save_scenario(filename):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Keine Daten"}), 400
    save_scenario(filename, data)
    return jsonify({"ok": True, "filename": filename})

@app.route("/api/scenarios", methods=["POST"])
def api_create_scenario():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Keine Daten"}), 400
    name = data.get("name", "neues_szenario")
    filename = re.sub(r'[^a-z0-9_-]', '_', name.lower().strip().replace(" ", "_"))
    if not filename:
        filename = f"szenario_{uuid.uuid4().hex[:8]}"
    save_scenario(filename, data)
    return jsonify({"ok": True, "filename": filename})

@app.route("/api/scenarios/<filename>", methods=["DELETE"])
def api_delete_scenario(filename):
    if delete_scenario(filename):
        return jsonify({"ok": True})
    return jsonify({"error": "Nicht gefunden"}), 404

# ---------- Experiment Control ----------

@app.route("/api/experiment/start", methods=["POST"])
def api_start_experiment():
    data = request.get_json(silent=True) or {}
    filename = data.get("scenario")
    if not filename:
        return jsonify({"error": "Kein Szenario angegeben"}), 400

    scenario = load_scenario(filename)
    if not scenario:
        return jsonify({"error": "Szenario nicht gefunden"}), 404

    exp_id = f"exp_{uuid.uuid4().hex[:12]}"
    experiment = ActiveExperiment(
        experiment_id=exp_id,
        scenario_name=filename,
        scenario=scenario
    )

    thread = threading.Thread(target=run_experiment, args=(experiment,), daemon=True)
    experiment.thread = thread
    ACTIVE_EXPERIMENTS[exp_id] = experiment
    thread.start()

    return jsonify({"ok": True, "experiment_id": exp_id})

@app.route("/api/experiment/<exp_id>/stop", methods=["POST"])
def api_stop_experiment(exp_id):
    exp = ACTIVE_EXPERIMENTS.get(exp_id)
    if not exp:
        return jsonify({"error": "Experiment nicht gefunden"}), 404
    exp.stop_event.set()
    exp.pause_event.set()  # unpause so thread can exit
    return jsonify({"ok": True})

@app.route("/api/experiment/stop-all", methods=["POST"])
def api_stop_all_experiments():
    """Stop all running experiments"""
    stopped = 0
    for exp_id, exp in ACTIVE_EXPERIMENTS.items():
        if exp.status in ("running", "paused"):
            exp.stop_event.set()
            exp.pause_event.set()
            stopped += 1
    return jsonify({"ok": True, "stopped": stopped})

@app.route("/api/health-check", methods=["POST"])
def api_health_check():
    """Check if a service URL is reachable"""
    data = request.get_json(silent=True) or {}
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "url required"}), 400
    result = check_service_health(url)
    return jsonify(result)

@app.route("/api/experiment/<exp_id>/pause", methods=["POST"])
def api_pause_experiment(exp_id):
    exp = ACTIVE_EXPERIMENTS.get(exp_id)
    if not exp:
        return jsonify({"error": "Experiment nicht gefunden"}), 404
    exp.pause_event.clear()
    exp.status = "paused"
    return jsonify({"ok": True})

@app.route("/api/experiment/<exp_id>/resume", methods=["POST"])
def api_resume_experiment(exp_id):
    exp = ACTIVE_EXPERIMENTS.get(exp_id)
    if not exp:
        return jsonify({"error": "Experiment nicht gefunden"}), 404
    exp.pause_event.set()
    exp.status = "running"
    return jsonify({"ok": True})

@app.route("/api/experiment/<exp_id>/status", methods=["GET"])
def api_experiment_status(exp_id):
    exp = ACTIVE_EXPERIMENTS.get(exp_id)
    if not exp:
        return jsonify({"error": "Experiment nicht gefunden"}), 404
    return jsonify({
        "experiment_id": exp.experiment_id,
        "scenario_name": exp.scenario_name,
        "status": exp.status,
        "round_count": exp.round_count,
        "message_count": len(exp.messages),
        "win_result": exp.win_result,
        "started_at": exp.started_at
    })

@app.route("/api/experiment/<exp_id>/messages", methods=["GET"])
def api_experiment_messages(exp_id):
    exp = ACTIVE_EXPERIMENTS.get(exp_id)
    if not exp:
        return jsonify({"error": "Experiment nicht gefunden"}), 404
    return jsonify({"messages": exp.messages})

@app.route("/api/experiment/<exp_id>/stream")
def api_experiment_stream(exp_id):
    """Server-Sent Events stream for live experiment updates"""
    exp = ACTIVE_EXPERIMENTS.get(exp_id)
    if not exp:
        return jsonify({"error": "Experiment nicht gefunden"}), 404

    def generate():
        while True:
            try:
                event = exp.event_queue.get(timeout=30)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("win", "system") and exp.status in ("finished", "stopped"):
                    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    })

@app.route("/api/experiments", methods=["GET"])
def api_list_experiments():
    """List all active/recent experiments"""
    result = []
    for exp_id, exp in ACTIVE_EXPERIMENTS.items():
        result.append({
            "experiment_id": exp.experiment_id,
            "scenario_name": exp.scenario_name,
            "status": exp.status,
            "round_count": exp.round_count,
            "started_at": exp.started_at
        })
    return jsonify({"experiments": result})

# ---------- Transcripts ----------

@app.route("/api/transcripts", methods=["GET"])
def api_list_transcripts():
    transcripts = []
    for f in sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                transcripts.append({
                    "filename": f.stem,
                    "scenario_name": data.get("scenario_name", ""),
                    "started_at": data.get("started_at", ""),
                    "finished_at": data.get("finished_at", ""),
                    "status": data.get("status", ""),
                    "total_rounds": data.get("total_rounds", 0),
                    "win_result": data.get("win_result")
                })
        except Exception:
            pass
    return jsonify({"transcripts": transcripts})

@app.route("/api/transcripts/<filename>", methods=["GET"])
def api_get_transcript(filename):
    path = TRANSCRIPTS_DIR / f"{filename}.json"
    if not path.exists():
        return jsonify({"error": "Transcript nicht gefunden"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    server_cfg = CONFIG.get("server", {})
    app.run(
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8050),
        debug=server_cfg.get("debug", True),
        threaded=True
    )

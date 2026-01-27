from flask import Flask, jsonify, request, Response
import requests
import time
import os
app = Flask(__name__)

# -------------------------------------------------
# BASIC DEMO
# -------------------------------------------------

@app.get("/")
def home():
    return jsonify({
        "ok": True,
        "message": "Hallo Meister!",
        "data": {
            "demo": "json response",
            "version": 1
        }
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


# -------------------------------------------------
# GET MIT QUERY PARAMS
# /echo?name=Meister&age=99
# -------------------------------------------------

@app.get("/echo")
def echo():
    name = request.args.get("name", "unknown")
    age = request.args.get("age", "not-set")

    return jsonify({
        "ok": True,
        "received": {
            "name": name,
            "age": age
        }
    })


# -------------------------------------------------
# POST MIT JSON BODY
# -------------------------------------------------

@app.post("/post-demo")
def post_demo():
    data = request.get_json(silent=True) or {}

    return jsonify({
        "ok": True,
        "received": data
    })


# -------------------------------------------------
# GET → WEITERLEITEN AN ANDEREN SERVICE
# /proxy-get?text=Hallo
# -------------------------------------------------

@app.get("/proxy-get")
def proxy_get():
    text = request.args.get("text", "")

    # Beispiel-Ziel (kann ein Container-Service sein)
    target_url = "http://example.com/api"

    try:
        r = requests.get(target_url, params={"text": text}, timeout=5)
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type")
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------------------------------
# POST → WEITERLEITEN + RESPONSE DURCHREICHEN
# -------------------------------------------------

@app.post("/proxy-post")
def proxy_post():
    payload = request.get_json(silent=True) or {}

    target_url = "http://example.com/api"

    try:
        r = requests.post(target_url, json=payload, timeout=5)
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type")
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -------------------------------------------------
# INTERNER SERVICE → ANDERER CONTAINER
# Beispiel: Weiterleitung an text_to_speech
# -------------------------------------------------

@app.post("/call-tts")
def call_tts():
    payload = request.get_json(silent=True) or {}

    # Service-Name aus docker-compose!
    tts_url = "http://127.0.0.1:5003/tts"

    try:
        r = requests.post(tts_url, json=payload, timeout=30)
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type")
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
def split_username_and_text(user_text: str):
    if ":" in user_text:
        username, text = user_text.split(":", 1)
        return username.strip(), text.strip()
    else:
        return None, user_text.strip()
#{text=`$t;play_audio=`$true;save_wav=`$false;wav_path=''out\only.wav''}
@app.get("/call-tts/<path:text>")
def call_tts_get(text):
    # Service-Name aus docker-compose!
    tts_url = "http://127.0.0.1:5003/tts"
    username, text = split_username_and_text(text)
    text = username + " sagt: " + text
    payload = {
        "text": text,
        "play_audio": True,
        "save_wav": False,
        "wav_path": r"out\only.wav",
    }

    try:
        r = requests.post(tts_url, json=payload, timeout=30)
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type")
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

 

 #{text=$t;play_audio=$true;save_wav=$false;wav_path='out\only.wav'}
@app.get("/call-tts-dilara/<path:text>")
def call_tts_dilara_get(text):
    # 1️⃣ An Chat-Service senden
    send_this = {
        "message": text
    }

    chat_url = "http://127.0.0.1:5001/chat"

    try:
        r = requests.post(chat_url, json=send_this, timeout=30)
        r.raise_for_status()
        reply = r.json()["reply"]
        emotion = r.json()["emotion"]
        print(reply,flush=True)
        send_this_tts ={"text": {"value":reply,"emotion":emotion}}
        send_this_emotion ={"emotion":emotion}
        tts_url2 = "http://127.0.0.1:5003/tts"
        print(send_this_tts,flush=True)
        print(send_this_emotion,flush=True)

        try:


            r = requests.post(tts_url2, json=send_this_tts, timeout=30)

 
            return Response(
                r.content,
                status=r.status_code,
                content_type=r.headers.get("Content-Type")
            )
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

 
# -------------------------------------------------
# START
# -------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)

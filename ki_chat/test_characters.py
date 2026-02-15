#!/usr/bin/env python3
"""
Test Script für das Multi-Character System
"""
import requests
import json
import time

BASE_URL = "http://localhost:5001"

def test_list_characters():
    print("\n=== TEST: List Characters ===")
    r = requests.get(f"{BASE_URL}/characters")
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Characters: {data['characters']}")
    print(f"Current: {data['current']}")
    return data

def test_current_character():
    print("\n=== TEST: Current Character ===")
    r = requests.get(f"{BASE_URL}/character/current")
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Name: {data['name']}")
        print(f"DB: {data['db_path']}")
        print(f"Model: {data['model']}")
        print(f"Thinking Rate: {data['thinking_rate']}")
        print(f"Auto Memory: {data['enable_auto_memory']}")
        print(f"Pervy Guard: {data['enable_pervy_guard']}")
        return data
    else:
        print(f"Error: {r.json()}")
        return None

def test_switch_character(char_name):
    print(f"\n=== TEST: Switch to {char_name} ===")
    r = requests.post(
        f"{BASE_URL}/character/switch",
        json={"character": char_name}
    )
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    return data

def test_chat(username, message):
    print(f"\n=== TEST: Chat ({username}) ===")
    print(f"Message: {message}")
    r = requests.post(
        f"{BASE_URL}/chat",
        json={"message": f"{username}:{message}"}
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Reply: {data['reply']}")
        print(f"Emotion: {data['emotion']}")
        return data
    else:
        print(f"Error: {r.text}")
        return None

def test_full_workflow():
    print("=" * 60)
    print("MULTI-CHARACTER SYSTEM TEST")
    print("=" * 60)
    
    # 1. Liste Characters
    chars_data = test_list_characters()
    
    # 2. Aktueller Character
    test_current_character()
    
    # 3. Chat mit Dilara
    test_chat("testuser", "hey dilara wie gehts?")
    time.sleep(1)
    
    # 4. Zu Alex wechseln (wenn vorhanden)
    if "alex" in chars_data.get("characters", []):
        test_switch_character("alex")
        time.sleep(1)
        
        test_current_character()
        
        # Chat mit Alex
        test_chat("testuser", "hey alex was geht?")
        time.sleep(1)
        
        # Zurück zu Dilara
        test_switch_character("dilara")
        time.sleep(1)
        
        test_current_character()
        
        # Chat mit Dilara
        test_chat("testuser", "bin wieder da dilara")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_full_workflow()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server!")
        print(f"   Make sure the server is running on {BASE_URL}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

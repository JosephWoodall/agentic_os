import socket
import time
import json
import os
import urllib.request
import urllib.error

# To use a real LLM, set these environment variables:
# LLM_PROVIDER=openai or LLM_PROVIDER=anthropic or LLM_PROVIDER=ollama
# LLM_API_KEY=your_key_here
# LLM_MODEL=gpt-4-turbo, qwen2.5:32b, etc.

def call_llm(prompt):
    provider = os.getenv("LLM_PROVIDER", "mock")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        url = f"{base_url.rstrip('/')}/api/generate"
        payload = {
            "model": model or "qwen2.5:32b",
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096
            }
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as f:
                res = json.loads(f.read().decode('utf-8'))
                return res['response']
        except Exception as e:
            return json.dumps({"command": "yield", "error": str(e)})

    elif provider == "openai":
        if not api_key: return "{\"error\": \"Missing LLM_API_KEY\"}"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model or "gpt-4-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as f:
                res = json.loads(f.read().decode('utf-8'))
                return res['choices'][0]['message']['content']
        except Exception as e:
            return json.dumps({"command": "yield", "error": str(e)})

    elif provider == "anthropic":
        if not api_key: return "{\"error\": \"Missing LLM_API_KEY\"}"
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model or "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "system": "Respond with ONLY valid JSON syscalls for Agentic OS. No preamble."
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as f:
                res = json.loads(f.read().decode('utf-8'))
                return res['content'][0]['text']
        except Exception as e:
            return json.dumps({"command": "yield", "error": str(e)})
    
    else:
        # Fallback to local keyword matching (Internal Mock)
        prompt_lower = prompt.lower()
        if "spawn" in prompt_lower or "run" in prompt_lower:
            name = "browser" if "browser" in prompt_lower else "user_task"
            if "terminal" in prompt_lower: name = "terminal"
            return json.dumps({"command": "spawn_process", "name": name, "priority": 5})
        elif "kill" in prompt_lower:
            return json.dumps({"command": "kill_process", "pid": 3})
        elif "read" in prompt_lower:
            return json.dumps({"command": "read_fs", "path": "/var/log/system.log"})
        elif "hello" in prompt_lower or "hi" in prompt_lower or "help" in prompt_lower:
            return json.dumps({"command": "query_state"})
        else:
            return json.dumps({"command": "query_state"})

def check_connectivity():
    provider = os.getenv("LLM_PROVIDER", "mock")
    
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        url = f"{base_url.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=2) as f:
                print(f"[+] Local Ollama detected.")
                return True
        except Exception:
            print(f"[!] ERROR: Cannot connect to Ollama at {url}")
            print(f"    Make sure you have run 'ollama serve' and the port is open.")
            return False
    elif provider == "mock":
        return True
    else:
        # For OpenAI/Anthropic, just check if API key is set
        if not os.getenv("LLM_API_KEY"):
            print(f"[!] ERROR: LLM_API_KEY is not set for provider '{provider}'")
            return False
        return True

def main():
    HOST = '127.0.0.1'
    PORT = 5557
    print(f"[*] Starting Agentic OS LLM bridge on {HOST}:{PORT}")
    print(f"[*] Mode: {os.getenv('LLM_PROVIDER', 'mock')}")
    
    if not check_connectivity():
        print("[!] LLM connectivity check failed. The kernel will use mock fallback.")
    
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5.0)
                s.connect((HOST, PORT))
                # Remove socket timeout after connection so long inferences don't break the serial connection
                s.settimeout(None)
                print("[*] Connected to Agentic OS Kernel via Serial COM1")
                
                buffer = ""
                in_prompt = False
                current_prompt = ""
                
                while True:
                    data = s.recv(4096)
                    if not data: break
                    buffer += data.decode('utf-8', errors='ignore')
                    
                    while True:
                        if not in_prompt:
                            if "---PROMPT---" in buffer:
                                pre, post = buffer.split("---PROMPT---", 1)
                                buffer = post
                                in_prompt = True
                                current_prompt = ""
                            else:
                                if '\n' in buffer:
                                    lines = buffer.split('\n')
                                    buffer = lines.pop()
                                    for line in lines:
                                        if line.strip(): print(f"[KERNEL] {line.strip()}")
                                break
                        else:
                            if "---END---" in buffer:
                                prompt_part, post = buffer.split("---END---", 1)
                                current_prompt += prompt_part
                                buffer = post
                                in_prompt = False
                                
                                prompt_clean = current_prompt.strip()
                                print(f"\n[>] KERNEL REQUESTED INFERENCE ({len(prompt_clean)} chars)")
                                response = call_llm(prompt_clean)
                                
                                # Basic JSON validation
                                try:
                                    clean_resp = response.strip()
                                    if clean_resp.startswith("```json"):
                                        clean_resp = clean_resp.split("```json")[1].split("```")[0].strip()
                                    elif clean_resp.startswith("```"):
                                        clean_resp = clean_resp.split("```")[1].split("```")[0].strip()
                                    
                                    json.loads(clean_resp)
                                    print(f"[<] Sending response: {clean_resp}")
                                    s.sendall(f"{clean_resp}\n---END_RESPONSE---\n".encode('utf-8'))
                                except json.JSONDecodeError:
                                    print(f"[!] LLM sent invalid JSON: {response}")
                                    fallback = '{"command": "yield"}'
                                    s.sendall(f"{fallback}\n---END_RESPONSE---\n".encode('utf-8'))
                            else:
                                if len(buffer) > 10:
                                    to_take = len(buffer) - 10
                                    current_prompt += buffer[:to_take]
                                    buffer = buffer[to_take:]
                                break
                                
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(1)
        except Exception as e:
            print(f"[!] Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()

import socket
import time
import json

def generate_response(prompt):
    prompt_lower = prompt.lower()
    if "spawn" in prompt_lower or "run" in prompt_lower:
        name = "browser" if "browser" in prompt_lower else "user_task"
        if "terminal" in prompt_lower: name = "terminal"
        return json.dumps({"command": "spawn_process", "name": name, "priority": 5})
    elif "kill" in prompt_lower:
        return json.dumps({"command": "kill_process", "pid": 3})
    elif "read" in prompt_lower:
        return json.dumps({"command": "read_fs", "path": "/var/log/system.log"})
    else:
        return json.dumps({"command": "query_state"})

def main():
    HOST = '127.0.0.1'
    PORT = 5557
    print(f"[*] Starting LLM bridge on {HOST}:{PORT}")
    
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
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
                                for line in pre.split('\n'):
                                    if line.strip(): print(f"[KERNEL] {line.strip()}")
                                buffer = post
                                in_prompt = True
                                current_prompt = ""
                                print("\n[>] KERNEL REQUESTED INFERENCE:")
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
                                print(f"[+] Received prompt ({len(prompt_clean)} chars).")
                                response = generate_response(prompt_clean)
                                print(f"[<] Sending response: {response}")
                                s.sendall(f"{response}\n---END_RESPONSE---\n".encode('utf-8'))
                            else:
                                # We have to wait for more data to be sure we don't have a partial ---END---
                                # but we can safely take anything before the last 10 chars
                                if len(buffer) > 10:
                                    to_take = len(buffer) - 10
                                    current_prompt += buffer[:to_take]
                                    buffer = buffer[to_take:]
                                break
                                
        except ConnectionRefusedError:
            time.sleep(1)
        except Exception as e:
            print(f"[!] Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()

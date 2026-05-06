import socket
import time
import json
import sys
import re

# To use AirLLM, uncomment these lines and follow AirLLM's documentation:
# from airllm import AutoModel
# model = AutoModel.from_pretrained("Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF")

def generate_response(prompt):
    """
    This is where you plug in your heavy LLM!
    Whether you are using AirLLM, vLLM, or Ollama, you pass the `prompt` string
    to your model and return the JSON response string.
    
    For example, with AirLLM:
    # input_text = [ "You are a bare-metal kernel scheduler. " + prompt ]
    # input_tokens = model.tokenizer(input_text, return_tensors="pt")
    # generation_output = model.generate(input_tokens.input_ids.cuda(), max_new_tokens=50)
    # return model.tokenizer.decode(generation_output[0])
    """
    
    # --- MOCK RESPONSE FOR TESTING THE BRIDGE ---
    # We parse the prompt just to give a somewhat relevant mock response
    prompt_lower = prompt.lower()
    
    if "spawn" in prompt_lower or "run" in prompt_lower:
        name = "browser" if "browser" in prompt_lower else "user_task"
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
                    data = s.recv(1024)
                    if not data:
                        break
                        
                    text = data.decode('utf-8', errors='ignore')
                    buffer += text
                    
                    # Print UEFI logs passing through the serial port
                    if not in_prompt:
                        lines = buffer.split('\n')
                        buffer = lines.pop()
                        for line in lines:
                            if "---PROMPT---" in line:
                                print("\n[>] KERNEL REQUESTED INFERENCE:")
                                in_prompt = True
                                current_prompt = ""
                            else:
                                print(f"[KERNEL] {line}")
                    else:
                        if "---END---" in buffer:
                            parts = buffer.split("---END---")
                            current_prompt += parts[0].replace("---PROMPT---", "").strip()
                            buffer = parts[1]
                            in_prompt = False
                            
                            print(f"[+] Received prompt ({len(current_prompt)} chars).")
                            print("-------------------------------------------------")
                            print(current_prompt)
                            print("-------------------------------------------------")
                            
                            # GENERATE RESPONSE
                            response = generate_response(current_prompt)
                            
                            print(f"[<] Sending response: {response}")
                            
                            # Send response back to kernel over serial
                            # We send it with the exact termination string the kernel expects
                            msg = f"{response}\n---END_RESPONSE---\n"
                            s.sendall(msg.encode('utf-8'))
                            
                        else:
                            current_prompt += buffer
                            buffer = ""
                            
        except ConnectionRefusedError:
            print(f"[-] Waiting for QEMU to start on {PORT}...")
            time.sleep(2)
        except Exception as e:
            print(f"[!] Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()

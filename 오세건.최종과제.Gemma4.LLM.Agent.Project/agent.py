
import subprocess
import json
import requests
import re 

import time

# 1. SFML 게임 실행 (파이썬이 게임의 입출력을 가로챔)
game_process = subprocess.Popen(
    ['./llm_game'], 
    stdout=subprocess.PIPE, 
    stdin=subprocess.PIPE, 
    text=True,
    bufsize=1
)


def get_gemma_move(state_json):
    data = json.loads(state_json)
    px, py = data['player']
    ix, iy = data['item']
    
    dx = ix - px
    dy = iy - py
    
    radar = []
    if dy < 0: radar.append(f"{-dy} Up")
    elif dy > 0: radar.append(f"{dy} Down")
    
    if dx > 0: radar.append(f"{dx} Right")
    elif dx < 0: radar.append(f"{-dx} Left")
    
    status_text = ", ".join(radar)
    
    if not status_text: 
        return "d"
    
    url = "http://127.0.0.1:8080/v1/chat/completions"
    
    messages = [
        {
            "role": "system",
            "content": "You are a pathfinding AI. Output EXACTLY ONE bracketed letter based on the Target.\nUp=[w], Left=[a], Down=[s], Right=[d]."
        },
        {"role": "user", "content": "Target: 4 Right, 2 Down"},
        {"role": "assistant", "content": "[d]"},
        
        {"role": "user", "content": "Target: 3 Up"},
        {"role": "assistant", "content": "[w]"},
        
        {"role": "user", "content": "Target: 1 Left, 5 Down"},
        {"role": "assistant", "content": "[a]"},
        
        # 실제 LLM에게 던지는 진짜 질문
        {"role": "user", "content": f"Target: {status_text}"}
    ]
    
    payload = {
        "messages": messages,
        "max_tokens": 5,      # 잘림 방지를 위해 토큰 여유를 조금 더 줍니다
        "temperature": 0.0
        # stop 조건은 삭제하여 자연스럽게 출력이 끝나도록 둡니다
    }
    
    try:
        response = requests.post(url, json=payload).json()
        reply = response['choices'][0]['message']['content'].strip().lower()
        
        # [수정됨] 배열의 '마지막(-1)' 요소를 출력하여 실제 던진 질문을 보여줍니다.
        print(f"  [Prompt]: {messages[-1]['content']}")
        print(f"  [LLM Raw Output]: '{reply}'") 
        
        # [수정됨] 괄호가 열려있든 잘렸든 상관없이, 응답에 포함된 w, a, s, d 중 첫 번째 문자를 무조건 추출합니다.
        match = re.search(r'([wasd])', reply)
        if match:
            return match.group(1)
                
    except Exception as e:
        print("API 오류:", e)
        
    return "w" if dy < 0 else "d"
    
    

print("Gemma AI Agent 시작됨...")

# 2. 게임 루프 (게임이 끝날 때까지 반복)
while True:
    # 게임에서 나오는 상태 정보(JSON) 읽기
    line = game_process.stdout.readline()
    if not line:
        break
        
    state_str = line.strip()
    if state_str.startswith("{"):
        print(f"현재 상태: {state_str}")
        
        # LLM에게 묻기
        move = get_gemma_move(state_str)
        print(f"Gemma의 선택: {move}")
        
        # 게임으로 명령어 전송
        game_process.stdin.write(move + "\n")
        game_process.stdin.flush()
        
        # 게임이 너무 빨리 도는 것을 방지 (Gemma가 생각하는 시간 + 관전용)
        time.sleep(0.1)

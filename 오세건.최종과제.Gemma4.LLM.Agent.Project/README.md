# 오세건 최종과제

# Gemma4 LLM Agent Project

## Source 디렉토리 참고

proj
└ llama.cpp						(LLM 추론 플랫폼)
	└ build
└ llm_game.cpp 					(LLM Interactive Environment)
└ agent.py						(로컬 LLM 서버와 Env를 연결, 구동하는 프로그램)
└ gemma-4-E2B-it-IQ4_XS.gguf	(GEMMA 4 모델 가중치 파일)



## RPI 기본 빌드 환경 설정
sudo apt-get update
sudo apt-get install build-essential cmake git clang


## llama.cpp 다운로드 후 빌드
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release -j 2


## gemma4 모델 다운로드
wget https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-IQ4_XS.gguf


## SFML 라이브러리 설치 및 Environment 코드 빌드 (llm_game.cpp)
sudo apt update
sudo apt install libsfml-dev

g++ llm_game.cpp -o llm_game -std=c++17 -pthread -lsfml-graphics -lsfml-window -lsfml-system

### - 이 Env 프로그램은 Env 프로그램과 LLM 서버를 연결해주는 agent.py 로 구동됨


## LLM CLI 테스트
./llama.cpp/build/bin/llama-cli --reasoning-budget 60 -m gemma-4-E2B-it-IQ4_XS.gguf -c 2048 -t 4 -n 512 -cnv

## LLM Agent 테스트
./llama.cpp/build/bin/llama-server --reasoning-budget 0 -m gemma-4-E2B-it-IQ4_XS.gguf -c 256 -t 4 -ngl 99 --port 8080 > /dev/null 2>&1 &
DISPLAY=:0 python ./agent.py

### - RPI의 기본 디스플레이 :0 으로 실행 창 출력. 


## 에이전트 백그라운드 서버 종료시
pkill llm_server


------------------------------------------------------------------------------------------------------------------


## [모델 다운로드 링크들 (확장자 gguf)]


## [GEMMA 2]
### gemma-2-2b-it-Q4_K_M.gguf
	wget https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf

### gemma-2-2b-it-IQ3_M
	wget https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-IQ3_M.gguf

### gemma-2-2b-it-IQ2_M
	wget https://huggingface.co/legraphista/gemma-2-2b-it-IMat-GGUF/resolve/main/gemma-2-2b-it.IQ2_XXS.gguf


## [GEMMA 4]
### gemma4 E2B IT UD IQ3 XXS
	wget https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-IQ3_XXS.gguf

### gemma4 E2B IT IQ4 XS
	wget https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-IQ4_XS.gguf

### gemma4 E2B IT QAT IQ4 XL
	wget https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf


## [ETC]
### TinyLlama-1.1B-Chat-v1.0-GGUF
	wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

### Llama-3.2-1B-Instruct
	wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf

### Qwen2.5-0.5B-Instruct
	wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf





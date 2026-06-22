#include <SFML/Graphics.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>
#include <string>

// 게임 설정
const int GRID_SIZE = 40;
const int COLS = 20;
const int ROWS = 15;
const int WIDTH = COLS * GRID_SIZE;
const int HEIGHT = ROWS * GRID_SIZE;

// 게임 상태 변수 (스레드 간 공유)
std::mutex gameMutex;
int playerX = 0, playerY = 0;
int itemX = 5, itemY = 5;
int score = 0;
std::atomic<bool> isRunning(true);

// 랜덤 생성기 초기화
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> distCol(0, COLS - 1);
std::uniform_int_distribution<int> distRow(0, ROWS - 1);

// 아이템 랜덤 스폰 함수 (뮤텍스 락이 걸린 상태에서 호출해야 함)
void spawnItem()
{
    do {
        itemX = distCol(gen);
        itemY = distRow(gen);
    } while (itemX == playerX && itemY == playerY); // 플레이어와 겹치지 않게
}

// 텍스트 통신 스레드 (LLM과의 상호작용 담당)
void consoleCommunicationThread()
{
    std::string command;
    while (isRunning)
    {
        // 1. 현재 상태 출력 (LLM이 파싱하기 쉽게 JSON 형태로 출력)
        {
            std::lock_guard<std::mutex> lock(gameMutex);
            std::cout << "{\"player\": [" << playerX << ", " << playerY
                << "], \"item\": [" << itemX << ", " << itemY
                << "], \"score\": " << score << "}\n";
            //std::cout << "명령 대기 (w/a/s/d): " << std::flush;
        }

        // 2. 명령어 입력 대기 (여기서 멈춰있어도 그래픽 화면은 멈추지 않음)
        std::cin >> command;
        if (!isRunning) break;

        // 3. 명령어 파싱 및 로직 업데이트
        std::lock_guard<std::mutex> lock(gameMutex);
        if (command == "w" && playerY > 0)          playerY--;
        else if (command == "s" && playerY < ROWS - 1)   playerY++;
        else if (command == "a" && playerX > 0)          playerX--;
        else if (command == "d" && playerX < COLS - 1)   playerX++;

        // 4. 아이템 수집 판정
        if (playerX == itemX && playerY == itemY)
        {
            score++;
            spawnItem();
        }
    }
}

int main()
{
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "LLM Interactive Game");
    window.setFramerateLimit(60);

    // 폰트 설정 (점수 표시용)
    // 주의: 실행 파일이 있는 폴더에 폰트 파일(예: arial.ttf)이 있어야 합니다.
    sf::Font font;
    bool hasFont = font.loadFromFile("arial.ttf");

    sf::Text scoreText;
    if (hasFont) {
        scoreText.setFont(font);
        scoreText.setCharacterSize(24);
        scoreText.setFillColor(sf::Color::White);
    }

    // 플레이어 객체 (초록색 네모)
    sf::RectangleShape playerShape(sf::Vector2f((float)GRID_SIZE, (float)GRID_SIZE));
    playerShape.setFillColor(sf::Color::Green);

    // 아이템 객체 (빨간색 동그라미)
    sf::CircleShape itemShape((float)GRID_SIZE / 2.0f);
    itemShape.setFillColor(sf::Color::Red);

    // 콘솔 통신 스레드 시작
    std::thread consoleThread(consoleCommunicationThread);

    // 메인 그래픽 렌더링 루프
    sf::Event event;
    while (window.isOpen())
    {
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                isRunning = false;
                window.close();
            }
        }

        // 화면 지우기
        window.clear(sf::Color(30, 30, 30));

        // 그리기 (상태 읽기 전 뮤텍스로 보호)
        {
            std::lock_guard<std::mutex> lock(gameMutex);

            // 위치 업데이트
            playerShape.setPosition((float)playerX * GRID_SIZE, (float)playerY * GRID_SIZE);
            itemShape.setPosition((float)itemX * GRID_SIZE, (float)itemY * GRID_SIZE);

            if (hasFont) {
                scoreText.setString("Score: " + std::to_string(score));
                // 오른쪽 상단 정렬
                sf::FloatRect textRect = scoreText.getLocalBounds();
                scoreText.setPosition(WIDTH - textRect.width - 20.f, 10.f);
            }

            // 화면에 그리기
            window.draw(itemShape);
            window.draw(playerShape);
            if (hasFont) window.draw(scoreText);
        }

        // 디스플레이 업데이트
        window.display();
    }

    // 창이 닫히면 콘솔 스레드 종료를 위해 강제 종료 (std::cin 블로킹 회피)
    std::exit(0);

    return 0; // 도달하지 않음
}

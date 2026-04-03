# Web Server API Handbook

Use a python environment to run these following command:

```bash
pip install -r requirements.txt
python main_webserver.py
```

## REMINDER: Set your environment variables

Please remember to set the .env file or export your api keys to access poe.com and huggingface:

Example:

```bash
export POE_API_KEY="sk-poe-YOUR_POE_API_KEY"
export HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"
```

## 1. 啟動遊戲 (Start Game)

```http
POST http://127.0.0.1:8080/api/start
```

用途: 初始化遊戲並取得第一幕劇情。

Request Body (JSON):

```json
{
  "player_id": "user_123" \\ A unique string-type user id, could be uuid or username
}
```

Response (JSON):

```json
{
  "status": "success",
  "player_id": "unique_user_id_001",
  "game_state": { "hp": 100.0 },
  "scene_data": {
    "text": "你在一個昏暗的房間醒來...",
    "story_arc": { "current_phase": "Exposition", "description": "故事開始" },
    "game_state": { "player_stats": { "hp": 100.0 }, "location_time": { "place": "Room", "time": "Dawn" } },
    "entities": { "npcs": [], "items": ["Brass Key"] },
    "choice_a": "檢查門",
    "choice_b": "搜索房間",
    "choice_c": "繼續睡覺",
    "end_game": false
  }
}
```

前端需要將回傳的 game_state 存起來（例如存在 Vuex/Redux 或是 Local State），下次請求時要帶上它

## 2. 送出玩家選擇 (Player Action)

```http
POST http://127.0.0.1:8080/api/action
```

用途: 將玩家點擊的按鈕（Choice A/B/C）或是自定義輸入傳給後端。

Request Body (JSON):

```json
{
  "player_id": "user_123",
  "player_choice": "I want to open the wooden door",
  "game_state": {
    "hp": 100.0
  }
}
```

Response (JSON):

```json
{
  "status": "success",
  "game_state": { "hp": 90.0 }, 
  "scene_data": {
    "text": "As you open the door, a trap triggers! You lose 10 HP.",
    "entities": { "npcs": [], "items": [] },
    "choice_a": "Fight the spider",
    "choice_b": "Run away",
    "choice_c": "Use a health potion",
    "end_game": false
  }
}
```

## 建議

非同步等待 (Loading State)：因為 LLM 生成加上兩個 RAG 的檢索需要幾秒鐘的時間，前端在發送 Axios請求時，務必做一個 Loading 動畫（例如轉圈圈），否則玩家會以為遊戲當機了。

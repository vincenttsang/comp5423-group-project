import os
import json
import openai
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==========================================
# 1. 初始化設定 (Configuration)
# ==========================================

app = Flask(__name__)
CORS(app) # 允許前端跨域請求 (CORS)

# Poe API 設定 (請確保環境變數已設定，或直接替換字串)
client_llm = openai.OpenAI(
    api_key=os.getenv("POE_API_KEY"), 
    base_url="https://api.poe.com/v1",
)

# ChromaDB 初始化
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

story_collection = chroma_client.get_or_create_collection(name="story_rag", embedding_function=emb_fn)
progress_collection = chroma_client.get_or_create_collection(name="progress_rag", embedding_function=emb_fn)

# ==========================================
# 2. RAG 與 生成核心函數 (與 main.py 相同)
# ==========================================

def load_story_rag(json_path="cys_excerpt.json"):
    if story_collection.count() > 0:
        return
    print("[System] Loading Story RAG from JSON...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)
        docs, metas, ids = [], [], []
        for story in stories:
            story_id = story['story_id']
            tag = story['tag']
            for i, scenario in enumerate(story['sets']):
                doc_text = f"Context: {scenario['Current_context']} Choice: {scenario['Player_choice']}"
                docs.append(doc_text)
                metas.append({
                    "story_id": story_id,
                    "tag": tag,
                    "expected_output": scenario['Expected_output']
                })
                ids.append(f"ID_{story_id}_{i}")
        story_collection.add(documents=docs, metadatas=metas, ids=ids)
        print(f"[System] Successfully loaded {story_collection.count()} story fragments.")
    except Exception as e:
        print(f"[Error] Failed to load story JSON: {e}")

def query_story_rag(player_action):
    results = story_collection.query(query_texts=[player_action], n_results=1)
    if results['metadatas'] and results['metadatas'][0]:
        return results['metadatas'][0][0]['expected_output']
    return "The environment reacts unexpectedly to your action."

def summarize_and_store_memory(player_id, scene_text, player_choice):
    summary_prompt = f"""
    Please summarize the following game scene into a short narrative memory (1-2 sentences).
    Focus on: Key plot points, character decisions, and current goals.
    DO NOT include game mechanics like HP, stats, or specific inventory items.
    
    SCENE: {scene_text}
    PLAYER ACTION: {player_choice}
    """
    try:
        response = client_llm.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = response.choices[0].message.content
        progress_collection.add(
            documents=[summary],
            metadatas=[{"player_id": player_id}],
            ids=[f"mem_{player_id}_{progress_collection.count()}"]
        )
    except Exception as e:
        print(f"[Warning] Failed to store memory: {e}")

def get_relevant_history(player_id, current_query, n_results=3):
    if progress_collection.count() == 0:
        return "No past memories retrieved."
    results = progress_collection.query(
        query_texts=[current_query],
        n_results=min(n_results, progress_collection.count()),
        where={"player_id": player_id}
    )
    if results['documents'] and results['documents'][0]:
        return " ".join(results['documents'][0])
    return "No past memories retrieved."

def generate_next_scene(player_choice, game_state, story_fragment, game_history):
    system_prompt = f"""
    # ROLE
    You are a creative and adaptive RPG Dungeon Master. 

    # NARRATIVE LOGIC
    1. SCRIPT AS TRUTH: The {{story_fragment}} defines the mandatory plot outcome.
    2. DYNAMIC BRIDGING: Use {{player_choice}} to bridge the narrative.
    3. FORMATTING: Use <color="#hex">key_terms</color> for emphasis.

    # INPUT PARAMETERS
    - Story Fragments: {story_fragment}
    - Game History: {game_history}
    - Player's Last Choice: {player_choice}

    # RESPONSE FORMAT
    Return a JSON object ONLY. Do not include markdown code blocks.
    {{
      "text": "Your creative narration...",
      "story_arc": {{"current_phase": "...", "description": "..."}},
      "game_state": {{"player_stats": {{"hp": {game_state.get('hp', 100.0)}}}, "location_time": {{"place": "...", "time": "..."}}}},
      "entities": {{"npcs": [], "items": []}},
      "choice_a": "First option", "choice_b": "Second option", "choice_c": "Third option",
      "end_game": false
    }}
    
    # GAME RULES & CONSTRAINTS
    - HP Logic: Player HP is a Double. If hp <= 0, set end_game to true. Modify HP logically based on the combat or event outcome.
    - Fixed Choices: You MUST provide exactly three choices (choice_a, choice_b, choice_c).
    """

    response = client_llm.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The player chose: {player_choice}"}
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# ==========================================
# 3. RESTful API Endpoints
# ==========================================

# 在伺服器啟動前先載入劇本 RAG
load_story_rag("cys_excerpt.json")

@app.route('/api/start', methods=['POST'])
def api_start_game():
    """初始化遊戲並返回第一幕場景"""
    data = request.json or {}
    player_id = data.get("player_id", "default_user")
    
    initial_choice = "I wake up and look around."
    initial_state = {"hp": 100.0}
    
    try:
        story_fragment = query_story_rag(initial_choice)
        game_history = get_relevant_history(player_id, initial_choice)
        
        response_json_str = generate_next_scene(initial_choice, initial_state, story_fragment, game_history)
        scene_data = json.loads(response_json_str)
        
        # 提取更新後的 hp
        current_hp = scene_data.get("game_state", {}).get("player_stats", {}).get("hp", initial_state["hp"])
        initial_state["hp"] = current_hp
        
        # 背景儲存記憶
        summarize_and_store_memory(player_id, scene_data.get("text", ""), initial_choice)
        
        return jsonify({
            "status": "success",
            "player_id": player_id,
            "game_state": initial_state,
            "scene_data": scene_data
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/action', methods=['POST'])
def api_player_action():
    """處理玩家行動，推進遊戲"""
    data = request.json
    if not data or 'player_choice' not in data or 'game_state' not in data:
        return jsonify({"status": "error", "message": "Missing 'player_choice' or 'game_state' in request body."}), 400
        
    player_id = data.get("player_id", "default_user")
    player_choice = data["player_choice"]
    game_state = data["game_state"] # 前端傳來的當前狀態 (例如 {"hp": 100.0})

    try:
        # 1. 檢索 RAG
        story_fragment = query_story_rag(player_choice)
        game_history = get_relevant_history(player_id, player_choice)
        
        # 2. 生成新場景
        response_json_str = generate_next_scene(player_choice, game_state, story_fragment, game_history)
        scene_data = json.loads(response_json_str)
        
        # 3. 更新 hp 邏輯 (判斷死亡)
        current_hp = scene_data.get("game_state", {}).get("player_stats", {}).get("hp", game_state.get("hp"))
        game_state["hp"] = current_hp
        
        if current_hp <= 0:
            scene_data["end_game"] = True

        # 4. 儲存新記憶
        summarize_and_store_memory(player_id, scene_data.get("text", ""), player_choice)

        return jsonify({
            "status": "success",
            "game_state": game_state,
            "scene_data": scene_data
        }), 200

    except json.JSONDecodeError:
         return jsonify({"status": "error", "message": "LLM returned invalid JSON."}), 502
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # 啟動 Web Server，預設 Port 8080
    print("\n🚀 [System] Starting Web API Server on http://127.0.0.1:8080 ...\n")
    app.run(host='0.0.0.0', port=8080, debug=True)
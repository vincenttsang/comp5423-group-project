import os
import json
import openai
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. 初始化設定 (Configuration)
# ==========================================

# Poe API 設定
client_llm = openai.OpenAI(
    api_key=os.getenv("POE_API_KEY"),
    base_url="https://api.poe.com/v1",
)

# ChromaDB 初始化
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

# 建立兩個 RAG Collection
story_collection = chroma_client.get_or_create_collection(name="story_rag", embedding_function=emb_fn)
progress_collection = chroma_client.get_or_create_collection(name="progress_rag", embedding_function=emb_fn)

# ==========================================
# 2. Story RAG (劇情檢索)
# ==========================================

def load_story_rag(json_path="cys_excerpt.json"):
    """讀取 JSON 並寫入 Story RAG (若已存在則跳過)"""
    if story_collection.count() > 0:
        return

    print("[System] Loading Story RAG from JSON...")
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

def query_story_rag(player_action):
    """根據玩家行動檢索最相關的劇本真相"""
    results = story_collection.query(query_texts=[player_action], n_results=1)
    if results['metadatas'][0]:
        return results['metadatas'][0][0]['expected_output']
    return "The environment reacts unexpectedly to your action."

# ==========================================
# 3. Progress RAG (記憶管理)
# ==========================================


def summarize_and_store_memory(player_id, scene_text, player_choice):
    """
    非同步調用 LLM 生成摘要，並存入 Progress RAG 
    這確保了記憶庫中不包含繁雜的遊戲數值 (HP/Inventory)
    """
    summary_prompt = f"""
    Please summarize the following game scene into a short narrative memory (1-2 sentences).
    Focus on: Key plot points, character decisions, and current goals.
    DO NOT include game mechanics like HP, stats, or specific inventory items.
    
    SCENE: {scene_text}
    PLAYER ACTION: {player_choice}
    """
    
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

def get_relevant_history(player_id, current_query, n_results=3):
    """檢索與當前場景最相關的過去記憶"""
    if progress_collection.count() == 0:
        return "No past memories retrieved."
        
    results = progress_collection.query(
        query_texts=[current_query],
        n_results=min(n_results, progress_collection.count()),
        where={"player_id": player_id}
    )
    if results['documents'][0]:
        return " ".join(results['documents'][0])
    return "No past memories retrieved."

# ==========================================
# 4. 生成核心 (LLM Generation)
# ==========================================

def generate_next_scene(player_choice, game_state, story_fragment, game_history):
    """組合 Prompt 並調用 LLM 生成下一幕 JSON"""
    system_prompt = f"""
    # ROLE
    You are a creative and adaptive RPG Dungeon Master. 

    # NARRATIVE LOGIC
    1. SCRIPT AS TRUTH: The {{story_fragment}} defines the mandatory plot outcome.
    2. DYNAMIC BRIDGING: Use {{player_choice}} to bridge the narrative.
    3. FORMATTING: Use <color="#hex">key_terms</color> for emphasis.

    # INPUT PARAMETERS
    - Story Fragments (RAG 1): {story_fragment}
    - Game History (RAG 2): {game_history}
    - Player's Last Choice: {player_choice}

    # RESPONSE FORMAT
    Return a JSON object ONLY. Do not include markdown code blocks.
    {{
      "text": "Your creative narration...",
      "story_arc": {{"current_phase": "...", "description": "..."}},
      "game_state": {{"player_stats": {{"hp": {game_state['hp']}}}, "location_time": {{"place": "...", "time": "..."}}}},
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
# 5. 遊戲主迴圈 (Game Loop)
# ==========================================

def start_game():
    load_story_rag("cys_excerpt.json")
    player_id = "test_user_01"
    
    # 初始遊戲狀態
    game_state = {"hp": 100.0}
    player_choice = "I wake up and look around the room." # 遊戲的開場動作
    
    print("\n" + "="*50)
    print("🚀 RPG ENGINE STARTED")
    print("="*50 + "\n")
    
    while True:
        # 1. 檢索雙 RAG 系統
        story_fragment = query_story_rag(player_choice)
        game_history = get_relevant_history(player_id, player_choice)
        
        # 2. 呼叫 LLM 生成場景
        print("[System] Generating next scene...\n")
        response_json_str = generate_next_scene(player_choice, game_state, story_fragment, game_history)
        
        try:
            # 解析 JSON
            scene_data = json.loads(response_json_str)
            
            # 更新本地遊戲狀態 (例如 HP 扣減)
            if "player_stats" in scene_data["game_state"]:
                game_state["hp"] = scene_data["game_state"]["player_stats"].get("hp", game_state["hp"])

            # 3. 渲染 UI (終端機輸出)
            print("-" * 50)
            # 加入 get() 保護，避免 game_state 或 location_time 缺失時崩潰
            place = scene_data.get('game_state', {}).get('location_time', {}).get('place', 'Unknown')
            print(f"❤️  HP: {game_state['hp']} | 📍 {place}")
            print("-" * 50)
            print(scene_data.get("text", "No text generated.") + "\n")
            
            # ==========================================
            # 🐛 修復後的 Entities 安全解析邏輯
            # ==========================================
            entities = scene_data.get("entities", {})
            if isinstance(entities, dict) and (entities.get("npcs") or entities.get("items")):
                print("👀 [Entities Spotted]:")
                
                # 安全解析 NPCs
                npcs = entities.get("npcs", [])
                if isinstance(npcs, list):
                    for npc in npcs:
                        if isinstance(npc, dict):  # 如果 LLM 乖乖給字典
                            name = npc.get('name', 'Unknown')
                            mob_type = npc.get('mob_type', 'unknown')
                            print(f"  - NPC: {name} ({mob_type})")
                        elif isinstance(npc, str): # 如果 LLM 偷懶只給字串
                            print(f"  - NPC: {npc}")
                            
                # 安全解析 Items
                items = entities.get("items", [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict): # 如果 LLM 乖乖給字典
                            name = item.get('name', 'Unknown')
                            print(f"  - Item: {name}")
                        elif isinstance(item, str): # 如果 LLM 偷懶只給字串
                            print(f"  - Item: {item}")
                print()
            # ==========================================
            
            # 判斷遊戲是否結束
            if scene_data.get("end_game", False) or game_state["hp"] <= 0:
                print("\n💀 GAME OVER 💀")
                break
                
            # 顯示選項
            print(f"A) {scene_data['choice_a']}")
            print(f"B) {scene_data['choice_b']}")
            print(f"C) {scene_data['choice_c']}")
            
            # 4. 更新 Progress RAG (背景處理記憶)
            summarize_and_store_memory(player_id, scene_data["text"], player_choice)
            
            # 5. 獲取玩家下一步輸入
            user_input = input("\nYour choice (A/B/C or type custom action): ").strip().upper()
            
            # 將玩家的 A/B/C 對應回實際的文字描述
            if user_input == 'A':
                player_choice = scene_data['choice_a']
            elif user_input == 'B':
                player_choice = scene_data['choice_b']
            elif user_input == 'C':
                player_choice = scene_data['choice_c']
            else:
                player_choice = user_input # 允許自定義動作以測試 RAG 動態橋接
                
        except json.JSONDecodeError:
            print("[Error] Failed to parse JSON. Raw output:")
            print(response_json_str)
            break
        except Exception as e:
            print(f"[Error] An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    start_game()
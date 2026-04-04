import os
import json
import openai
import uuid
import datetime
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import re

# ==========================================
# 1. Configurations
# ==========================================

app = Flask(__name__)
CORS(app) # 允許前端跨域請求 (CORS)

# Server configuration
HOST = '0.0.0.0'
PORT = 8085

# Poe API 設定 (請確保環境變數已設定，或直接替換字串)
client_llm = openai.OpenAI(
    api_key='HF_TOKEN', 
    base_url="https://router.huggingface.co/v1",
)
llm_model_name = 'meta-llama/Llama-3.1-8B-Instruct:sambanova'

# ChromaDB 初始化
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

story_collection = chroma_client.get_or_create_collection(name="story_rag", embedding_function=emb_fn)
progress_collection = chroma_client.get_or_create_collection(name="progress_rag", embedding_function=emb_fn)

# ==========================================
# 2. RAG and Core Game Functions
# ==========================================

def load_story_rag(json_path="cys_excerpt.json"):
    if story_collection.count() > 0:
        return
    print("[System] Loading Story RAG from JSON...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)
        docs = []
        for story in stories:
            id = story['story_id']
            tag = story['tag']
            for i, scenario in enumerate(story['sets']):
                for key, value in scenario.items():
                    doc = {
                        "story_id": id,
                        "scene_tag": tag,
                        "scene_order": i+1,
                        key: value
                    }
                    doc_text = json.dumps(doc, separators=(',', ':'))
                    docs.append(doc_text)
        ids = [str(uuid.uuid4()) for _ in docs]
        story_collection.add(documents=docs, ids=ids)
        print(f"[System] Successfully loaded {story_collection.count()} story fragments.")
    except Exception as e:
        print(f"[Error] Failed to load story JSON: {e}")

def query_story_rag(progress_context, player_action, current_scene, current_scene_idx, n_results=4):
    if current_scene_idx is None:
        raise ValueError('current_scene_idx is required and must be a numeric value')
    
    effective_n_results = min(n_results, story_collection.count())
    
    query_story_json = json.dumps({
        'progress_context': progress_context,
        'current_scene': current_scene,
        'current_scene_idx': current_scene_idx
    })
    query_playerChoice_json = json.dumps({
        'player_choice': player_action,
        'current_scene_idx': current_scene_idx
    })

    results = story_collection.query(
        query_texts=[query_story_json, query_playerChoice_json], 
        n_results=effective_n_results
    )

    if not results['documents']:
        return "No relevant story fragments found."
    progress_docs = [s for s in results['documents'][0] if s.strip()]
    player_choice_docs = [s for s in results['documents'][1] if s.strip()]
    return progress_docs, player_choice_docs

def load_progress_rag(json_path="adventure_progresses.json"):
    if progress_collection.count() > 0:
        return
    print("[System] Loading Progress (progress_rag) from JSON...")

    if not os.path.exists(json_path):
        print(f"[Warning] Progress file not found at {json_path}. Skipping load.")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            progress = json.load(f)

        for item in progress:
            if isinstance(item, str):
                try:
                    parsed = json.loads(item)
                except json.JSONDecodeError:
                    print(f"[Warning] Skipping invalid JSON item: {item}")
                    continue
            else:
                parsed = item

            if isinstance(parsed, dict):
                adventure_id = parsed.get("adventure_id")
                player_name = parsed.get("player_name")
                scene_text = parsed.get("scene_text")
                player_choice = parsed.get("player_choice")
                timestamp = parsed.get("timestamp")
                scene_idx = parsed.get("scene_idx")
                outcome = parsed.get("outcome")

                progress_entry = {
                    "adventure_id": adventure_id,
                    "player_name": player_name,
                    "scene_text": scene_text,
                    "player_choice": player_choice,
                    "timestamp": timestamp,
                    "scene_idx": scene_idx,
                    "outcome": outcome,
                    "end_game": parsed.get("end_game", False)
                }

                update_progress_rag(progress_entry)

        print(f"[System] Successfully loaded {progress_collection.count()} progress entries into progress_rag.")
    except Exception as e:
        print(f"[Error] Failed to load progress JSON: {e}")

def create_progress_entry(adventure_id, player_name, scene_text, player_choice, scene_idx, outcome, end_game):
    timestamp = datetime.datetime.utcnow().isoformat() + 'Z'
    return {
        "adventure_id": adventure_id,
        "player_name": player_name,
        "scene_text": scene_text,
        "player_choice": player_choice,
        "timestamp": timestamp,
        "scene_idx": scene_idx,
        "outcome": outcome,
        "end_game": end_game
    }

def save_progress_flat(adventure_id, player_name, scene_text, player_choice, scene_idx, outcome, end_game, json_path="adventure_progresses.json"):
    progress_entry = create_progress_entry(adventure_id, player_name, scene_text, player_choice, scene_idx, outcome, end_game)
    return save_progress(progress_entry, json_path)

def save_progress(progress_entry, json_path = "adventure_progresses.json"):
    progress_json = json.dumps(progress_entry, separators=(',', ':'))
    
    file_path = json_path
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                progresses = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            progresses = []
    else:
        progresses = []
    
    progresses.append(progress_json)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(progresses, f, indent=2)
    
    return progress_entry["timestamp"]

def get_all_adventure_id_json(json_path="adventure_progresses.json"):
    """Return a list of all adventures with their IDs and player names"""
    adventures = set()  # Use set to avoid duplicates

    file_path = json_path
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                progresses = json.load(f)
            for item in progresses:
                if isinstance(item, str):
                    try:
                        entry = json.loads(item)
                    except json.JSONDecodeError:
                        continue
                else:
                    entry = item
                if isinstance(entry, dict):
                    adventure_id = entry.get("adventure_id")
                    player_name = entry.get("player_name")
                    if adventure_id and player_name:
                        adventures.add((adventure_id, player_name))
        except Exception as e:
            print(f"[Warning] Error reading {file_path}: {e}")
    
    # Convert set to list of dicts
    result = [{"adventure_id": adv_id, "player_name": p_name} for adv_id, p_name in adventures]
    return json.dumps(result)

def save_and_update_progress_flat(adventure_id, player_name, scene_text, player_choice, scene_idx, outcome, end_game):
    progress_entry = create_progress_entry(adventure_id, player_name, scene_text, player_choice, scene_idx, outcome, end_game)
    save_progress(progress_entry)
    update_progress_rag(progress_entry)

def save_and_update_progress(progress_entry):
    save_progress(progress_entry)
    update_progress_rag(progress_entry)

def update_progress_rag_flat(adventure_id, player_name, scene_text, player_choice, scene_idx, outcome, end_game):
    progress_entry = create_progress_entry(adventure_id, player_name, scene_text, player_choice, scene_idx, outcome, end_game)
    update_progress_rag(progress_entry)

def update_progress_rag(progress_entry):
    # Prepare doc_text: remove timestamp from doc, add to metadata
    doc_copy = progress_entry.copy()
    timestamp_value = doc_copy.pop("timestamp", "")
    doc_text = json.dumps(doc_copy, separators=(',', ':'))

    # Generate new ID
    new_id = str(uuid.uuid4())

    # Metadata
    metadata = {
        "adventure_id": progress_entry["adventure_id"],
        "player_name": progress_entry["player_name"],
        "timestamp": timestamp_value
    }

    # Add to progress_collection
    progress_collection.add(
        documents=[doc_text],
        metadatas=[metadata],
        ids=[new_id]
    )

    print(f"[System] Added new progress entry to progress_rag with ID: {new_id}")

def check_adventure_ended(adventure_id, player_name, json_path="adventure_progresses.json"):
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            progresses = json.load(f)
        for item in progresses:
            if isinstance(item, str):
                try:
                    entry = json.loads(item)
                except json.JSONDecodeError:
                    continue
            else:
                entry = item
            if isinstance(entry, dict):
                if (entry.get("adventure_id") == adventure_id and
                    entry.get("player_name") == player_name and
                    entry.get("end_game") == True):
                    return True
        return False
    except Exception as e:
        print(f"[Error] Failed to check adventure ended: {e}")
        return False

def get_all_ended_adventures(json_path="adventure_progresses.json"):
    ended_entries = []
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            progresses = json.load(f)
        for item in progresses:
            if isinstance(item, str):
                try:
                    entry = json.loads(item)
                except json.JSONDecodeError:
                    continue
            else:
                entry = item
            if isinstance(entry, dict) and entry.get("end_game") == True:
                ended_entries.append(entry)
        return ended_entries
    except Exception as e:
        print(f"[Error] Failed to get ended adventures: {e}")
        return []

def query_progress_rag(adventure_id, player_name, current_scene, player_choice, current_scene_idx, n_results=5):
    if current_scene_idx is None:
        raise ValueError('current_scene_idx is required and must be a numeric value')

    if progress_collection.count() == 0:
        return []

    effective_n_results = min(n_results, progress_collection.count())

    query_json = json.dumps({
        'current_scene': current_scene,
        'player_choice': player_choice,
        'current_scene_idx': current_scene_idx
    }, separators=(',', ':'))
    query_texts = query_json

    results = progress_collection.query(
        query_texts=query_texts,
        n_results=effective_n_results,
        where={
            "$and": [
            {"adventure_id": adventure_id},
            {"player_name": player_name}
        ]}
    )

    docs = []
    if results.get('documents'):
        # Chroma may return nested list for each query text; flatten
        for group in results['documents']:
            if not group:
                continue
            for d in group:
                if isinstance(d, str) and d.strip():
                    docs.append(d)
                elif d is not None:
                    docs.append(str(d))

    # Order by scene_idx extracted from doc payload (if JSON); missing or invalid scene_idx go to end.
    ordered = []
    for i, d in enumerate(docs):
        scene_idx = None
        try:
            loaded = json.loads(d)
            if isinstance(loaded, dict) and 'scene_idx' in loaded:
                idx_val = loaded.get('scene_idx')
                if isinstance(idx_val, (int, float)):
                    scene_idx = float(idx_val)
                else:
                    scene_idx = None
            else:
                scene_idx = None
        except Exception:
            scene_idx = None

        if scene_idx is None:
            order_key = float('inf')
        else:
            order_key = scene_idx

        ordered.append((order_key, d))

    ordered.sort(key=lambda x: x[0])
    return [doc for _, doc in ordered]

def generate_next_scene(last_scene, player_choice, story_fragment, choice_fragment, game_progress, scene_idx):
    system_prompt = f"""
    # ROLE
    You are a creative and adaptive RPG Dungeon Master. 

    # NARRATIVE LOGIC
    1. RESPECT THE SCRIPT: The {{story_fragment}} defines the general direction of the plot outcome. You can add your own creative details, but the core narrative should not deviate from the provided fragment too much.
    2. PLAYER CHOICES REFERENCES: The {{choice_fragment}} gives examples of choices. You can add your own creative ideas, but the tone and the core intent should be similar to the provided examples.
    3. CONTEXTUAL AWARENESS: Use {{game_progress}} to maintain continuity and coherence in the story. This progress node may contain important clues, character developments, and items that should influence the narrative.
    4. IMMEDIATE SCENE FOCUS: The {{last_scene}} is the most recent scene. Build upon it to create a seamless transition to the next scene. Generated output should be highly consistent with the last scene's details and tone.
    5. DYNAMIC BRIDGING: Use {{player_last_choice}} to bridge the narrative.
    6. FORMATTING: Use <color=\"#hex\">key_terms</color> for emphasis.
    7. ENTITY RECOGNITION: Use <character>character_name</character> to indicate a character/NPC. Use <item>item_name</item> for important items. This ensure persistent tracking in the game state.

    # INPUT PARAMETERS
    - story_fragment: {story_fragment}
    - choice_fragment: {choice_fragment}
    - game_progress: {game_progress}
    - last_scene: {last_scene}
    - player_last_choice: {player_choice}
    - scene_idx: {scene_idx}

    # RESPONSE FORMAT
    Return a JSON object ONLY. The JSON object shall be minify. Do not include markdown code blocks. Do not include anything outside of the JSON such as explanation, comments, text, punctuation, control characters etc. The JSON must be valid and parseable.
    It is critical that your WHOLE response is a single JSON object. Text that is not a part of the JSON object will cause parsing errors. 
    The JSON object should have the following structure:
    {{"next_scene":"Your creative narration...","choice_a":"First option","choice_b":"Second option","choice_c":"Third option","end_game":false}}
    
    # GAME RULES & CONSTRAINTS
    - The story should progress with reference to the {{scene_idx}}. The story should end before reaching scene_idx 20. The story should have a clear progression and escalation of stakes, leading to a climax before scene_idx 20.
    - Fixed Choices: You MUST provide exactly three choices (choice_a, choice_b, choice_c). These choices should be non-empty, distinct and lead to different narrative paths. They should be relevant to the current scene and player action. 
    """

    max_retries = 2
    for attempt in range(max_retries):
        try:
            
            response = client_llm.chat.completions.create(
                model=llm_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"The player chose: {player_choice}"}
                ],
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content
            # result = '{"next_scene":"You find yourself in a cold, stone hall with stone walls and a high, vaulted ceiling. The air is damp and musty, and the only light comes from a few torches flickering on the walls. You are tied to a chair, and your head is pounding. You try to remember how you got here, but your memories are hazy. You see a figure standing across the room, a massive Commander with a cruel grin on his face. He seems to be enjoying your predicament. You notice a small, rusty key hanging from a nail on the wall, just out of your reach. What do you do?","choice_a":"Try to reason with the Commander","choice_b":"Struggle against your restraints","choice_c":"Look for a way to reach the key","end_game":false}'
            match = re.search(r'\{.*\}', result, flags=re.DOTALL)
            if match:
                result = match.group(0)
            if validate_scene_json(result):
                return json.loads(result)
            else:
                print(f"[Warning] Invalid JSON structure on attempt {attempt + 1}, retrying...")
        except Exception as e:
            print(f"[Error] LLM request failed on attempt {attempt + 1}: {e}, retrying...")
    
    # If all retries fail, raise an exception
    raise Exception("Failed to generate valid scene JSON after maximum retries")

# ==========================================
# 3. RESTful API Endpoints
# ==========================================

# 在伺服器啟動前先載入劇本 RAG
load_story_rag()
load_progress_rag()

@app.route('/api/start', methods=['POST'])
def api_start_game():
    """初始化遊戲並返回第一幕場景"""
    adventure_id = str(uuid.uuid4())
    data = request.json or {}
    player_name = data.get("player_name", "default_user")
    
    game_progress = "This is the beginning of the adventure. The player has no prior history."
    initial_choice = "I wake up and look around."
    current_scene_idx = 1
    try:
        story_docs, player_choice_docs = query_story_rag(
            progress_context= game_progress, 
            player_action=initial_choice, 
            current_scene=initial_choice,
            current_scene_idx=current_scene_idx)
        
        scene_data = generate_next_scene(
            last_scene=game_progress, 
            player_choice=initial_choice,
            story_fragment=story_docs, 
            choice_fragment=player_choice_docs, 
            game_progress=game_progress, 
            scene_idx=current_scene_idx)
        
        # 背景儲存記憶
        save_and_update_progress_flat(
            adventure_id=adventure_id,
            player_name=player_name,
            scene_text=game_progress, 
            player_choice=initial_choice,
            scene_idx=current_scene_idx,
            outcome=scene_data.get("next_scene", ""),
            end_game=scene_data.get("end_game", False)
        )
        
        return jsonify({
            "status": "success",
            "adventure_id": adventure_id,
            "player_name": player_name,
            "scene_data": scene_data
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/action', methods=['POST'])
def api_player_action():
    """處理玩家行動，推進遊戲"""
    data = request.json
    if not data or 'player_choice' not in data or 'scene_idx' not in data or 'adventure_id' not in data or 'player_name' not in data:
        return jsonify({"status": "error", "message": "Missing required fields in request body."}), 400

    adventure_id = data["adventure_id"]
    player_name = data.get("player_name", "default_user")

    current_scene_idx = data["scene_idx"]
    current_scene = data.get("current_scene", "No previous scene provided.")
    player_choice = data["player_choice"]

    try:
        # 1. 檢索 RAG
        game_progress = query_progress_rag(
            adventure_id=adventure_id,
            player_name=player_name,
            current_scene=current_scene,
            player_choice=player_choice,
            current_scene_idx=current_scene_idx
        )
        story_fragment, player_choice_fragment = query_story_rag(
            progress_context=game_progress, 
            player_action=player_choice, 
            current_scene=current_scene,
            current_scene_idx=current_scene_idx)
        
        # 2. 生成新場景
        scene_data = generate_next_scene(
            last_scene=current_scene, 
            player_choice=player_choice,
            story_fragment=story_fragment, 
            choice_fragment=player_choice_fragment,
            game_progress=game_progress, 
            scene_idx=current_scene_idx
        )
        
        # 4. 儲存新記憶
        save_and_update_progress_flat(
            adventure_id=adventure_id,
            player_name=player_name,
            scene_text=current_scene, 
            player_choice=player_choice,
            scene_idx=current_scene_idx,
            outcome=scene_data.get("next_scene", ""),
            end_game=scene_data.get("end_game", False)
        )

        return jsonify({
            "status": "success",
            "adventure_id": adventure_id,
            "player_id": player_name,
            "scene_data": scene_data
        }), 200

    except json.JSONDecodeError:
         return jsonify({"status": "error", "message": "LLM returned invalid JSON."}), 502
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/api/adventure-ended', methods=['GET'])
def api_get_ended_adventures():
    ended = get_all_ended_adventures()
    return jsonify(ended)

@app.route('/api/adventures', methods=['GET'])
def get_adventures():
    return get_all_adventure_id_json()

# ==========================================
# 4. Helper Functions
# ==========================================
def is_json(my_str):
    try:
        json.loads(my_str)
        return True
    except (ValueError, json.JSONDecodeError):
        return False

def validate_scene_json(json_str):
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return False
        required_keys = ['next_scene', 'choice_a', 'choice_b', 'choice_c', 'end_game']
        for key in required_keys:
            if key not in data:
                return False
        if not isinstance(data['end_game'], bool):
            return False
        for choice in ['choice_a', 'choice_b', 'choice_c']:
            if not isinstance(data[choice], str) or not data[choice].strip():
                return False
        return True
    except (ValueError, json.JSONDecodeError):
        return False
    
# ==========================================
# 5. Main
# ==========================================
def main():
    if __name__ == '__main__':
        # 啟動 Web Server，預設 Port 8080
        print(f"\n🚀 [System] Starting Web API Server on http://{HOST}:{PORT} ...\n")
        app.run(host=HOST, port=PORT, debug=True)

def debug():
    player_name = "test_user_01"
    
    try:
        with app.test_client() as client:
            # Start game
            start_response = client.post('/api/start', json={'player_name': player_name})
            start_data = start_response.get_json()
            print("Start Game Response:")
            print(json.dumps(start_data, indent=2))
            
            if start_data.get('status') != 'success':
                print("Failed to start game")
                return
            
            adventure_id = start_data['adventure_id']
            scene_data = start_data['scene_data']
            current_scene_idx = 1
            last_scene = scene_data.get('next_scene', '')
            
            while not scene_data.get('end_game', False):
                print("\nCurrent Scene:")
                print(json.dumps(scene_data, indent=2))
                
                # Get user choice
                choice = input("Choose (a/b/c): ").strip().lower()
                while choice not in ['a', 'b', 'c']:
                    choice = input("Invalid choice. Choose (a/b/c): ").strip().lower()
                
                player_choice = scene_data[f'choice_{choice}']
                
                # Call action
                action_data = {
                    'player_choice': player_choice,
                    'scene_idx': current_scene_idx,
                    'adventure_id': adventure_id,
                    'player_name': player_name,
                    'last_scene': last_scene
                }
                action_response = client.post('/api/action', json=action_data)
                action_result = action_response.get_json()
                print("\nAction Response:")
                print(json.dumps(action_result, indent=2))
                
                if action_result.get('status') != 'success':
                    print("Action failed")
                    return
                
                scene_data = action_result['scene_data']
                current_scene_idx += 1
                last_scene = scene_data.get('next_scene', '')
            
            print("\nYour adventure has ended!")
            input("Press Enter to exit...")
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

debug()
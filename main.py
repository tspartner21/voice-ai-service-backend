import os
import shutil
import json
import base64
import random
import numpy as np
import librosa
import psycopg2
from psycopg2.extras import RealDictCursor
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
import pytz
from pydub import AudioSegment
from difflib import SequenceMatcher

# 1. 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# [PostgreSQL 설정] - 본인 환경에 맞게 수정
DB_HOST = "localhost"
DB_NAME = "quest_db"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_PORT = "5432"

app = FastAPI()

origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/images", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_kst_now():
    return datetime.now(pytz.timezone('Asia/Seoul'))

# --- DB 연결 ---
def get_db_connection():
    try:
        return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    except Exception as e:
        print(f"❌ DB Fail: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if not conn: return
    try:
        cur = conn.cursor()

        # 테이블 생성
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                                                         username VARCHAR(50) PRIMARY KEY, password VARCHAR(50), role VARCHAR(20), full_name VARCHAR(50), created_at TIMESTAMP
                        );
                    CREATE TABLE IF NOT EXISTS products (
                                                            id VARCHAR(50) PRIMARY KEY, category VARCHAR(50), title VARCHAR(100), price VARCHAR(50),
                        image_url TEXT, description TEXT, persona VARCHAR(50), situation VARCHAR(100), examples TEXT
                        );
                    CREATE TABLE IF NOT EXISTS speaking_logs (
                                                                 id SERIAL PRIMARY KEY, username VARCHAR(50), theme_id VARCHAR(50), user_text TEXT,
                        tech_score INT, feedback TEXT, created_at TIMESTAMP
                        );
                    CREATE TABLE IF NOT EXISTS bookings (
                                                            id SERIAL PRIMARY KEY, username VARCHAR(50), theme_id VARCHAR(50), theme_title VARCHAR(100),
                        reserved_date VARCHAR(20), people INT, status VARCHAR(20) DEFAULT 'confirmed', created_at TIMESTAMP
                        );
                    """)
        cur.execute("INSERT INTO users (username, password, role, full_name, created_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
                    ('1111', '1111', 'user', 'Tester 1111', get_kst_now()))

        seed_products = [
            ("cafe_order", "basic", "☕ Ordering Coffee", "Free", "", "Order drinks like a local", "Barista", "Cafe", '["Iced Americano, please"]'),
            ("subway", "basic", "🚇 Subway Navigation", "Free", "", "Asking for directions", "Citizen", "Subway Station", '["Where is Line 2?"]'),
            ("quest_tour", "offline", "🏯 Palace Tour", "50,000", "https://via.placeholder.com/400", "Hanbok Photo Tour", "Guide", "Gyeongbokgung", '["Take a photo please"]')
        ]
        for p in seed_products:
            cur.execute("INSERT INTO products (id, category, title, price, image_url, description, persona, situation, examples) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING", p)
        conn.commit()
        print("✅ DB Initialized")
    except Exception as e: print(f"Init Error: {e}")
    finally: conn.close()

init_db()

# --- Models ---
class AuthRequest(BaseModel): username: str; password: str
class RegisterRequest(BaseModel): username: str; password: str; full_name: str
class BookingRequest(BaseModel): username: str; theme_id: str; date: str; people: int
class CancelRequest(BaseModel): booking_id: int
class QuestRequest(BaseModel): username: str; theme_id: str

# --- Deep Tech Logic ---
def analyze_audio_similarity(user_path, target_path):
    try:
        y1, sr1 = librosa.load(user_path, sr=16000)
        y2, sr2 = librosa.load(target_path, sr=16000)
        y1, _ = librosa.effects.trim(y1)
        y2, _ = librosa.effects.trim(y2)
        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)
        mfcc1 -= (np.mean(mfcc1, axis=1, keepdims=True) + 1e-8)
        mfcc2 -= (np.mean(mfcc2, axis=1, keepdims=True) + 1e-8)
        dist, path = fastdtw(mfcc1.T, mfcc2.T, dist=cosine, radius=10)
        avg_dist = dist / len(path)
        if avg_dist > 0.6: score = 10
        else: score = int((1 - (avg_dist / 0.6)) * 100)
        if score > 60: score = min(100, score + 15)
        return score
    except: return 0

def check_text_similarity(text1, text2):
    t1 = text1.replace(" ", "").replace(".", "").replace("?", "").lower()
    t2 = text2.replace(" ", "").replace(".", "").replace("?", "").lower()
    return SequenceMatcher(None, t1, t2).ratio() >= 0.85

# --- API ---
@app.post("/login")
def login(req: AuthRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, role FROM users WHERE username=%s AND password=%s", (req.username, req.password))
    user = cur.fetchone()
    conn.close()
    if user: return {"status": "success", "username": user[0], "role": user[1]}
    raise HTTPException(401, "Login failed")

@app.post("/register")
def register(req: RegisterRequest):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password, full_name, created_at) VALUES (%s, %s, %s, %s)", (req.username, req.password, req.full_name, get_kst_now()))
        conn.commit()
        return {"status": "success"}
    except: raise HTTPException(400, "User exists")
    finally: conn.close()

@app.get("/themes")
def get_themes():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM products")
    rows = cur.fetchall()
    conn.close()
    themes = {}
    for r in rows:
        r['icon'] = "🚩" if r['category'] == 'offline' else "💬"
        themes[r['id']] = dict(r)
    return themes

@app.post("/book")
def create_booking(req: BookingRequest):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT title FROM products WHERE id=%s", (req.theme_id,))
        title = cur.fetchone()[0]
        cur.execute("INSERT INTO bookings (username, theme_id, theme_title, reserved_date, people, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                    (req.username, req.theme_id, title, req.date, req.people, get_kst_now()))
        conn.commit()
        return {"status": "success"}
    except: raise HTTPException(500, "Fail")
    finally: conn.close()

@app.get("/bookings/{username}")
def get_bookings(username: str):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM bookings WHERE username=%s ORDER BY created_at DESC", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.post("/bookings/cancel")
def cancel(req: CancelRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM bookings WHERE id=%s", (req.booking_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/reports/{username}")
def get_reports(username: str):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT theme_id, tech_score, created_at FROM speaking_logs WHERE username=%s ORDER BY created_at DESC LIMIT 7", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.post("/quest")
def generate_quest(req: QuestRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    today_start = get_kst_now().replace(hour=0, minute=0, second=0, microsecond=0)
    cur.execute("SELECT user_text FROM speaking_logs WHERE username=%s AND created_at >= %s ORDER BY RANDOM() LIMIT 1", (req.username, today_start))
    row = cur.fetchone()
    conn.close()
    context_text = row[0] if row else "Can I get an iced americano?"

    sys_prompt = "You are a helpful Korean Tutor. Generate a NEW, APPLIED Korean sentence."
    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system", "content": sys_prompt}, {"role":"user", "content": f"Context: {context_text}. Create quest. JSON: {{'korean': '...', 'romanized': '...', 'english': '...', 'grammar': 'Eng explanation', 'context': 'Eng context'}}"}],
        response_format={"type": "json_object"}
    )
    data = json.loads(res.choices[0].message.content)

    # Quest 문장 오디오 생성
    tts = openai_client.audio.speech.create(model="tts-1", voice="nova", input=data['korean'], speed=1.0)
    audio_b64 = base64.b64encode(tts.content).decode('utf-8')

    return {"quest_data": data, "audio_base64": audio_b64}

@app.post("/talk")
async def talk(file: UploadFile = File(...), theme_id: str = Form(...), username: str = Form(...), quest_target: str = Form(None)):
    filename = file.filename
    temp_webm = f"temp_audio/raw_{filename}"
    user_wav = f"temp_audio/in_{filename}.wav"
    target_path = f"temp_audio/tgt_{filename}.mp3"

    try:
        with open(temp_webm, "wb") as b: shutil.copyfileobj(file.file, b)
        try:
            audio = AudioSegment.from_file(temp_webm)
            audio.export(user_wav, format="wav")
        except: shutil.copy(temp_webm, user_wav)

        prompt_lang = "Korean conversation." if quest_target else "English conversation."
        with open(user_wav, "rb") as af:
            res = openai_client.audio.transcriptions.create(model="whisper-1", file=af, prompt=prompt_lang)
        user_text = res.text
        if not user_text.strip(): return {"error": "No voice detected"}

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT persona, situation FROM products WHERE id=%s", (theme_id,))
        r = cur.fetchone()
        conn.close()
        persona, situation = r if r else ("Tutor", "Practice")

        content_match = False
        if quest_target:
            content_match = check_text_similarity(user_text, quest_target)
            data = {
                "korean": quest_target, "romanized": "Pronunciation Practice", "english": "Practice Mode",
                "grammar": "Focus on intonation.", "context": "Quest Challenge"
            }
            target_korean = quest_target
        else:
            sys_prompt = f"""Role: {persona} in {situation}. Task: Translate English to Korean.
            Output JSON: {{
                "korean": "Target Korean", "romanized": "...", "english": "Original English",
                "grammar": "Grammar(Eng)", "context": "Context(Eng)"
            }}"""
            gpt = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys_prompt}, {"role":"user","content":f"User said: {user_text}"}],
                response_format={"type": "json_object"}
            )
            data = json.loads(gpt.choices[0].message.content)
            target_korean = data.get("korean", "")
            content_match = True

        tts = openai_client.audio.speech.create(model="tts-1", voice="nova", input=target_korean, speed=1.0)
        tts.stream_to_file(target_path)
        score = analyze_audio_similarity(user_wav, target_path)
        data['tech_score'] = score
        data['content_match'] = content_match

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO speaking_logs (username, theme_id, user_text, tech_score, feedback, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                    (username, theme_id, user_text, score, data.get('context', ''), get_kst_now()))
        conn.commit()
        conn.close()

        # [수정] 오디오에는 '한글 문장'만 담습니다 (컨텍스트 재생 방지)
        tts_final = openai_client.audio.speech.create(model="tts-1", voice="nova", input=target_korean, speed=1.0)
        audio_b64 = base64.b64encode(tts_final.content).decode('utf-8')

        return {"user_text": user_text, "structured_data": data, "audio_base64": audio_b64}
    except Exception as e: return {"error": str(e)}
    finally:
        for p in [temp_webm, user_wav, target_path]:
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
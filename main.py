import os
import shutil
import json
import base64
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

# 1. 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# [PostgreSQL 설정] 본인 환경에 맞게 수정
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

# [핵심] 한국 시간 구하기 (DB 저장용)
def get_kst_now():
    return datetime.now(pytz.timezone('Asia/Seoul'))

# --- 2. DB 연결 및 초기화 ---
def get_db_connection():
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        return conn
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

        # 초기 데이터
        cur.execute("INSERT INTO users (username, password, role, full_name, created_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
                    ('1111', '1111', 'user', 'Tester 1111', get_kst_now()))

        seed_products = [
            ("cafe_order", "basic", "☕ 카페 주문하기", "Free", "", "직원에게 주문", "알바생", "카페", '["아이스 아메리카노 주세요"]'),
            ("subway", "basic", "🚇 지하철 길찾기", "Free", "", "환승역 물어보기", "시민", "지하철역", '["2호선 어디로 가요?"]'),
            ("quest_tour", "offline", "🏯 경복궁 한복 투어", "50,000", "https://via.placeholder.com/400", "인생샷 투어", "작가", "경복궁", '["사진 찍어주세요"]')
        ]
        for p in seed_products:
            cur.execute("""
                        INSERT INTO products (id, category, title, price, image_url, description, persona, situation, examples)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                        """, p)

        conn.commit()
        print("✅ DB Initialized (KST)")
    except Exception as e:
        print(f"❌ Init Error: {e}")
    finally:
        conn.close()

init_db()

# --- Models ---
class AuthRequest(BaseModel): username: str; password: str
class RegisterRequest(BaseModel): username: str; password: str; full_name: str
class BookingRequest(BaseModel): username: str; theme_id: str; date: str; people: int
class CancelRequest(BaseModel): booking_id: int

# --- Deep Tech ---
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
        if avg_dist > 0.6: final_score = 10
        else: final_score = int((1 - (avg_dist / 0.6)) * 100)
        if final_score > 60: final_score = min(100, final_score + 15)
        return final_score
    except: return 0

# --- API ---
@app.post("/login")
def login(req: AuthRequest):
    conn = get_db_connection()
    if not conn: raise HTTPException(500)
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
    if not conn: return {}
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM products")
    rows = cur.fetchall()
    conn.close()
    themes = {}
    for row in rows:
        item = dict(row)
        try: item['examples'] = json.loads(item['examples'])
        except: item['examples'] = []
        icons = {"basic": "📚", "offline": "🚩"}
        item['icon'] = icons.get(item['category'], "🎤")
        themes[item['id']] = item
    return themes

@app.post("/book")
def create_booking(req: BookingRequest):
    conn = get_db_connection()
    if not conn: raise HTTPException(500)
    try:
        cur = conn.cursor()
        cur.execute("SELECT title FROM products WHERE id=%s", (req.theme_id,))
        res = cur.fetchone()
        title = res[0] if res else "Unknown"
        cur.execute("""
                    INSERT INTO bookings (username, theme_id, theme_title, reserved_date, people, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (req.username, req.theme_id, title, req.date, req.people, get_kst_now()))
        conn.commit()
        return {"status": "success"}
    except: raise HTTPException(500, "Fail")
    finally: conn.close()

@app.get("/bookings/{username}")
def get_my_bookings(username: str):
    conn = get_db_connection()
    if not conn: return []
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM bookings WHERE username=%s ORDER BY created_at DESC", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.post("/bookings/cancel")
def cancel_booking(req: CancelRequest):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM bookings WHERE id=%s", (req.booking_id,))
        conn.commit()
        return {"status": "success"}
    finally: conn.close()

@app.get("/reports/{username}")
def get_reports(username: str):
    conn = get_db_connection()
    if not conn: return []
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT theme_id, tech_score, created_at FROM speaking_logs WHERE username = %s ORDER BY created_at DESC LIMIT 7", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.post("/talk")
async def talk_to_ai(file: UploadFile = File(...), theme_id: str = Form(...), username: str = Form(...)):
    filename = file.filename
    user_path = f"temp_audio/in_{filename}"
    target_path = f"temp_audio/tgt_{filename}.mp3"

    try:
        with open(user_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

        # 1. STT
        with open(user_path, "rb") as af:
            transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=af)
        user_text = transcript.text
        if not user_text.strip(): return {"error": "No voice"}

        # 2. Persona
        conn = get_db_connection()
        persona, situation = "Tutor", "Practice"
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT persona, situation FROM products WHERE id=%s", (theme_id,))
            res = cur.fetchone()
            conn.close()
            if res: persona, situation = res

        # 3. LLM (한글/영어 동시 생성)
        SYSTEM_PROMPT = f"""
        Role: {persona} in {situation}. Task: Translate English to Korean.
        Output JSON: {{
            "korean": "Target Korean sentence",
            "english_meaning": "English Translation of the Korean sentence", 
            "romanized": "...",
            "grammar_kor": "문법설명(한글)",
            "grammar_eng": "Grammar(Eng)",
            "expl_kor": "상황설명(한글)",
            "expl_eng": "Context(Eng)"
        }}
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":f"User: '{user_text}'. Return JSON."}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        target_korean = data.get("korean", "다시 말해주세요.")

        # 4. Deep Tech
        tts_tgt = openai_client.audio.speech.create(model="tts-1", voice="nova", input=target_korean, speed=1.0)
        tts_tgt.stream_to_file(target_path)
        score = analyze_audio_similarity(user_path, target_path)
        data['tech_score'] = score

        # 5. DB Save (KST)
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO speaking_logs (username, theme_id, user_text, tech_score, feedback, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                        (username, theme_id, user_text, score, data.get('expl_eng', ''), get_kst_now()))
            conn.commit()
            conn.close()

        # 6. Audio
        full_text = f"{target_korean}. {data.get('expl_eng')}"
        tts_final = openai_client.audio.speech.create(model="tts-1", voice="nova", input=full_text, speed=1.0)
        audio_b64 = base64.b64encode(tts_final.content).decode('utf-8')

        return {
            "user_text": user_text,
            "structured_data": data,
            "audio_base64": audio_b64
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    finally:
        for p in [user_path, target_path]:
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
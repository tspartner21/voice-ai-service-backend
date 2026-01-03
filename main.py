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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# [DB 설정] 본인 환경에 맞게 수정
DB_HOST = "localhost"
DB_NAME = "quest_db"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_PORT = "5432"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/images", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_kst_now():
    return datetime.now(pytz.timezone('Asia/Seoul'))

def get_db_connection():
    try:
        return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
    except: return None

def init_db():
    conn = get_db_connection()
    if not conn: return
    cur = conn.cursor()

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

    cur.execute("INSERT INTO users (username, password, role, full_name) VALUES (%s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
                ('1111', '1111', 'user', 'Tester'))

    seed = [
        ("cafe", "basic", "☕ 스타벅스 주문", "Free", "", "사이렌오더 대신 말로 주문", "바리스타", "카페", '["아메리카노 톨 사이즈요"]'),
        ("subway", "basic", "🚇 지하철 환승", "Free", "", "길 잃었을 때 질문", "시민", "지하철역", '["강남역 가려면 어디로 가요?"]'),
        ("quest_tour", "offline", "🏯 경복궁 야간개장", "30,000", "", "한복 입고 인생샷", "가이드", "경복궁", '["사진 좀 찍어주세요"]'),
        ("oliveyoung", "basic", "💄 올리브영 쇼핑", "Free", "", "인기 틴트 추천받기", "직원", "매장", '["쿨톤 틴트 추천해주세요"]')
    ]
    for p in seed:
        cur.execute("INSERT INTO products (id, category, title, price, image_url, description, persona, situation, examples) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING", p)

    conn.commit()
    conn.close()

init_db()

class AuthRequest(BaseModel): username: str; password: str
class BookingRequest(BaseModel): username: str; theme_id: str; date: str; people: int
class CancelRequest(BaseModel): booking_id: int

def analyze_audio_similarity(user_path, target_path):
    try:
        y1, _ = librosa.load(user_path, sr=16000)
        y2, _ = librosa.load(target_path, sr=16000)
        mfcc1 = librosa.feature.mfcc(y=y1, sr=16000, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=16000, n_mfcc=13)
        dist, path = fastdtw(mfcc1.T, mfcc2.T, dist=cosine, radius=10)
        score = max(0, min(100, int((1 - (dist / len(path) / 0.6)) * 100)))
        if score > 60: score = min(100, score + 15)
        return score
    except: return 0

@app.post("/login")
def login(req: AuthRequest):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, role FROM users WHERE username=%s AND password=%s", (req.username, req.password))
    user = cur.fetchone()
    conn.close()
    if user: return {"status": "success", "username": user[0], "role": user[1]}
    raise HTTPException(401, "Login failed")

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
    cur = conn.cursor()
    cur.execute("SELECT title FROM products WHERE id=%s", (req.theme_id,))
    title = cur.fetchone()[0]
    cur.execute("INSERT INTO bookings (username, theme_id, theme_title, reserved_date, people, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (req.username, req.theme_id, title, req.date, req.people, get_kst_now()))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/bookings/{username}")
def get_bookings(username: str):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM bookings WHERE username=%s ORDER BY created_at DESC", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.post("/bookings/cancel")
def cancel_booking(req: CancelRequest):
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

@app.post("/talk")
async def talk(file: UploadFile = File(...), theme_id: str = Form(...), username: str = Form(...)):
    filename = file.filename
    user_path = f"temp_audio/in_{filename}"
    target_path = f"temp_audio/tgt_{filename}.mp3"

    try:
        with open(user_path, "wb") as b: shutil.copyfileobj(file.file, b)

        # STT
        with open(user_path, "rb") as af:
            transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=af)
        user_text = transcript.text
        if not user_text.strip(): return {"error": "No voice"}

        # Context
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT persona, situation FROM products WHERE id=%s", (theme_id,))
        res = cur.fetchone()
        conn.close()
        persona, situation = res if res else ("Tutor", "Practice")

        # LLM (Naver Style Output Request)
        sys_prompt = f"""Role: {persona} in {situation}. Task: Translate Eng to Kor.
        Output JSON: {{
            "korean": "Natural Korean",
            "romanized": "...",
            "english": "Meaning",
            "grammar_kor": "문법/표현 팁 (한글)",
            "grammar_eng": "Grammar Tip (Eng)",
            "expl_kor": "상황 설명 (한글)",
            "expl_eng": "Context (Eng)",
            "tip": "Better Expression Tip"
        }}"""

        gpt = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys_prompt}, {"role":"user","content":f"User said: {user_text}"}],
            response_format={"type": "json_object"}
        )
        data = json.loads(gpt.choices[0].message.content)
        target_korean = data.get("korean", "")

        # Deep Tech
        tts = openai_client.audio.speech.create(model="tts-1", voice="nova", input=target_korean, speed=1.0)
        tts.stream_to_file(target_path)
        score = analyze_audio_similarity(user_path, target_path)
        data['tech_score'] = score

        # Save
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO speaking_logs (username, theme_id, user_text, tech_score, feedback, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                    (username, theme_id, user_text, score, data.get('tip', ''), get_kst_now()))
        conn.commit()
        conn.close()

        # Audio
        full_text = f"{target_korean}. {data.get('expl_eng')}"
        tts_final = openai_client.audio.speech.create(model="tts-1", voice="nova", input=full_text, speed=1.0)
        audio_b64 = base64.b64encode(tts_final.content).decode('utf-8')

        return {"user_text": user_text, "structured_data": data, "audio_base64": audio_b64}

    except Exception as e:
        print(e)
        return {"error": str(e)}
    finally:
        for p in [user_path, target_path]:
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
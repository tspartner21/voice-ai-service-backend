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

# 1. 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# [중요] PostgreSQL 접속 정보 (본인 환경에 맞게 수정 필수)
DB_HOST = "localhost"
DB_NAME = "quest_db"
DB_USER = "postgres"
DB_PASSWORD = "1234"  # <--- 본인 비밀번호 입력
DB_PORT = "5432"

app = FastAPI()

# CORS 설정 (프론트엔드 포트 허용)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

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

# --- 2. DB 연결 및 초기화 ---
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return None

def init_db():
    print("🔄 DB 초기화 중...")
    conn = get_db_connection()
    if not conn:
        print("❌ DB 연결 불가. PostgreSQL이 켜져있는지, 'quest_db'가 존재하는지 확인하세요.")
        return

    try:
        cur = conn.cursor()

        # 테이블 생성
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                                                         username VARCHAR(50) PRIMARY KEY,
                        password VARCHAR(50) NOT NULL,
                        role VARCHAR(20) DEFAULT 'user',
                        full_name VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS products (
                                                            id VARCHAR(50) PRIMARY KEY,
                        category VARCHAR(20),
                        title VARCHAR(100),
                        price VARCHAR(50),
                        image_url TEXT,
                        description TEXT,
                        persona VARCHAR(50),
                        situation VARCHAR(50),
                        examples TEXT
                        );
                    """)
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS speaking_logs (
                                                                 id SERIAL PRIMARY KEY,
                                                                 username VARCHAR(50),
                        theme_id VARCHAR(50),
                        user_text TEXT,
                        tech_score INT,
                        feedback TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

        # 기초 데이터 삽입
        cur.execute("INSERT INTO users (username, password, role, full_name) VALUES (%s, %s, %s, %s) ON CONFLICT (username) DO NOTHING", ('admin', 'admin', 'admin', 'Admin'))
        cur.execute("INSERT INTO users (username, password, role, full_name) VALUES (%s, %s, %s, %s) ON CONFLICT (username) DO NOTHING", ('1111', '1111', 'user', 'Tester 1111'))

        seed_products = [
            ("kpop", "basic", "🎤 K-POP 콘서트", "Free", "", "콘서트장 상황극", "열정적인 MC", "콘서트장", '["Scream!", "Encore!"]'),
            ("store", "basic", "🏪 편의점 알바", "Free", "", "편의점 상황극", "친절한 알바생", "편의점", '["How much?", "I need a bag."]'),
            ("date", "basic", "💕 홍대 첫 데이트", "Free", "", "데이트 상황극", "설레는 상대방", "홍대 맛집", '["You look pretty.", "Lets eat."]'),
            ("offline_hongdae", "offline", "🔥 홍대 언어교환", "35,000원", "https://via.placeholder.com/400", "현지인 친구 사귀기", "모임장", "언어교환", '["Hello"]')
        ]
        for p in seed_products:
            cur.execute("""
                        INSERT INTO products (id, category, title, price, image_url, description, persona, situation, examples)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                        """, p)

        conn.commit()
        print("✅ PostgreSQL DB 준비 완료")
    except Exception as e:
        print(f"❌ DB Init Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

init_db()

# --- Models ---
class AuthRequest(BaseModel):
    username: str; password: str
class RegisterRequest(BaseModel):
    username: str; password: str; full_name: str

# --- 딥테크 알고리즘 (MFCC + DTW + Cosine) ---
def analyze_audio_similarity(user_path, target_path):
    print(f"📡 신호 분석 시작: {user_path}")
    try:
        y1, sr1 = librosa.load(user_path, sr=16000)
        y2, sr2 = librosa.load(target_path, sr=16000)
        y1, _ = librosa.effects.trim(y1)
        y2, _ = librosa.effects.trim(y2)

        mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

        # CMN 정규화 (음색 제거)
        mfcc1 -= (np.mean(mfcc1, axis=1, keepdims=True) + 1e-8)
        mfcc2 -= (np.mean(mfcc2, axis=1, keepdims=True) + 1e-8)

        dist, path = fastdtw(mfcc1.T, mfcc2.T, dist=cosine, radius=10)
        avg_dist = dist / len(path)

        # 점수 스케일링
        if avg_dist > 0.6: final_score = 10
        else: final_score = int((1 - (avg_dist / 0.6)) * 100)

        if final_score > 60: final_score = min(100, final_score + 15)

        print(f"✅ 최종 점수: {final_score}")
        return final_score
    except Exception as e:
        print(f"Algorithm Error: {e}")
        return 0

# --- API ---
@app.post("/login")
def login(req: AuthRequest):
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB Error")
    try:
        cur = conn.cursor()
        cur.execute("SELECT username, role FROM users WHERE username=%s AND password=%s", (req.username, req.password))
        user = cur.fetchone()
        conn.close()
        if user: return {"status": "success", "username": user[0], "role": user[1]}
        raise HTTPException(status_code=401, detail="로그인 실패: 아이디/비번을 확인하세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register")
def register(req: RegisterRequest):
    conn = get_db_connection()
    if not conn: raise HTTPException(status_code=500, detail="DB Error")
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password, full_name) VALUES (%s, %s, %s)", (req.username, req.password, req.full_name))
        conn.commit()
        return {"status": "success"}
    except:
        raise HTTPException(status_code=400, detail="이미 존재하는 ID입니다.")
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
        if item['category'] == 'basic': item['icon'] = "📚"
        themes[item['id']] = item
    return themes

@app.get("/reports/{username}")
def get_reports(username: str):
    conn = get_db_connection()
    if not conn: return []
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT theme_id, tech_score, created_at FROM speaking_logs WHERE username = %s ORDER BY created_at DESC LIMIT 20", (username,))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.post("/talk")
async def talk_to_ai(file: UploadFile = File(...), theme_id: str = Form(...), username: str = Form(...)):
    filename = file.filename
    # 확장자 유지
    user_path = f"temp_audio/in_{filename}"
    target_path = f"temp_audio/tgt_{filename}.mp3"

    try:
        with open(user_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

        # 1. STT
        with open(user_path, "rb") as af:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1", file=af,
                prompt="English conversation input." # 힌트 추가
            )
        user_text = transcript.text
        if not user_text.strip(): return {"error": "목소리가 너무 작거나 없습니다."}

        # 2. 페르소나
        conn = get_db_connection()
        persona, situation = "Tutor", "Practice"
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT persona, situation FROM products WHERE id=%s", (theme_id,))
            res = cur.fetchone()
            conn.close()
            if res: persona, situation = res

        # 3. LLM
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Role: {persona} in {situation}. Task: Translate English to Korean. Output JSON: {{'korean': '...', 'romanized': '...', 'english': '...', 'grammar': '...', 'expl': '...'}}"},
                {"role": "user", "content": f"User: {user_text}. Return JSON."}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        target_korean = data.get("korean", "다시 말해주세요.")

        # 4. 딥테크 (TTS 생성 및 비교)
        tts_tgt = openai_client.audio.speech.create(model="tts-1", voice="nova", input=target_korean, speed=1.0)
        tts_tgt.stream_to_file(target_path)

        score = analyze_audio_similarity(user_path, target_path)
        data['tech_score'] = score

        # 5. DB 저장
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO speaking_logs (username, theme_id, user_text, tech_score, feedback) VALUES (%s, %s, %s, %s, %s)",
                        (username, theme_id, user_text, score, data.get('expl', '')))
            conn.commit()
            conn.close()

        # 6. 결과 오디오
        full_text = f"{target_korean}. {data.get('expl')}"
        tts_final = openai_client.audio.speech.create(model="tts-1", voice="nova", input=full_text, speed=1.0)
        audio_b64 = base64.b64encode(tts_final.content).decode('utf-8')

        return {
            "user_text": user_text,
            "structured_data": data,
            "audio_base64": audio_b64
        }

    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}
    finally:
        for p in [user_path, target_path]:
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
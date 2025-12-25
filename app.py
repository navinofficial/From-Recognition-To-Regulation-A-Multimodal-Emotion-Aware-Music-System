# app.py
import os
import io
import re
import uuid
import time
import json
import base64
import requests
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# ML imports
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import SwinForImageClassification
from sentence_transformers import SentenceTransformer, util
from PIL import Image

# ---------- Spotify client id/secret (used only for read-only extraction) ----------
SPOTIFY_CLIENT_ID = "a75e5cbc067b478292fb29ed87943dba"
SPOTIFY_CLIENT_SECRET = "4f7809b3b5a141a8b1b3721865338fe2"

# ---------- Flask app ----------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///moodtune.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'super_secret_key'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ---------- DB Models ----------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    name = db.Column(db.String(150))
    password = db.Column(db.String(150), nullable=False)

class EmotionCapture(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    filename = db.Column(db.String(300))
    image_data = db.Column(db.LargeBinary)
    emotion = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    # Query.get() is legacy but works here
    return User.query.get(int(user_id))

# ---------- Emotion model setup ----------
MODEL_PATH = "models/face/best_swin_ck.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# load swin model (weights should be compatible)
model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=len(EMOTION_LABELS),
    ignore_mismatched_sizes=True
)

# fix: use map_location
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---------- SentenceTransformer for songs & title matching ----------
song_model = SentenceTransformer("all-MiniLM-L6-v2")

EMOTION_SENTENCES = {
    "happy": (
        "joyful uplifting cheerful positive upbeat energetic feel-good dance pop bright lively "
        "celebration playful sunshine happy vibes"
    ),

    "sad": (
        "emotional heartbreaking sorrow pain loneliness slow melodic soft soulful acoustic ballad "
        "broken heart sentimental sad vibes"
    ),

    "angry": (
        "intense aggressive powerful heavy loud strong fast dark rock metal explosive forceful "
        "anger tension high-energy"
    ),

    "calm": (
        "relaxing peaceful soothing mellow soft gentle chill ambient slow acoustic serene tranquil "
        "lofi sleep calm vibes"
    ),

    "energetic": (
        "high-energy fast powerful upbeat dance edm workout running motivation pump-up intense "
        "rhythmic exciting dynamic"
    ),
}


def ai_filter_songs(song_titles, target_emotion):
    """Rank songs by similarity to a target-emotion description."""
    if not song_titles:
        return []
    emotion_text = EMOTION_SENTENCES.get(target_emotion, target_emotion)
    emotion_emb = song_model.encode(emotion_text, convert_to_tensor=True)
    song_embs = song_model.encode(song_titles, convert_to_tensor=True)
    similarities = util.cos_sim(emotion_emb, song_embs)[0]
    ranked = sorted(zip(song_titles, similarities), key=lambda x: x[1], reverse=True)
    # return top 10 (or fewer)
    return [s for s, sc in ranked[:10]]

# ---------- YouTube search scraping helpers (no API) ----------
YOUTUBE_SEARCH_URL = "https://www.youtube.com/results"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}

def extract_video_candidates_from_html(html, limit=8):
    """
    Try to extract (video_id, title) pairs from YouTube search page HTML.
    Returns first `limit` unique candidates.
    """
    candidates = []
    seen = set()

    # Attempt 1: find videoRenderer JSON chunks in ytInitialData (robust)
    try:
        m = re.search(r"var ytInitialData = (.*?);</script>", html, re.S)
        if not m:
            m = re.search(r"ytInitialData\s*=\s*(\{.*?\})\s*;</script>", html, re.S)
        if m:
            data = json.loads(m.group(1))
            # navigate recursively to find videoRenderer items
            def find_video_renderers(node):
                if isinstance(node, dict):
                    if 'videoRenderer' in node:
                        vr = node['videoRenderer']
                        vid = vr.get('videoId')
                        title_obj = vr.get('title', {})
                        # title text can be nested
                        if isinstance(title_obj, dict):
                            runs = title_obj.get('runs') or []
                            title = " ".join([r.get('text','') for r in runs]) if runs else title_obj.get('simpleText','')
                        else:
                            title = str(title_obj)
                        return [(vid, title)]
                    out = []
                    for v in node.values():
                        out += find_video_renderers(v)
                    return out
                if isinstance(node, list):
                    out = []
                    for item in node:
                        out += find_video_renderers(item)
                    return out
                return []
            found = find_video_renderers(data)
            for vid, title in found:
                if vid and vid not in seen:
                    seen.add(vid)
                    candidates.append((vid, title or ""))
                    if len(candidates) >= limit:
                        return candidates
    except Exception:
        pass

    # Fallback: simple href/title regex (less robust)
    for match in re.finditer(r'<a[^>]+href="(/watch\?v=([^"&]+)[^"]*)"[^>]*>(.*?)</a>', html, re.S):
        vid = match.group(2)
        inner = re.sub('<.*?>', '', match.group(3)).strip()
        title = inner or ""
        if vid and vid not in seen and "list=" not in match.group(1):
            seen.add(vid)
            candidates.append((vid, title))
            if len(candidates) >= limit:
                break

    return candidates

def search_youtube_candidates(song_title, max_candidates=6, pause=0.35):
    """
    Return candidate (video_id, title) list for a song_title by scraping YouTube search.
    """
    params = {"search_query": song_title + " official audio"}
    try:
        r = requests.get(YOUTUBE_SEARCH_URL, params=params, headers=HEADERS, timeout=8)
        html = r.text
        candidates = extract_video_candidates_from_html(html, limit=max_candidates)
        # if not enough candidates, try a simpler search (without 'official audio')
        if len(candidates) < max_candidates:
            r2 = requests.get(YOUTUBE_SEARCH_URL, params={"search_query": song_title}, headers=HEADERS, timeout=8)
            html2 = r2.text
            more = extract_video_candidates_from_html(html2, limit=max_candidates)
            for vid, t in more:
                if vid not in [v for v, _ in candidates]:
                    candidates.append((vid, t))
                    if len(candidates) >= max_candidates:
                        break
        # small pause to be polite
        time.sleep(pause)
        return candidates
    except Exception:
        return []

def choose_best_youtube_video(song_title, candidates):
    """
    Uses embedding similarity between song_title and each youtube candidate title
    to select the best match. Returns best video_id or None.
    """
    if not candidates:
        return None
    titles = [c[1] or "" for c in candidates]
    # embed song title and candidates
    emb_song = song_model.encode(song_title, convert_to_tensor=True)
    emb_titles = song_model.encode(titles, convert_to_tensor=True)
    sims = util.cos_sim(emb_song, emb_titles)[0].cpu().numpy()
    best_idx = int(sims.argmax())
    best_vid = candidates[best_idx][0]
    return best_vid

def build_youtube_autoplay_link(video_ids):
    """
    Given list of YouTube video ids, build watch_videos autoplay URL.
    """
    ids = [vid for vid in video_ids if vid]
    if not ids:
        return None
    joined = ",".join(ids)
    return f"https://www.youtube.com/watch_videos?video_ids={joined}"

# ---------- Routes & logic ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    # basic login/signup used in your project — keep as-is
    if request.method == "POST":
        action = request.form.get("action")
        if action == "signup":
            name = request.form.get("name")
            email = request.form.get("email")
            password = request.form.get("password")
            confirm = request.form.get("confirm_password")
            if User.query.filter_by(email=email).first():
                flash("Email already registered!")
                return redirect(url_for('login'))
            if password != confirm:
                flash("Passwords do not match!")
                return redirect(url_for('login'))
            new_user = User(name=name, email=email, password=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            flash("Account created! Please login.")
            return redirect(url_for('login'))
        if action == "login":
            email = request.form.get("email")
            password = request.form.get("password")
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('home'))
            flash("Invalid email or password.")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('home.html', name=current_user.name)

@app.route('/input')
@login_required
def input_page():
    return render_template('input.html')

@app.route('/output')
@login_required
def output_page():
    image_url = request.args.get("image_url")
    emotion = request.args.get("emotion")
    confidence = request.args.get("confidence")

    return render_template(
        "output.html",
        image_url=image_url,
        emotion=emotion,
        confidence=float(confidence)
    )

@app.route('/predict_emotion', methods=['POST'])
@login_required
def predict_emotion():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image"}), 400
    base64_str = data["image"].split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    filename = f"{uuid.uuid4().hex}.jpg"
    os.makedirs("static/captured", exist_ok=True)
    filepath = os.path.join("static/captured", filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inp = inference_transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inp)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())
    emotion = EMOTION_LABELS[top_idx]
    confidence = float(probs[top_idx])
    # save capture (optional)
    rec = EmotionCapture(user_id=current_user.id, filename=filename, image_data=image_bytes, emotion=emotion, confidence=confidence)
    db.session.add(rec)
    db.session.commit()
    return jsonify({"emotion": emotion, "confidence": confidence, "image_url": url_for('static', filename=f"captured/{filename}")})

@app.route('/process_playlist', methods=['POST'])
@login_required
def process_playlist():
    """
    Expects JSON:
    { playlist_url: "...", mode: "match"|"regulate", emotion: "happy" }
    """
    data = request.get_json()
    playlist_url = data.get("playlist_url")
    mode = data.get("mode", "match")
    emotion = data.get("emotion", "neutral")

    if not playlist_url:
        return jsonify({"error": "Missing playlist_url"}), 400

    songs = []

    # Extract from YouTube or Spotify (you already had this logic)
    try:
        if "youtube.com/playlist" in playlist_url or "youtu.be" in playlist_url:
            # Pytube playlist extraction (fast)
            from pytube import Playlist
            pl = Playlist(playlist_url)
            songs = [v.title for v in pl.videos]
        elif "spotify.com/playlist" in playlist_url:
            # use spotipy client-credentials (read-only extraction)
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            # pass client id/secret explicitly
            creds = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
            sp = spotipy.Spotify(auth_manager=creds)
            pl_id = playlist_url.split("/")[-1].split("?")[0]

            # fetch all tracks with pagination
            offset = 0
            limit = 100
            while True:
                results = sp.playlist_items(pl_id, offset=offset, limit=limit)
                items = results.get("items", [])
                if not items:
                    break
                for item in items:
                    t = item.get("track")
                    if t:
                        artists = t.get("artists", [])
                        artist_name = artists[0]["name"] if artists else ""
                        songs.append(f"{t.get('name')} - {artist_name}")
                if results.get("next"):
                    offset += limit
                else:
                    break

        else:
            return jsonify({"error": "Unsupported playlist URL"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to extract songs: {e}"}), 500

    # select songs using AI filter
    if mode == "match":
        selected_songs = ai_filter_songs(songs, emotion)
    else:
        # regulate logic (example)
        if emotion in ["sad", "angry", "fear", "disgust"]:
            selected_songs = ai_filter_songs(songs, "calm")
        elif emotion == "neutral":
            selected_songs = ai_filter_songs(songs, "happy")
        elif emotion == "surprise":
            selected_songs = ai_filter_songs(songs, "energetic")
        else:
            selected_songs = ai_filter_songs(songs, "calm")

    # Save session file of selected songs
    session_id = str(uuid.uuid4())
    os.makedirs("static/sessions", exist_ok=True)
    session_path = f"static/sessions/{session_id}.txt"
    with open(session_path, "w", encoding="utf-8") as f:
        for s in selected_songs:
            f.write(s + "\n")

    # For each selected song, find best YouTube video via scraping + embedding match
    chosen_ids = []
    for s in selected_songs:
        try:
            candidates = search_youtube_candidates(s, max_candidates=6)
            best_vid = choose_best_youtube_video(s, candidates)
            if best_vid:
                chosen_ids.append(best_vid)
            else:
                # fallback: do a quick search, take first id
                if candidates:
                    chosen_ids.append(candidates[0][0])
        except Exception:
            # ignore and continue
            continue

    youtube_playlist_url = build_youtube_autoplay_link(chosen_ids)

    # return session id + playlist url
    return jsonify({
        "session_id": session_id,
        "playlist_url": youtube_playlist_url,
        "songs": selected_songs
    })


@app.route('/playlist_result')
@login_required
def playlist_result():
    session_id = request.args.get("session_id")
    playlist_url = request.args.get("playlist_url")
    if not session_id:
        return "Missing session_id", 400
    session_path = f"static/sessions/{session_id}.txt"
    if not os.path.exists(session_path):
        return "Session expired or not found.", 404
    with open(session_path, "r", encoding="utf-8") as f:
        songs = [line.strip() for line in f.readlines() if line.strip()]
    return render_template("playlist_result.html", songs=songs, playlist_url=playlist_url)


if __name__ == "__main__":
    app.run(debug=True)

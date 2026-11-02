"""
AI HILL CLIMB CONTROLLER  -  game + hand gesture control in ONE app
-------------------------------------------------------------------
Left side : Hill Climb Racing style game (fullscreen)
Right side: your webcam with hand landmarks

  Open palm   -> GAS
  Closed fist -> BRAKE / REVERSE
  No hand     -> coast (nothing pressed)

Keyboard backup: RIGHT/D = gas, LEFT/A = brake, R = restart,
F = fullscreen on/off, ESC = quit.
"""

import math
import random
import sys
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pygame
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ======================================================================
# Settings
# ======================================================================
CAMERA_INDEX = 0            # try 1 if you have several cameras
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BUFFER_SIZE = 3             # frames that must agree before a gesture counts
MAX_FAILED_READS = 10
STALE_SECONDS = 0.7         # no fresh camera data for this long -> treat as no hand

EXTEND_RATIO = 1.15         # lower (1.10) if open palm is not detected
FIST_TIP_RATIO = 1.10       # raise (1.25) if fist is not detected
MIN_HAND_SIZE_PX = 30

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "assets" / "hand_landmarker.task"

# ---- Game physics (units: metres, seconds; car mass = 1) ----
GRAVITY = 9.8
MASS = 1.0
INERTIA = 1.0
SPRING_K = 200.0
SPRING_C = 10.0
WHEEL_R = 0.35
DRIVE_FORCE = 5.5           # engine push per wheel (raise = faster car)
MAX_SPEED = 18.0            # m/s, engine stops pushing above this
BRAKE_GAIN = 4.0
BRAKE_MAX = 6.0
REVERSE_SPEED = -3.0
ROLL_RESIST = 0.08
FRICTION_MU = 1.5
AIR_TORQUE = 3.0            # tilt strength in the air
ANG_DAMP = 0.8
AIR_DRAG = 0.03

# Contact points on the car body: (local x, local y, radius, is_wheel)
CONTACTS = [
    (-0.9, -0.25, 0.35, True),    # rear wheel
    (0.9, -0.25, 0.35, True),     # front wheel
    (-1.25, -0.15, 0.0, False),   # rear bumper
    (1.25, -0.15, 0.0, False),    # front bumper
    (-0.5, 0.75, 0.0, False),     # roof back
    (0.25, 0.75, 0.0, False),     # roof front
]
HEAD_LOCAL = (0.0, 0.98)
HEAD_R = 0.2

# ---- Colours (RGB) ----
GREEN = (60, 220, 90)
RED = (235, 60, 60)
YELLOW = (250, 210, 50)
WHITE = (240, 240, 240)

OPEN_PALM = "OPEN_PALM"
CLOSED_FIST = "CLOSED_FIST"
UNCERTAIN = "UNCERTAIN"
NO_HAND = "NO_HAND"

GESTURE_LABELS = {
    OPEN_PALM: "ACCELERATING",
    CLOSED_FIST: "BRAKING",
    UNCERTAIN: "IDLE",
    NO_HAND: "NO HAND DETECTED",
}
LABEL_COLORS = {
    "ACCELERATING": GREEN,
    "BRAKING": RED,
    "IDLE": YELLOW,
    "NO HAND DETECTED": WHITE,
}

FINGER_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]
FINGER_TIPS = [8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


# ======================================================================
# Hand model download
# ======================================================================
def ensure_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading hand model (one time only)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise RuntimeError(
            f"Could not download the model: {exc}. Download it manually from "
            f"{MODEL_URL} and save as {MODEL_PATH}"
        ) from exc
    return MODEL_PATH


# ======================================================================
# Gesture classification (pure geometry)
# ======================================================================
def landmarks_to_points(landmarks, width, height) -> np.ndarray:
    return np.array([[lm.x * width, lm.y * height] for lm in landmarks],
                    dtype=np.float32)


def dist(a, b) -> float:
    return float(np.linalg.norm(a - b))


def classify_gesture(pts: np.ndarray) -> str:
    wrist = pts[0]
    hand_size = dist(wrist, pts[9])
    if hand_size < MIN_HAND_SIZE_PX:
        return UNCERTAIN

    extended = 0
    for tip, pip in FINGER_TIP_PIP:
        if dist(pts[tip], wrist) > dist(pts[pip], wrist) * EXTEND_RATIO:
            extended += 1
    if dist(pts[4], pts[17]) > dist(pts[3], pts[17]) * EXTEND_RATIO:
        extended += 1

    if extended >= 4:
        return OPEN_PALM

    palm_center = pts[PALM_POINTS].mean(axis=0)
    tip_d = [dist(pts[t], palm_center) / hand_size for t in FINGER_TIPS]
    if extended <= 1 and float(np.mean(tip_d)) < FIST_TIP_RATIO:
        return CLOSED_FIST
    return UNCERTAIN


class GestureBuffer:
    """Reports a gesture only if the last N frames all agree."""

    def __init__(self, size):
        self.size = size
        self.items = deque(maxlen=size)

    def update(self, gesture):
        self.items.append(gesture)
        if len(self.items) == self.size and len(set(self.items)) == 1:
            return gesture
        return UNCERTAIN

    def reset(self):
        self.items.clear()


def draw_hand(rgb, pts, color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(rgb, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                 color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(rgb, tuple(p.astype(int)), 4, (255, 255, 255), -1, cv2.LINE_AA)


# ======================================================================
# Webcam + MediaPipe in a background thread (keeps the game smooth)
# ======================================================================
class HandTracker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame = None
        self._gesture = NO_HAND
        self._tracking = False
        self._fps = 0.0
        self._stamp = 0.0
        self.status = "Starting..."
        self.error = None

    def _publish(self, frame, gesture, tracking, fps):
        with self._lock:
            self._frame = frame
            self._gesture = gesture
            self._tracking = tracking
            self._fps = fps
            self._stamp = time.monotonic()

    def get(self):
        """Returns (rgb_frame, gesture, tracking, fps). Old data counts as no hand."""
        with self._lock:
            fresh = (time.monotonic() - self._stamp) < STALE_SECONDS
            gesture = self._gesture if fresh else NO_HAND
            tracking = self._tracking and fresh
            return self._frame, gesture, tracking, self._fps

    def run(self):
        cap = None
        try:
            self.status = "Loading hand model..."
            model_bytes = ensure_model().read_bytes()

            self.status = "Opening webcam..."
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError("Cannot open webcam (close other apps using it, "
                                   "or change CAMERA_INDEX).")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
            )

            buf = GestureBuffer(BUFFER_SIZE)
            last_ts = 0
            fps = 0.0
            prev = time.perf_counter()
            fails = 0
            self.status = "Running"

            with vision.HandLandmarker.create_from_options(options) as landmarker:
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        fails += 1
                        self._publish(None, NO_HAND, False, fps)
                        if fails >= MAX_FAILED_READS:
                            raise RuntimeError("Webcam stopped working.")
                        time.sleep(0.05)
                        continue
                    fails = 0

                    frame = cv2.flip(frame, 1)                 # mirror
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts = int(time.monotonic() * 1000)
                    if ts <= last_ts:
                        ts = last_ts + 1
                    last_ts = ts
                    result = landmarker.detect_for_video(mp_image, ts)

                    if result.hand_landmarks:
                        pts = landmarks_to_points(result.hand_landmarks[0], w, h)
                        stable = buf.update(classify_gesture(pts))
                        color = LABEL_COLORS[GESTURE_LABELS[stable]]
                        draw_hand(rgb, pts, color)
                        tracking = True
                    else:
                        buf.reset()
                        stable = NO_HAND
                        tracking = False

                    now = time.perf_counter()
                    inst = 1.0 / max(now - prev, 1e-6)
                    prev = now
                    fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                    self._publish(rgb, stable, tracking, fps)
        except Exception as exc:
            self.error = str(exc)
            print(f"[camera error] {exc}")
        finally:
            self._publish(None, NO_HAND, False, 0.0)
            if cap is not None:
                cap.release()


# ======================================================================
# Terrain (endless hills, harder as you go further)
# ======================================================================
def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ground_y(x):
    f = smoothstep(x / 40.0)                       # flat start
    a = 0.75 + min(max(x, 0.0) / 500.0, 0.5)       # hills grow with distance
    h = (a * (0.9 * math.sin(0.22 * x)
              + 0.5 * math.sin(0.47 * x + 1.3)
              + 0.25 * math.sin(0.93 * x + 2.1))
         + 2.5 * math.sin(0.05 * x))
    return f * h


def ground_slope(x):
    return (ground_y(x + 0.05) - ground_y(x - 0.05)) / 0.1


def build_world():
    """Coins and fuel cans are placed with a fixed seed (same track every time)."""
    rng = random.Random(12345)
    coins = []
    x = 18.0
    while x < 6000:
        for i in range(rng.randint(4, 7)):
            cx = x + i * 1.6
            coins.append((cx, ground_y(cx) + 1.3 + 0.3 * math.sin(i)))
        x += rng.uniform(30, 55)
    cans = []
    x = 140.0
    while x < 6000:
        cx = x + rng.uniform(-15, 15)
        cans.append((cx, ground_y(cx) + 0.8))
        x += rng.uniform(140, 190)
    return coins, cans


# ======================================================================
# Game state + physics
# ======================================================================
class Game:
    def __init__(self):
        self.coins, self.cans = build_world()
        self.best = 0.0
        self.reset()

    def reset(self):
        self.x = 0.0
        self.y = ground_y(0.0) + 0.75
        self.vx = 0.0
        self.vy = 0.0
        self.angle = 0.0
        self.omega = 0.0
        self.wheel_angle = 0.0
        self.fuel = 1.0
        self.coin_count = 0
        self.coin_taken = [False] * len(self.coins)
        self.can_taken = [False] * len(self.cans)
        self.started = False
        self.over = False
        self.reason = ""
        self.stalled = 0.0
        self.grounded = 0

    def to_world(self, px, py):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return self.x + c * px - s * py, self.y + s * px + c * py

    def finish(self, reason):
        self.over = True
        self.reason = reason
        self.best = max(self.best, self.x)

    def _substep(self, dt, gas, brake):
        c, s = math.cos(self.angle), math.sin(self.angle)
        fx = -AIR_DRAG * self.vx
        fy = -GRAVITY * MASS - AIR_DRAG * self.vy
        tau = 0.0
        grounded = 0

        for lx, ly, rad, is_wheel in CONTACTS:
            rx = c * lx - s * ly
            ry = s * lx + c * ly
            wx = self.x + rx
            wy = self.y + ry
            pen = ground_y(wx) + rad - wy
            if pen <= 0.0:
                continue

            slope = ground_slope(wx)
            inv = 1.0 / math.sqrt(1.0 + slope * slope)
            nx, ny = -slope * inv, inv            # ground normal
            tx, ty = ny, -nx                      # ground tangent (forward)

            vpx = self.vx - self.omega * ry       # velocity of this contact point
            vpy = self.vy + self.omega * rx
            vn = vpx * nx + vpy * ny
            vt = vpx * tx + vpy * ty

            fn = SPRING_K * pen - SPRING_C * vn   # suspension push
            if fn <= 0.0:
                continue

            if is_wheel:
                grounded += 1
                if gas:
                    ft = DRIVE_FORCE * max(0.0, 1.0 - vt / MAX_SPEED)
                elif brake:
                    ft = BRAKE_GAIN * (REVERSE_SPEED - vt)
                    ft = max(-BRAKE_MAX, min(BRAKE_MAX, ft))
                else:
                    ft = -ROLL_RESIST * vt
            else:
                ft = -3.0 * vt
            lim = FRICTION_MU * fn
            ft = max(-lim, min(lim, ft))

            cfx = fn * nx + ft * tx
            cfy = fn * ny + ft * ty
            fx += cfx
            fy += cfy
            tau += rx * cfy - ry * cfx

        if grounded == 0:                          # tilt control in the air
            tau += AIR_TORQUE * ((1.0 if gas else 0.0) - (1.0 if brake else 0.0))

        self.vx += fx / MASS * dt
        self.vy += fy / MASS * dt
        self.omega += (tau / INERTIA - ANG_DAMP * self.omega) * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.omega * dt
        self.grounded = grounded

    def update(self, frame_dt, gas, brake):
        if self.over:
            return
        if gas and not self.started:
            self.started = True

        n = max(1, int(math.ceil(frame_dt * 240.0)))
        dt = frame_dt / n
        for _ in range(n):
            self._substep(dt, gas, brake)

        if self.x < -8.0:                          # invisible wall at the start
            self.x = -8.0
            self.vx = max(self.vx, 0.0)

        values = (self.x, self.y, self.vx, self.vy, self.angle, self.omega)
        if not all(math.isfinite(v) for v in values):
            self.reset()
            return
        self.vx = max(-60.0, min(60.0, self.vx))
        self.vy = max(-60.0, min(60.0, self.vy))
        self.omega = max(-15.0, min(15.0, self.omega))
        gy = ground_y(self.x)
        if self.y < gy - 3.0:                      # safety: never fall through the ground
            self.y = gy + 1.0
            self.vy = 0.0

        forward_speed = self.vx * math.cos(self.angle) + self.vy * math.sin(self.angle)
        self.wheel_angle += forward_speed / WHEEL_R * frame_dt

        if self.started:
            self.fuel -= (0.010 + (0.014 if gas else 0.0)) * frame_dt
            self.fuel = max(0.0, self.fuel)

        for i, (cx, cy) in enumerate(self.coins):
            if not self.coin_taken[i] and abs(cx - self.x) < 1.5:
                if math.hypot(cx - self.x, cy - self.y) < 1.3:
                    self.coin_taken[i] = True
                    self.coin_count += 1
        for i, (cx, cy) in enumerate(self.cans):
            if not self.can_taken[i] and abs(cx - self.x) < 2.0:
                if math.hypot(cx - self.x, cy - self.y) < 1.6:
                    self.can_taken[i] = True
                    self.fuel = min(1.0, self.fuel + 0.6)

        hx, hy = self.to_world(*HEAD_LOCAL)        # head touches ground = crash
        if hy - ground_y(hx) < HEAD_R:
            self.finish("CRASHED!")
            return

        if self.fuel <= 0.0 and math.hypot(self.vx, self.vy) < 0.5:
            self.stalled += frame_dt
            if self.stalled > 1.0:
                self.finish("OUT OF FUEL!")
        else:
            self.stalled = 0.0
        self.best = max(self.best, self.x)


# ======================================================================
# Drawing
# ======================================================================
class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.W, self.H = screen.get_size()
        self.game_w = int(self.W * 0.70)
        self.panel_w = self.W - self.game_w
        self.ppm = self.H / 13.0                   # pixels per metre
        self.sc = self.H / 1080.0
        self.car_sx = int(self.game_w * 0.30)
        self.cam_sy = int(self.H * 0.62)
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.game_surf = screen.subsurface(pygame.Rect(0, 0, self.game_w, self.H))
        self.sky = self._make_sky()

        def font(size):
            return pygame.font.SysFont("arial", max(12, int(size * self.sc)), bold=True)
        self.f_title = font(30)
        self.f_big = font(80)
        self.f_mid = font(40)
        self.f_hud = font(34)
        self.f_small = font(26)
        self.f_tiny = font(22)

    # ---------- helpers ----------
    def sx(self, wx):
        return self.car_sx + (wx - self.cam_x) * self.ppm

    def sy(self, wy):
        return self.cam_sy - (wy - self.cam_y) * self.ppm

    def text(self, surf, s, font, color, x, y, center=False):
        shadow = font.render(s, True, (0, 0, 0))
        img = font.render(s, True, color)
        rect = img.get_rect()
        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        surf.blit(shadow, rect.move(2, 2))
        surf.blit(img, rect)
        return rect

    def _make_sky(self):
        sky = pygame.Surface((self.game_w, self.H))
        top, bot = (70, 150, 235), (190, 228, 255)
        for y in range(self.H):
            t = y / self.H
            col = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
            pygame.draw.line(sky, col, (0, y), (self.game_w, y))
        pygame.draw.circle(sky, (255, 244, 170),
                           (int(self.game_w * 0.82), int(self.H * 0.17)),
                           int(60 * self.sc))
        return sky

    def snap(self, game):
        self.cam_x = game.x
        self.cam_y = game.y

    def follow(self, game, dt):
        self.cam_x = game.x
        self.cam_y += (game.y - self.cam_y) * min(1.0, 5.0 * dt)

    # ---------- game scene ----------
    def draw_game(self, g, fps):
        s = self.game_surf
        s.blit(self.sky, (0, 0))
        self._mountains()
        self._terrain()
        self._pickups(g)
        self._car(g)
        self._hud(g, fps)
        if not g.started and not g.over:
            cx = self.game_w // 2
            self.text(s, "SHOW AN OPEN PALM TO START", self.f_mid, WHITE,
                      cx, int(self.H * 0.22), center=True)
            self.text(s, "(or press the RIGHT ARROW key)", self.f_small, WHITE,
                      cx, int(self.H * 0.27), center=True)

    def _mountains(self):
        s = self.game_surf
        layers = ((0.55, 90, 0.15, (130, 165, 205)),
                  (0.66, 70, 0.30, (95, 145, 115)))
        for base, amp, speed, col in layers:
            off = self.cam_x * self.ppm * speed
            pts = []
            for px in range(0, self.game_w + 16, 16):
                y = (self.H * base
                     - amp * self.sc * (0.5 + 0.5 * math.sin((px + off) * 0.004))
                     - 40 * self.sc * math.sin((px + off) * 0.011 + 1.0))
                pts.append((px, y))
            pygame.draw.polygon(s, col, pts + [(self.game_w + 16, self.H), (0, self.H)])

    def _terrain(self):
        s = self.game_surf
        step = 8
        pts = []
        for px in range(0, self.game_w + step, step):
            wx = self.cam_x + (px - self.car_sx) / self.ppm
            pts.append((px, self.sy(ground_y(wx))))
        bottom = [(self.game_w + step, self.H), (0, self.H)]
        pygame.draw.polygon(s, (125, 78, 40), pts + bottom)
        deep = [(px, py + 1.6 * self.ppm) for px, py in pts]
        pygame.draw.polygon(s, (95, 58, 30), deep + bottom)
        pygame.draw.lines(s, (70, 170, 60), False, pts, max(6, int(14 * self.sc)))
        pygame.draw.lines(s, (110, 210, 80), False, pts, max(2, int(5 * self.sc)))

    def _pickups(self, g):
        s = self.game_surf
        r = max(5, int(0.28 * self.ppm))
        for i, (cx, cy) in enumerate(g.coins):
            if g.coin_taken[i]:
                continue
            px = self.sx(cx)
            if -30 < px < self.game_w + 30:
                p = (int(px), int(self.sy(cy)))
                pygame.draw.circle(s, (255, 205, 40), p, r)
                pygame.draw.circle(s, (200, 140, 10), p, r, max(2, r // 4))
        w, h = int(0.6 * self.ppm), int(0.8 * self.ppm)
        for i, (cx, cy) in enumerate(g.cans):
            if g.can_taken[i]:
                continue
            px = self.sx(cx)
            if -60 < px < self.game_w + 60:
                rect = pygame.Rect(0, 0, w, h)
                rect.center = (int(px), int(self.sy(cy)))
                pygame.draw.rect(s, (210, 40, 40), rect, border_radius=6)
                pygame.draw.rect(s, (120, 10, 10), rect, 3, border_radius=6)
                label = self.f_tiny.render("F", True, WHITE)
                s.blit(label, label.get_rect(center=rect.center))

    def _car(self, g):
        s = self.game_surf
        ppm = self.ppm

        def P(px, py):
            wx, wy = g.to_world(px, py)
            return (self.sx(wx), self.sy(wy))

        # wheels (drawn first, body overlaps them)
        r = WHEEL_R * ppm
        for wx_local in (-0.9, 0.9):
            cx, cy = P(wx_local, -0.25)
            pygame.draw.circle(s, (25, 25, 25), (cx, cy), r)
            pygame.draw.circle(s, (150, 150, 150), (cx, cy), r * 0.55)
            for k in range(4):
                a = g.wheel_angle + k * math.pi / 2
                pygame.draw.line(s, WHITE, (cx, cy),
                                 (cx + r * 0.55 * math.cos(a), cy + r * 0.55 * math.sin(a)), 3)

        body = [(-1.25, -0.15), (1.25, -0.15), (1.3, 0.15), (0.8, 0.3), (0.4, 0.3),
                (0.2, 0.78), (-0.55, 0.78), (-0.75, 0.3), (-1.3, 0.3)]
        poly = [P(px, py) for px, py in body]
        pygame.draw.polygon(s, (215, 50, 45), poly)
        pygame.draw.polygon(s, (110, 20, 20), poly, max(2, int(4 * self.sc)))

        win = [P(-0.45, 0.35), P(0.3, 0.35), P(0.15, 0.7), P(-0.45, 0.7)]
        pygame.draw.polygon(s, (170, 220, 255), win)

        hx, hy = P(*HEAD_LOCAL)
        pygame.draw.circle(s, (255, 213, 170), (hx, hy), HEAD_R * ppm)
        pygame.draw.circle(s, (40, 90, 200), (hx, hy), HEAD_R * ppm, max(3, int(6 * self.sc)))

    def _hud(self, g, fps):
        s = self.game_surf
        m = int(24 * self.sc)
        self.text(s, f"{max(g.x, 0):.0f} m", self.f_mid, WHITE, m, m)
        self.text(s, f"Coins: {g.coin_count}", self.f_hud, (255, 215, 60), m, m + int(55 * self.sc))
        self.text(s, f"Best: {g.best:.0f} m", self.f_small, WHITE, m, m + int(100 * self.sc))
        speed = math.hypot(g.vx, g.vy) * 3.6
        self.text(s, f"{speed:.0f} km/h", self.f_small, WHITE, m, m + int(135 * self.sc))

        bw, bh = int(360 * self.sc), int(26 * self.sc)
        bx = self.game_w - bw - m
        pygame.draw.rect(s, (30, 30, 30), (bx - 4, m - 4, bw + 8, bh + 8), border_radius=8)
        col = (int(255 * (1 - g.fuel)), int(220 * g.fuel + 35), 40)
        pygame.draw.rect(s, col, (bx, m, int(bw * g.fuel), bh), border_radius=6)
        self.text(s, "FUEL", self.f_tiny, WHITE, bx, m + bh + int(6 * self.sc))
        self.text(s, f"FPS {fps:.0f}", self.f_tiny, WHITE, bx + bw - int(90 * self.sc),
                  m + bh + int(6 * self.sc))

    def draw_game_over(self, g, progress):
        s = self.game_surf
        veil = pygame.Surface((self.game_w, self.H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 150))
        s.blit(veil, (0, 0))
        cx = self.game_w // 2
        self.text(s, g.reason, self.f_big, (255, 90, 90), cx, int(self.H * 0.28), center=True)
        self.text(s, f"Distance: {max(g.x, 0):.0f} m     Coins: {g.coin_count}",
                  self.f_mid, WHITE, cx, int(self.H * 0.42), center=True)
        self.text(s, f"Best: {g.best:.0f} m", self.f_hud, YELLOW, cx,
                  int(self.H * 0.50), center=True)
        self.text(s, "Hold an OPEN PALM for 1.5 s to play again  (or press R)",
                  self.f_small, WHITE, cx, int(self.H * 0.60), center=True)
        bw, bh = int(420 * self.sc), int(22 * self.sc)
        bx, by = cx - bw // 2, int(self.H * 0.65)
        pygame.draw.rect(s, (50, 50, 50), (bx, by, bw, bh), border_radius=8)
        pygame.draw.rect(s, GREEN, (bx, by, int(bw * progress), bh), border_radius=8)

    # ---------- camera panel ----------
    def draw_panel(self, frame, tracking, cam_fps, status, error, gesture, game_fps):
        scr = self.screen
        x0 = self.game_w
        m = int(24 * self.sc)
        pygame.draw.rect(scr, (20, 24, 33), (x0, 0, self.panel_w, self.H))
        pygame.draw.line(scr, (70, 80, 100), (x0, 0), (x0, self.H), 3)

        self.text(scr, "AI HILL CLIMB CONTROLLER", self.f_title, (255, 220, 0), x0 + m, m)

        label = GESTURE_LABELS[gesture]
        color = LABEL_COLORS[label]
        if error:
            label, color = "CAMERA ERROR", RED

        cw = self.panel_w - 2 * m
        ch = int(cw * 0.75)
        cy = m + int(60 * self.sc)
        box = pygame.Rect(x0 + m, cy, cw, ch)
        if frame is not None:
            fh, fw = frame.shape[:2]
            ch = int(cw * fh / fw)
            box = pygame.Rect(x0 + m, cy, cw, ch)
            surf = pygame.image.frombuffer(frame.tobytes(), (fw, fh), "RGB")
            scr.blit(pygame.transform.smoothscale(surf, (cw, ch)), box.topleft)
        else:
            pygame.draw.rect(scr, (10, 12, 18), box)
            msg = error if error else status
            self.text(scr, msg[:42], self.f_tiny, WHITE, box.centerx, box.centery, center=True)
        pygame.draw.rect(scr, color, box, max(3, int(5 * self.sc)))

        y = box.bottom + int(18 * self.sc)
        self.text(scr, label, self.f_mid, color, x0 + m, y)
        y += int(64 * self.sc)
        lines = [
            f"Hand tracking: {'ACTIVE' if tracking else 'LOST'}",
            f"Gesture: {gesture}",
            f"Camera FPS: {cam_fps:.0f}    Game FPS: {game_fps:.0f}",
        ]
        for line in lines:
            self.text(scr, line, self.f_small, WHITE, x0 + m, y)
            y += int(36 * self.sc)

        y += int(14 * self.sc)
        how = [
            ("HOW TO PLAY", YELLOW),
            ("Open palm   =  GAS", GREEN),
            ("Closed fist  =  BRAKE / REVERSE", RED),
            ("No hand      =  coast", WHITE),
            ("Collect coins and fuel cans!", WHITE),
            ("Keys: arrows / A D   R restart", (170, 170, 170)),
            ("F fullscreen     ESC quit", (170, 170, 170)),
        ]
        for line, col in how:
            self.text(scr, line, self.f_tiny, col, x0 + m, y)
            y += int(32 * self.sc)


# ======================================================================
# Main
# ======================================================================
def make_screen(fullscreen):
    if fullscreen:
        return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    return pygame.display.set_mode((1280, 720))


def main():
    try:                                    # real screen resolution on Windows
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    pygame.init()
    pygame.display.set_caption("AI HILL CLIMB CONTROLLER")
    fullscreen = True
    screen = make_screen(fullscreen)
    renderer = Renderer(screen)
    game = Game()
    renderer.snap(game)
    clock = pygame.time.Clock()

    tracker = HandTracker()
    tracker.start()

    hold_time = 0.0
    armed = True
    running = True
    try:
        while running:
            dt = min(clock.tick(60) / 1000.0, 1.0 / 30.0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()
                        renderer.snap(game)
                    elif event.key == pygame.K_f:
                        fullscreen = not fullscreen
                        screen = make_screen(fullscreen)
                        renderer = Renderer(screen)
                        renderer.snap(game)

            frame, gesture, tracking, cam_fps = tracker.get()
            keys = pygame.key.get_pressed()
            gas = gesture == OPEN_PALM or keys[pygame.K_RIGHT] or keys[pygame.K_d]
            brake = gesture == CLOSED_FIST or keys[pygame.K_LEFT] or keys[pygame.K_a]
            if gas and brake:
                gas = brake = False

            if game.over:
                # restart by holding an open palm (must release it first)
                if gesture != OPEN_PALM:
                    armed = True
                if armed and gesture == OPEN_PALM:
                    hold_time += dt
                else:
                    hold_time = 0.0
                if hold_time >= 1.5:
                    game.reset()
                    renderer.snap(game)
                    hold_time = 0.0
                    armed = False
            else:
                game.update(dt, gas, brake)
                armed = False
                hold_time = 0.0

            renderer.follow(game, dt)
            renderer.draw_game(game, clock.get_fps())
            if game.over:
                renderer.draw_game_over(game, min(hold_time / 1.5, 1.0))
            renderer.draw_panel(frame, tracking, cam_fps, tracker.status,
                                tracker.error, gesture, clock.get_fps())
            pygame.display.flip()
    finally:
        tracker.stop_event.set()
        tracker.join(timeout=3.0)
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
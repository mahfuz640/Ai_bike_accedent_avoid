"""
Smart Vehicle — ADXL345 + SIM800L GSM + Neo-6M GPS + HC-SR04 + L298N + YOLO + Streamlit
Motor ENB max=255, speed decreases near obstacles.
Tilt > threshold → motor stop → SMS with location
Objects detected: person, car
"""

import time, threading, traceback, gc, subprocess
import cv2, numpy as np

# ================= GPIO SETUP ==================
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except Exception:
    ON_PI = False
    class DummyPWM:
        def start(self,*a,**k): pass
        def ChangeDutyCycle(self,*a,**k): pass
        def stop(self,*a,**k): pass
    class _DummyGPIO:
        BCM = OUT = IN = LOW = HIGH = None
        def setmode(self,*a,**k): pass
        def setwarnings(self,*a,**k): pass
        def setup(self,*a,**k): pass
        def output(self,*a,**k): pass
        def input(self,*a,**k): return 0
        def PWM(self,*a,**k): return DummyPWM()
        def cleanup(self,*a,**k): pass
    GPIO = _DummyGPIO()
    pwm_motor = DummyPWM()

# ================= OPTIONAL MODULES ==================
try: import serial
except Exception: serial=None
try: import busio, board
except Exception: busio=None
try:
    from adafruit_adxl34x import ADXL345
    ADXL_AVAILABLE = True
except Exception: ADXL_AVAILABLE=False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception: YOLO_AVAILABLE=False

# -----------------------------
# LCD SETUP
# -----------------------------
try:
    from RPLCD.i2c import CharLCD
    lcd = CharLCD('PCF8574', 0x27)
    ip = subprocess.getoutput("hostname -I").strip().split()[0]
    lcd.clear()
    lcd.write_string(f"{ip[:16]}")
    lcd.cursor_pos = (1, 0)
    lcd.write_string(":8501")
except Exception as e:
    print("LCD not connected or error:", e)


# ================= STREAMLIT SETUP ==================
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="Transport Security Camera Feed", layout="wide")
    st.markdown("<h2 style='text-align:center;color:#1E90FF'>Live Camera Feed</h2>", unsafe_allow_html=True)
    frame_pl = st.empty()   # ✅ only camera shown
else:
    frame_pl = None

# ================= CONFIG ==================
PHONE_NUMBER = "+8801869974728"
YOLO_MODEL = "yolov8n.pt"
CAM_INDEX = 0

TRIG = 23
ECHO = 24
ENA = 12
ENB = 13
IN3 = 5

ADXL_THRESHOLD_G = 0.6
SMS_COOLDOWN = 10  # seconds

# ================= HARDWARE SETUP ==================
if ON_PI:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.output(TRIG, False)
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(ENB, GPIO.HIGH)

    # ==== PWM SAFE INIT ====
    try:
        for obj in gc.get_objects():
            if isinstance(obj, GPIO.PWM):
                try:
                    obj.stop()
                    del obj
                except:
                    pass
    except:
        pass

    try:
        pwm_motor = GPIO.PWM(ENB, 255)
        pwm_motor.start(0)
        print("PWM initialized safely ✅")
    except RuntimeError as e:
        print("PWM already active — skipping reinit:", e)
        try:
            pwm_motor.ChangeDutyCycle(0)
        except:
            pwm_motor = GPIO.PWM(ENB, 255)
            pwm_motor.start(0)
else:
    pwm_motor = _DummyGPIO().PWM(0,0)

# ================= ADXL345 ==================
adxl = None
if ADXL_AVAILABLE:
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        print("I2C bus started")
        for addr in [0x1D]:
            try:
                adxl = ADXL345(i2c, address=addr)
                print(f"ADXL345 detected at 0x{addr:X}")
                break
            except ValueError as e:
                print(f"No ADXL at 0x{addr:X}, {e}")
        if adxl is None: ADXL_AVAILABLE=False
    except Exception as e:
        print("ADXL345 init failed:", e)
        ADXL_AVAILABLE=False

# ================= GSM ==================
gsm_ser = None
if serial:
    try:
        gsm_ser = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
        print("GSM initialized")
    except: gsm_ser=None

def safe_send_gsm(text, phone=PHONE_NUMBER, timeout=8):
    if gsm_ser is None: return False
    try:
        gsm_ser.write(b'AT\r'); time.sleep(0.3)
        gsm_ser.write(b'AT+CMGF=1\r'); time.sleep(0.5)
        cmd = f'AT+CMGS="{phone}"\r'.encode()
        gsm_ser.write(cmd); time.sleep(0.5)
        gsm_ser.write(text.encode() + b"\r")
        time.sleep(0.2)
        gsm_ser.write(bytes([26]))  # Ctrl+Z
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = gsm_ser.readline().decode(errors='ignore').strip()
            if not line: continue
            if line.startswith("+CMGS") or "OK" in line: return True
            if "ERROR" in line: return False
        return False
    except: return False

# ================= ULTRASONIC ==================
def get_distance():
    if not ON_PI: return 100.0
    GPIO.output(TRIG, True); time.sleep(0.00001); GPIO.output(TRIG, False)
    start = time.time(); timeout = start + 0.02
    while GPIO.input(ECHO) == 0 and time.time() < timeout: start = time.time()
    stop_timeout = time.time() + 0.02
    while GPIO.input(ECHO) == 1 and time.time() < stop_timeout: stop = time.time()
    try: return max(0, (stop - start) * 34300 / 2)
    except: return 100.0

# ================= ADXL ==================
def read_adxl_g():
    if not ADXL_AVAILABLE or adxl is None: return 0.0, 0.0, 0.0
    try:
        ax, ay, az = adxl.acceleration
        return ax/9.80665, ay/9.80665, az/9.80665
    except: return 0.0, 0.0, 0.0

# ================= MOTOR ==================
def motor_stop():
    if ON_PI:
        GPIO.output(IN3, GPIO.LOW)
        pwm_motor.ChangeDutyCycle(0)

def motor_forward(speed=255):
    duty = max(0, min(speed / 255 * 100, 100))
    if ON_PI:
        GPIO.output(IN3, GPIO.HIGH)
        pwm_motor.ChangeDutyCycle(duty)

# ================= YOLO ==================
if YOLO_AVAILABLE:
    try: ymodel = YOLO(YOLO_MODEL)
    except: ymodel=None
else: ymodel=None

def detect_objects(frame):
    if ymodel is None: return frame, {"person":0,"car":0}
    results = ymodel(frame)
    annotated = frame.copy()
    counts = {"person":0,"car":0}
    for res in results:
        for box in res.boxes.cpu().numpy():
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = ymodel.names[cls]
            if name in counts:
                counts[name] += 1
                color = (0,255,0) if name=="person" else (255,0,0)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                cv2.putText(annotated,f"{name} {conf:.2f}",(x1,y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
    return annotated, counts

# ================= CAMERA ==================
class CameraStream:
    def __init__(self, src=0, w=320, h=240, fps=30):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()
    def update(self):
        while self.running:
            ret, f = self.cap.read()
            if ret:
                with self.lock: self.frame = f
            else: time.sleep(0.01)
    def read(self):
        with self.lock: return None if self.frame is None else self.frame.copy()
    def release(self):
        self.running = False
        try: self.cap.release()
        except: pass

# ================= GPS ==================
gps_available = False
gps_ser = None
LAT, LON = 0.0, 0.0
if serial:
    try:
        gps_ser = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1)
        gps_available = True
    except: gps_ser=None

def read_gps():
    global LAT, LON
    if not gps_available or gps_ser is None: return 0.0, 0.0
    try:
        line = gps_ser.readline().decode(errors='ignore').strip()
        if line.startswith("$GPRMC"):
            parts = line.split(",")
            if parts[2]=='A':
                lat = float(parts[3]); lat_dir = parts[4]
                lon = float(parts[5]); lon_dir = parts[6]
                LAT = (lat//100 + (lat%100)/60) * (1 if lat_dir=='N' else -1)
                LON = (lon//100 + (lon%100)/60) * (1 if lon_dir=='E' else -1)
    except: pass
    return LAT, LON

# ================= MAIN LOOP ==================
cam = CameraStream(CAM_INDEX, w=320, h=240, fps=30)
last_sms = 0

try:
    while True:
        frame = cam.read()
        if frame is None:
            if STREAMLIT_AVAILABLE:
                frame_pl.text("Waiting for camera...")
            time.sleep(0.05)
            continue

        annotated, counts = detect_objects(frame)

        # ✅ Streamlit: Only camera feed
        if STREAMLIT_AVAILABLE:
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_pl.image(frame_rgb, caption="Live Camera", use_column_width=True)

        distance = get_distance()
        ax, ay, az = read_adxl_g()
        tilt = (abs(ax) >= ADXL_THRESHOLD_G or abs(ay) >= ADXL_THRESHOLD_G)

        lat, lon = read_gps()

        if tilt:
            motor_stop()
            now = time.time()
            if now - last_sms > SMS_COOLDOWN:
                msg = "Accident detected! Please help!"
                if lat and lon:
                    msg += f"\nLocation: {lat:.6f},{lon:.6f}\nhttps://maps.google.com/?q={lat},{lon}"
                ok = safe_send_gsm(msg, PHONE_NUMBER)
                print("Alert SMS:", "Sent" if ok else "Failed")
                last_sms = now
        else:
            distance_based_speed = 255
            if counts["car"]>0 or counts["person"]>0:
                if distance > 50: distance_based_speed = 254
                elif distance > 40: distance_based_speed = 200
                elif distance > 30: distance_based_speed = 120
                elif distance > 20: distance_based_speed = 75
                elif distance > 10: distance_based_speed = 30
                else: distance_based_speed = 20
            motor_forward(distance_based_speed)

        time.sleep(0.05)

except Exception as e:
    print(traceback.format_exc())
finally:
    try: cam.release()
    except: pass
    motor_stop()
    if ON_PI: GPIO.cleanup()
    if gsm_ser: gsm_ser.close()
    if gps_ser: gps_ser.close()


// ===========================================================================
// Catboy vs Puppyboy Detector — main sketch
// Board: Seeed XIAO ESP32S3 Sense
// Arduino IDE: Tools > Board > XIAO_ESP32S3, PSRAM enabled
//
// Flow: button press -> capture JPEG -> POST to companion server ->
//       parse per-feature scores + ml score -> heuristic + hybrid fusion ->
//       display result
//
// Requires library: ArduinoJson (Library Manager > search "ArduinoJson", by
// Benoit Blanchon)
// ===========================================================================

#include "esp_camera.h"
#include "camera_pins.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ---------------------------------------------------------------------------
// WiFi + server config — fill these in
// ---------------------------------------------------------------------------

const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// IP of the laptop running the FastAPI companion server, e.g. "192.168.1.42"
const char* SERVER_HOST = "192.168.1.42";
const int SERVER_PORT = 8000;
const char* SERVER_PATH = "/analyze";

// Scan button — adjust to whatever pin you actually wire. Using internal
// pull-up, so button should connect this pin to GND when pressed.
const int SCAN_BUTTON_PIN = 2;

// ---------------------------------------------------------------------------
// Scoring code (yours, with the two typos fixed: cheekboneScore ->
// cheekbonesScore, and cosntrainScore -> constrainScore)
// ---------------------------------------------------------------------------

float WEIGHT_CANTHAL_TILT = 1.0;
float WEIGHT_EYE_SIZE = 1.0;
float WEIGHT_NOSE_BRIDGE = 1.0;
float WEIGHT_NOSE_WIDTH = 1.0;
float WEIGHT_FACE_SHAPE = 1.0;
float WEIGHT_CHEEKBONES = 1.0;
float WEIGHT_EYEBROWS = 1.0;

float HEURISTIC_WEIGHT = 0.50;
float ML_WEIGHT = 0.50;

float canthalTiltScore = 0.0;
float eyeSizeScore = 0.0;
float noseBridgeScore = 0.0;
float noseWidthScore = 0.0;
float faceShapeScore = 0.0;
float cheekbonesScore = 0.0;
float eyebrowsScore = 0.0;

float mlScore = 0.0;

float constrainScore(float score) {
  if (score < -1.0) return -1.0;
  if (score > 1.0) return 1.0;
  return score;
}

float calculateHeuristicScore() {
  float weightedScore =
    (canthalTiltScore * WEIGHT_CANTHAL_TILT) +
    (eyeSizeScore * WEIGHT_EYE_SIZE) +
    (noseBridgeScore * WEIGHT_NOSE_BRIDGE) +
    (noseWidthScore * WEIGHT_NOSE_WIDTH) +
    (faceShapeScore * WEIGHT_FACE_SHAPE) +
    (cheekbonesScore * WEIGHT_CHEEKBONES) +
    (eyebrowsScore * WEIGHT_EYEBROWS);

  float totalWeight =
    WEIGHT_CANTHAL_TILT +
    WEIGHT_EYE_SIZE +
    WEIGHT_NOSE_BRIDGE +
    WEIGHT_NOSE_WIDTH +
    WEIGHT_FACE_SHAPE +
    WEIGHT_CHEEKBONES +
    WEIGHT_EYEBROWS;

  if (totalWeight == 0.0) return 0.0;

  return constrainScore(weightedScore / totalWeight);
}

float scoreToPuppyPercent(float score) {
  score = constrainScore(score);
  return (score + 1.0) * 50.0;
}

float scoreToCatPercent(float score) {
  return 100.0 - scoreToPuppyPercent(score);
}

float calculateHybridScore(float heuristicScore, float classifierScore) {
  float totalWeight = HEURISTIC_WEIGHT + ML_WEIGHT;
  if (totalWeight == 0.0) return 0.0;

  float hybridScore = (heuristicScore * HEURISTIC_WEIGHT) + (classifierScore * ML_WEIGHT);
  return constrainScore(hybridScore / totalWeight);
}

// ---------------------------------------------------------------------------
// Camera setup
// ---------------------------------------------------------------------------

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;   // 640x480 — plenty for face landmarks
    config.jpeg_quality = 12;            // lower number = higher quality
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;  // fallback if PSRAM isn't enabled
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// WiFi setup
// ---------------------------------------------------------------------------

void connectWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected, IP address: ");
  Serial.println(WiFi.localIP());
}

// ---------------------------------------------------------------------------
// Send captured frame to companion server, parse response into score vars
// ---------------------------------------------------------------------------

bool scanFace() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return false;
  }

  HTTPClient http;
  String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT + SERVER_PATH;
  http.begin(url);
  http.addHeader("Content-Type", "image/jpeg");

  int httpCode = http.POST(fb->buf, fb->len);
  esp_camera_fb_return(fb);  // release frame buffer as soon as we're done with it

  if (httpCode != HTTP_CODE_OK) {
    Serial.printf("HTTP POST failed, code: %d\n", httpCode);
    http.end();
    return false;
  }

  String payload = http.getString();
  http.end();

  // Expected JSON shape from the server, e.g.:
  // {
  //   "canthal_tilt": 0.2, "eye_size": -0.1, "nose_bridge": 0.05,
  //   "nose_width": -0.3, "face_shape": 0.1, "cheekbones": -0.2,
  //   "eyebrows": 0.0, "ml_score": 0.15
  // }
  JsonDocument doc;  // ArduinoJson v7 — auto-sized
  DeserializationError parseErr = deserializeJson(doc, payload);
  if (parseErr) {
    Serial.print("JSON parse failed: ");
    Serial.println(parseErr.c_str());
    return false;
  }

  canthalTiltScore = doc["canthal_tilt"] | 0.0;
  eyeSizeScore      = doc["eye_size"] | 0.0;
  noseBridgeScore   = doc["nose_bridge"] | 0.0;
  noseWidthScore    = doc["nose_width"] | 0.0;
  faceShapeScore    = doc["face_shape"] | 0.0;
  cheekbonesScore   = doc["cheekbones"] | 0.0;
  eyebrowsScore     = doc["eyebrows"] | 0.0;
  mlScore           = doc["ml_score"] | 0.0;

  return true;
}

// ---------------------------------------------------------------------------
// Display — placeholder until the actual screen is picked. Swap this out
// for real display-driving code (TFT/OLED/LED matrix library) later; every
// other part of the pipeline stays the same.
// ---------------------------------------------------------------------------

void showResult(float catPct, float dogPct, bool conflicted) {
  Serial.println("----------------------------------------");
  Serial.printf("CAT:  %.1f%%\n", catPct);
  Serial.printf("DOG:  %.1f%%\n", dogPct);
  Serial.println(catPct >= dogPct ? "VERDICT: CATBOY" : "VERDICT: PUPPYBOY");
  if (conflicted) {
    Serial.println("(the machine is conflicted)");
  }
  Serial.println("----------------------------------------");
}

// ---------------------------------------------------------------------------
// setup() / loop()
// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(SCAN_BUTTON_PIN, INPUT_PULLUP);

  if (!initCamera()) {
    Serial.println("Halting — camera failed to init.");
    while (true) delay(1000);
  }

  connectWiFi();

  Serial.println("Ready. Press the scan button.");
}

void loop() {
  static bool lastButtonState = HIGH;
  bool buttonState = digitalRead(SCAN_BUTTON_PIN);

  // Detect a press (HIGH -> LOW transition), with a simple debounce delay
  if (lastButtonState == HIGH && buttonState == LOW) {
    delay(30);  // debounce
    if (digitalRead(SCAN_BUTTON_PIN) == LOW) {
      Serial.println("Scanning...");

      if (scanFace()) {
        float heuristicScore = calculateHeuristicScore();
        float hybridScore = calculateHybridScore(heuristicScore, mlScore);

        float catPct = scoreToCatPercent(hybridScore);
        float dogPct = scoreToPuppyPercent(hybridScore);
        bool conflicted = fabs(scoreToCatPercent(heuristicScore) - scoreToCatPercent(mlScore)) > 30.0;

        showResult(catPct, dogPct, conflicted);
      } else {
        Serial.println("Scan failed — check WiFi/server connection.");
      }
    }
  }
  lastButtonState = buttonState;

  delay(20);
}

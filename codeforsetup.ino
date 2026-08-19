#include "esp_camera.h"
#include "camera_pins.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

const char* SERVER_HOST = "192.168.1.42";
const int SERVER_PORT = 8000;
const char* SERVER_PATH = "/analyze";


const int SCAN_BUTTON_PIN = 2

loat WEIGHT_CANTHAL_TILT = 1.0;
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
    config.frame_size = FRAMESIZE_VGA;   
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;  /
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

//============
void connectWiFi() {
  Serial.print("Connecting to WiFi . . .");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected, IP address: ");
  Serial.println(WiFi.localIP());
}


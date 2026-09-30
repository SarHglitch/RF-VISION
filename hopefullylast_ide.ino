#include <WiFi.h>
#include "esp_wifi.h"
#include <string.h>

#define WIFI_CHANNEL 11      // 11 room    6 marvel
uint8_t ROUTER_BSSID[6] = {0x60, 0x32, 0xb1, 0x0f, 0xa6, 0x8c}; // Match your MAC exactly  f4:27:56:78:35:d8 hall 60:32:b1:0f:a6:8c room  e0:1c:fc:66:f2:08 marvel
#define SERIAL_BAUD 115200

void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;
  // MAC filter commented out so ALL packets pass through:
  // if (memcmp(info->mac, ROUTER_BSSID, 6) != 0) return;

  Serial.printf("CSI,%lu,%d,%d,%d", millis(), info->rx_ctrl.rssi, info->rx_ctrl.channel, info->len);
  for (int i = 0; i < info->len; i++) {
    Serial.printf(",%d", info->buf[i]);
  }
  Serial.println();
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(2000); // Wait for serial connection to stabilize

  // 1. Initialize Wi-Fi in Station Mode
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  // 2. Start Wi-Fi driver explicitly (CRITICAL to prevent crashes)
  ESP_ERROR_CHECK(esp_wifi_start());

  // 3. Enable promiscuous mode
  ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);

  // 4. Configure and enable CSI
  wifi_csi_config_t csi_config;
  memset(&csi_config, 0, sizeof(wifi_csi_config_t));
  csi_config.lltf_en = true;
  csi_config.htltf_en = true;
  csi_config.ltf_merge_en = true;
  csi_config.channel_filter_en = false; // Capture packets across all subcarriers
  csi_config.manu_scale = false;

  ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
  ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&wifi_csi_rx_cb, NULL));
  ESP_ERROR_CHECK(esp_wifi_set_csi(true));

  Serial.println("READY_CSI_STREAM");
}

void loop() {
  // Lock to Channel 1 permanently to prevent gain jumps and packet drops
  esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
  delay(100);
}

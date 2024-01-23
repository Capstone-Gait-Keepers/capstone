#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecureBearSSL.h>

#define LOCAL_DEBUG
#define DEFAULT_ATTEMPTS 3

// Replace with your network credentials
const char* ssid = "DataCity-802 - 2.4GHz";
const char* password = "6340631697";

// Create wifi initialization function
bool initialize_wifi() {
    #ifdef LOCAL_DEBUG
      return true;
    #endif
    //Connect to Wi-Fi
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi ..");
    unsigned int attempts = 0;
    while (WiFi.status() != WL_CONNECTED) {
        Serial.print('.');
        delay(1000);
        attempts++;
        if (attempts > 10) {
            Serial.println("Failed to connect to WiFi");
            break;
        }
    }
    if ((WiFi.status() == WL_CONNECTED)) {
        Serial.println("Connected to the WiFi network");
        return true;
    }
    else {
        Serial.println("Failed to connect to WiFi");
        return false;
    }
}

void send_data(String post_data) {
    send_data(post_data, DEFAULT_ATTEMPTS);
}

void send_data(String post_data, uint8_t max_attempts) {
    int httpCode = _send_data(post_data);
    uint8_t attempts = 1;
    while (is_bad_http_code(httpCode) && attempts < max_attempts) {
        Serial.println("\nRetrying...");
        attempts++;
        httpCode = _send_data(post_data);
    }
    if (is_bad_http_code(httpCode)) {
        Serial.println("ERROR: POST REQUEST FAILED");
        sos_mode();
    }
}

// Create function to send data to backend server
#ifdef LOCAL_DEBUG
int _send_data(String post_data) {
    Serial.println("BEGIN_TRANSMISSION");
    Serial.println("http://localhost:8000/api/send_recording");
    Serial.println(post_data);
    // TODO: Check response from host
    return 200;
}
#else
int _send_data(String post_data) {
    int httpCode = -1;
    if (T1C != 0) {
        Serial.println("ERROR: INTERRUPT WAS NOT DISABLED");
        sos_mode();
    }
    std::unique_ptr<BearSSL::WiFiClientSecure>client(new BearSSL::WiFiClientSecure);
    HTTPClient https;

    client->setInsecure(); // Ignore SSL certificate validation
    client->setTimeout(50000); // Time out in milliseconds

    if (https.begin(*client, "https://capstone-backend-f6qu.onrender.com/api/send_recording")) {
        delay(1000); // See if this prevents the problem with connection refused and deep sleep
        https.addHeader("Content-Type", "application/json");
        httpCode = https.POST(post_data);
        String payload = https.getString();
        https.end();

        Serial.printf("[HTTPS] POST Response: (%d) %s\n", httpCode, "Message:");
        Serial.println(payload);
    } else {
      Serial.println("[HTTPS] Unable to connect");
      sos_mode();
    }
    return httpCode;
}
#endif

bool is_bad_http_code(int httpCode) {
    return httpCode < 200 || 300 <= httpCode;
}

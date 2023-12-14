#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecureBearSSL.h>

// Replace with your network credentials
const char* ssid = "DataCity-802 - 2.4GHz";
const char* password = "6340631697";

// Create wifi initialization function
bool initialize_wifi() {
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

// Create function to send data to backend server


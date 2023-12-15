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
void send_data(String post_data) {
    std::unique_ptr<BearSSL::WiFiClientSecure>client(new BearSSL::WiFiClientSecure);

    client->setInsecure();  // Ignore SSL certificate validation
    HTTPClient https;    //create an HTTPClient instance
    
    Serial.print("[HTTPS] begin...\n");     //Initializing an HTTPS communication using the secure client
    if (https.begin(*client, "https://capstone-backend-f6qu.onrender.com/api/send_recording")) {
        delay(1000);            // See if this prevents the problm with connection refused and deep sleep
        https.addHeader("Content-Type", "application/json");    //Specify content-type header
        int httpCode = https.POST(post_data);   //Send the request
        String payload = https.getString();    //Get the response payload

        // httpCode will be negative on error
        if (httpCode > 0) {
            // HTTP header has been send and Server response header has been handled
            Serial.printf("[HTTPS] POST... code: %d\n", httpCode);
            // file found at server
            Serial.println(payload);
        } else {
            Serial.printf("[HTTPS] POST... failed, error: %s\n", https.errorToString(httpCode).c_str(), "Message:\n");
            Serial.println(https.getString());
            sos_mode();
        }
        https.end();
    }
    else {
      Serial.printf("[HTTPS] Unable to connect\n");
      sos_mode();
    }

}

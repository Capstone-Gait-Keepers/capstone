/*
  Complete project details: https://RandomNerdTutorials.com/esp8266-nodemcu-https-requests/ 
  Based on the BasicHTTPSClient.ino Created on: 20.08.2018 (ESP8266 examples)
*/

#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecureBearSSL.h>

// Replace with your network credentials
const char* ssid = "DataCity-802 - 2.4GHz";
const char* password = "6340631697";

//stop pinging server repeatedly
bool sentRequest = false;
unsigned long lastRequestTime = 0;
const unsigned long requestInterval = 120000; // Interval between requests in milliseconds (2 minutes)


void setup() {
  Serial.begin(115200);
  //Serial.setDebugOutput(true);

  Serial.println();
  Serial.println();
  Serial.println();

  //Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi ..");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(1000);
  }
}

void loop() {
//declare
    unsigned long currentTime = millis();

  // wait for WiFi connection
  if ((WiFi.status() == WL_CONNECTED)) {

    std::unique_ptr<BearSSL::WiFiClientSecure>client(new BearSSL::WiFiClientSecure);

    // Ignore SSL certificate validation
    client->setInsecure();
    
    //create an HTTPClient instance
    HTTPClient https;
    
    //Initializing an HTTPS communication using the secure client
    Serial.print("[HTTPS] begin...\n");
    if (https.begin(*client, "https://capstone-backend-f6qu.onrender.com/api/sarah_test4")) {  // https://capstone-backend-f6qu.onrender.com/api/sensor_metadata
      Serial.print("[HTTPS] GET...\n");
      // start connection and send HTTP header
      //int httpCode = https.GET();

    //send a gift to julia
  String postData = "{\"text1\":\"Hi worldd\",\"text2\":\"Hi juliaa\"}";
   //{"sensorid": "210", "sampling": 100, "floor": "cork", "user": "daniel"}

    delay(1000);            // See if this prevents the problm with connection refused and deep sleep
    https.addHeader("Content-Type", "application/json");    //Specify content-type header

    int httpCode = https.POST(postData);   //Send the request
    String payload = https.getString();    //Get the response payload

      // httpCode will be negative on error
      if (httpCode > 0) {
        // HTTP header has been send and Server response header has been handled
        Serial.printf("[HTTPS] GET... code: %d\n", httpCode);
        // file found at server
        if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_MOVED_PERMANENTLY) {
          String payload = https.getString();
          Serial.println(payload);
        }
      } else {
        Serial.printf("[HTTPS] GET... failed, error: %s\n", https.errorToString(httpCode).c_str(), "Message:\n");
        Serial.println(https.getString());
      }

      https.end();
      sentRequest = true;
      lastRequestTime = currentTime;
    } else {
      Serial.printf("[HTTPS] Unable to connect\n");
    }
  } else if (sentRequest && currentTime - lastRequestTime >= requestInterval) {
      sentRequest = false;
    }
  Serial.println();
  Serial.println("Waiting 2min before the next round...");
  delay(12000);
}




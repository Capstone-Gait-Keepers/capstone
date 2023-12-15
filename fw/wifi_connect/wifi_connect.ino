#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecureBearSSL.h>

// Replace with your network credentials
const char* ssid = "DataCity-802 - 2.4GHz";
const char* password = "6340631697";

void setup() {
  // put your setup code here, to run once:
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

  //old loop
  if ((WiFi.status() == WL_CONNECTED)) {

    std::unique_ptr<BearSSL::WiFiClientSecure>client(new BearSSL::WiFiClientSecure);

    // Ignore SSL certificate validation
    client->setInsecure();
    
    //create an HTTPClient instance
    HTTPClient https;
    
    //Initializing an HTTPS communication using the secure client
    Serial.print("[HTTPS] begin...\n");
    if (https.begin(*client, "https://capstone-backend-f6qu.onrender.com/api/send_recording")) {  // https://capstone-backend-f6qu.onrender.com/api/sensor_metadata
      Serial.print("[HTTPS] GET...\n");

    //send a gift to julia
  String postData = "{\"sensorid\": \"18\",\"timestamp\":\"2023-11-25 03:41:23.295\",\"ts_data\":[1.23, -0.0200,-0.0500,-0.0400,-0.0500,-0.0100]}";

    // delay(1000);            // See if this prevents the problm with connection refused and deep sleep
    https.addHeader("Content-Type", "application/json");    //Specify content-type header

    int httpCode = https.POST(postData);   //Send the request
    String payload = https.getString();    //Get the response payload

      // httpCode will be negative on error
      if (httpCode > 0) {
        // HTTP header has been send and Server response header has been handled
        Serial.printf("[HTTPS] GET... code: %d\n", httpCode);
        // file found at server
        Serial.println(payload);
      } else {
        Serial.printf("[HTTPS] GET... failed, error: %s\n", https.errorToString(httpCode).c_str(), "Message:\n");
        Serial.println(https.getString());
      }

      https.end();
    } else {
      Serial.printf("[HTTPS] Unable to connect\n");
    }
    } 
  Serial.println();

}

void loop() {
  // no looping please and thank you

}

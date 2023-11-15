#include <ESP8266WiFi.h>


/*ADD YOUR PASSWORD BELOW*/
const char *ssid = "DataCity-802 - 2.4GHz";
const char *password = "psych";

WiFiClient client;

/*
* Connect your controller to WiFi
*/
void connectToWiFi() {
  //Connect to WiFi Network
  Serial.println();
  Serial.println();
  Serial.print("Connecting to WiFi");
  Serial.println("...");
  WiFi.begin(ssid, password);
  int retries = 0;
  while ((WiFi.status() != WL_CONNECTED) && (retries < 15)) {
    retries++;
    delay(500);
    Serial.print(".");
  }
  if (retries > 14) {
      Serial.println(F("WiFi connection FAILED"));
  }
  if (WiFi.status() == WL_CONNECTED) {
      Serial.println(F("WiFi connected!"));
      Serial.println("IP address: ");
      Serial.println(WiFi.localIP());
  }
  Serial.println(F("Setup ready"));
}

/*
 * call connectToWiFi() in setup()
 */

void setup() {
  Serial.begin(9600);
  connectToWiFi();
}

void loop() {
}



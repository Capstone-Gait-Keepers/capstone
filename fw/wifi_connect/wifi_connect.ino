#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>


/*ADD YOUR PASSWORD BELOW*/
const char *ssid = "DataCity-802 - 2.4GHz";
const char *password = "6340631697";

WiFiClientSecure client;
HTTPClient http;

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

//SETUP

void setup() {
  Serial.begin(9600);
  connectToWiFi();

  String URL = "https://capstone-backend-f6qu.onrender.com/api/sensor_metadata"; // Works with HTTP
  http.begin(client, URL); // Works with HTTP

  /*send a gift to julia*/
    String postData = "data=hellojulia";

    delay(1000);            // See if this prevents the problm with connection refused and deep sleep
    http.addHeader("Content-Type", "application/x-www-form-urlencoded");    //Specify content-type header

    int httpCode = http.POST(postData);   //Send the request
    String payload = http.getString();    //Get the response payload

//debug
    if (httpCode == -1) {
      Serial.print("HTTP return code: -1. Failure to establish a connection or an issue during the HTTP request");
  // Code to execute if the condition is false
  } else { 
    Serial.println(httpCode);   //Print HTTP return code}
  }

    Serial.print(" Payload: ");
    Serial.println(payload);    //Print request response payload

  http.end();  //Close connection

}

void loop() {
}

/*connect and do things
IPAddress server(10,0,0,138);
String PostData = "someDataToPost";

if (client.connect(server, 80)) {
  client.println("POST /Api/AddParking/3 HTTP/1.1");
  client.println("Host: 10.0.0.138");
  client.println("User-Agent: Arduino/1.0");
  client.println("Connection: close");
  client.print("Content-Length: ");
  client.println(PostData.length());
  client.println();
  client.println(PostData);
}
*/

//
// Post data


  




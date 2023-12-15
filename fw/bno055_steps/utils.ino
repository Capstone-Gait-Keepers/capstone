// misc functions

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "ESP8266TimerInterrupt.h"              //https://github.com/khoih-prog/ESP8266TimerInterrupt

void led_on() {
  digitalWrite(LED_BUILTIN, LOW);
}
void led_off() {
  digitalWrite(LED_BUILTIN, HIGH);
}

void sos_mode() {
  i_timer.disableTimer();
  while (true) {
    // blink LED to indicate no wifi connection (SOS in morse code). Use a loop for ech letter to make it easier to read
    for (int i = 0; i < 3; i++) {
      led_on();
      delay(100);
      led_off(); 
      delay(100);                      
    }
    delay(200);
    for (int i = 0; i < 3; i++) {
      led_on();
      delay(1000);
      led_off();
      delay(100);
    }
    delay(200);
    for (int i = 0; i < 3; i++) {
      led_on();
      delay(100);
      led_off(); 
      delay(100);                      
    }
    delay(1000);
  }
}
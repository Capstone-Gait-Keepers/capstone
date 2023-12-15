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

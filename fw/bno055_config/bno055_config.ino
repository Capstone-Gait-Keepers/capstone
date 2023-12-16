#include <Wire.h>
#include <Adafruit_Sensor.h>

#include "Adafruit_BNO055/Adafruit_BNO055.h"


// Check I2C device address and correct line below (by default address is 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);


// Displays some basic information on the sensor from the unified sensor API sensor_t type (see Adafruit_Sensor for more information)
void displaySensorDetails(void) {
  sensor_t sensor;
  bno.getSensor(&sensor);
  Serial.println("------------------------------------");
  Serial.print  ("Sensor:       "); Serial.println(sensor.name);
  Serial.print  ("Driver Ver:   "); Serial.println(sensor.version);
  Serial.print  ("Unique ID:    "); Serial.println(sensor.sensor_id);
  Serial.print  ("Max Value:    "); Serial.print(sensor.max_value); Serial.println(" xxx");
  Serial.print  ("Min Value:    "); Serial.print(sensor.min_value); Serial.println(" xxx");
  Serial.print  ("Resolution:   "); Serial.print(sensor.resolution); Serial.println(" xxx");
  Serial.println("------------------------------------");
  Serial.println("");
  delay(500);
}

void setup(void) {
  Serial.begin(115200);
  Serial.println("Orientation Sensor Test"); Serial.println("");
  // Initialise the sensor
  if(!bno.begin()) {
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  // Use external crystal for better accuracy
  bno.setExtCrystalUse(true);
  // displaySensorDetails();
}

void loop(void) {
  print_sample();
  delay(100);
}

void print_sample() {
  sensors_event_t event;
  bno.getEvent(&event, Adafruit_BNO055::VECTOR_LINEARACCEL);
  Serial.println(event.acceleration.z);
}

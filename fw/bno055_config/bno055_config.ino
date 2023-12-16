#include <Wire.h>
#include <Adafruit_Sensor.h>

#include "BNO055_accel.h"


// Check I2C device address and correct line below (by default address is 0x28)
//                                   id, address
BNO055_accel bno = BNO055_accel(55, 0x28);


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

  Serial.print("Accelerometer Config:");
  Serial.println(bno.read8(BNO055_accel::BNO055_ACC_CONFIG_ADDR));
}

void loop(void) {
  // print_sample();

  bno.update_range(BNO055_accel::G8);
  Serial.print("Accelerometer Config:");
  Serial.println(bno.read8(BNO055_accel::BNO055_ACC_CONFIG_ADDR));

  bno.update_range(BNO055_accel::G2);
  Serial.print("Accelerometer Config:");
  Serial.println(bno.read8(BNO055_accel::BNO055_ACC_CONFIG_ADDR));
  delay(3000);
}

void print_sample() {
  sensors_event_t event;
  bno.getEvent(&event, BNO055_accel::VECTOR_LINEARACCEL);
  Serial.println(event.acceleration.z);
}

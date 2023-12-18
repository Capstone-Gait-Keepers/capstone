#include <Wire.h>
#include <Adafruit_Sensor.h>

#include "BNO055_accel.h"


// Check I2C device address and correct line below (by default address is 0x28)
//                                   id, address
BNO055_accel bno = BNO055_accel(55, 0x28);


// Displays some basic information on the sensor from the unified sensor API sensor_t type (see Adafruit_Sensor for more information)
void displaySensorDetails(void) {
  Serial.println("------------------------------------");
  Serial.print  ("Config: "); Serial.println(bno.read8(BNO055_accel::BNO055_ACC_CONFIG_ADDR));
  Serial.print  ("Units:  "); Serial.println(bno.read8(BNO055_accel::BNO055_UNIT_SEL_ADDR));
  Serial.println("------------------------------------\n");
  delay(500);
}

void setup(void) {
  Serial.begin(115200);
  Serial.println("Orientation Sensor Test\n");
  // Initialise the sensor
  if(!bno.begin()) {
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  // Use external crystal for better accuracy
  bno.setExtCrystalUse(true);
  displaySensorDetails();
}

void loop(void) {
  print_sample();
  delay(100);
}

void print_sample() {
  double accel = bno.getVerticalAcceleration();
  Serial.println(accel);
}

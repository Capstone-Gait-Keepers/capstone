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
  // displaySensorDetails();
  // bno.update_range(BNO055_accel::G2);
  // displaySensorDetails();
  // bno.update_bandwidth(BNO055_accel::Hz250);
  displaySensorDetails();
  Serial.println("accx,accy,accz,gyrx,gyry,gyrz,magx,magy,magz,eulx,euly,eulz,quaw,quax,quay,quaz,linx,liny,linz,grvx,grvy,grvz");
}

void loop(void) {
  // Serial.println("G8");
  // bno.update_range(BNO055_accel::G8);
  // displaySensorDetails();
  // for (int i = 0; i < 20; i++) {
  //   print_sample();
  //   delay(100);
  // }
  // Serial.println("G2");
  // bno.update_range(BNO055_accel::G2);
  // displaySensorDetails();
  for (int i = 0; i < 20; i++) {
    print_sample();
    delay(100);
  }
}

void print_sample() {
  // sensors_event_t event;
  // bno.getEvent(&event);
  // Serial.println(event.acceleration.z);
  BNO055_accel::BNO055_sensed_t sensed;
  bno.getAll(&sensed);

  // Serial.print(sensed.acc_x); Serial.print(","); // Gyro
  // Serial.print(sensed.acc_y); Serial.print(",");
  // Serial.print(sensed.acc_z);
  // Serial.print(",");
  // Serial.print(sensed.mag_x); Serial.print(","); // Gyro
  // Serial.print(sensed.mag_y); Serial.print(",");
  // Serial.print(sensed.mag_z);
  // Serial.print(",");
  Serial.print(sensed.gyr_x); Serial.print(","); // Accel
  Serial.print(sensed.gyr_y); Serial.print(",");
  Serial.print(sensed.gyr_z);
  // Serial.print(",");
  // Serial.print(sensed.eul_x); Serial.print(","); // Euler
  // Serial.print(sensed.eul_y); Serial.print(",");
  // Serial.print(sensed.eul_z);
  // Serial.print(",");
  // Serial.print(sensed.qua_w); Serial.print(","); // Quaternion (probably?)
  // Serial.print(sensed.qua_x); Serial.print(",");
  // Serial.print(sensed.qua_y); Serial.print(",");
  // Serial.print(sensed.qua_z);
  // Serial.print(",");
  // Serial.print(sensed.lin_x); Serial.print(",");
  // Serial.print(sensed.lin_y); Serial.print(",");
  // Serial.print(sensed.lin_z);
  // Serial.print(",");
  // Serial.print(sensed.grv_x); Serial.print(","); // Gravity
  // Serial.print(sensed.grv_y); Serial.print(",");
  // Serial.print(sensed.grv_z);
  Serial.println();
}

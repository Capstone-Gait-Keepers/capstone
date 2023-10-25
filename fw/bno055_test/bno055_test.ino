#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "ESP8266TimerInterrupt.h"              //https://github.com/khoih-prog/ESP8266TimerInterrupt

/* This driver uses the Adafruit unified sensor library (Adafruit_Sensor)

   You should also assign a unique ID to this sensor for use with
   the Adafruit Sensor API so that you can identify this particular
   sensor in any data logs, etc.  To assign a unique ID, simply
   provide an appropriate value in the constructor below (12345
   is used by default in this example).

   Connections
   ===========
   Connect SCL to analog 5
   Connect SDA to analog 4
   Connect VDD to 3.3-5V DC
   Connect GROUND to common ground
*/

/* Set the delay between fresh samples */
#define BNO055_SAMPLERATE_DELAY_MS (10)

// Check I2C device address and correct line below (by default address is 0x29 or 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
ESP8266Timer ITimer;
volatile bool startRunning = false;
int scale = 10;

/**************************************************************************/
/*
    Displays some basic information on this sensor from the unified
    sensor API sensor_t type (see Adafruit_Sensor for more information)
*/
/**************************************************************************/
void displaySensorDetails(void)
{
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

void ICACHE_RAM_ATTR TimerHandler(void)
{
  if (startRunning) {
    Serial.println("ERROR: PREVIOUS SAMPLED WAS NOT FINISHED");
  }
  startRunning = true;
}

void setup(void)
{
  Serial.begin(115200);
  Serial.println("Orientation Sensor Test"); Serial.println("");

  /* Initialise the sensor */
  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
   
  delay(1000);

  /* Use external crystal for better accuracy */
  bno.setExtCrystalUse(true);
   
  /* Display some basic information on this sensor */
  displaySensorDetails();

  // Setup Timer
  ITimer.attachInterruptInterval(10000, TimerHandler); // 10ms interval
  ITimer.enableTimer();
}

void loop(void)
{
  if (startRunning) {
    /* Get a new sensor event for linear acceleration */
    sensors_event_t event;
    bno.getEvent(&event, Adafruit_BNO055::VECTOR_LINEARACCEL);
    startRunning = false;

    // Print in format: timestamp, acceleration, event
    Serial.print(millis());
    Serial.print(",");
    Serial.print(event.acceleration.z);
    Serial.println(", ");

    // Print absolute value of acceleration
    // Serial.print("ABS:");
    // Serial.println(sqrt(event.acceleration.x*event.acceleration.x + event.acceleration.y*event.acceleration.y + event.acceleration.z*event.acceleration.z));
  }

  // add event button

  // delay(BNO055_SAMPLERATE_DELAY_MS);
}

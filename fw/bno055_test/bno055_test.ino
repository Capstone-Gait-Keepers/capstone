#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "ESP8266TimerInterrupt.h"              //https://github.com/khoih-prog/ESP8266TimerInterrupt

/*
  Note from Adafruit:
  You should also assign a unique ID to this sensor for use with
  the Adafruit Sensor API so that you can identify this particular
  sensor in any data logs, etc.  To assign a unique ID, simply
  provide an appropriate value in the constructor below (12345
  is used by default in this example).

   Connections
   ===========
    Accelerometer:
    Connect SCL to D1 (GPI05)
    Connect SDA to D2 (GPI04)
    Connect VDD to 3.3-5V DC
    Connect GROUND to common ground
    Button:
    Connect the sense wire from button to D3 (GPI00)
    Connect one side of button to 3v and the other to ground
*/

#define BUTTON_PIN 0 // D3 (GPI00)

// Check I2C device address and correct line below (by default address is 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
ESP8266Timer i_timer;  // Hardware Timer


volatile bool start_sampling = false; // Flag to indicate if a new sample should be taken based on timer interrupt
boolean button_pressed = false; // Flag to indicate if button is pressed
boolean button_pressed_last = false; // Flag to indicate if button was pressed last time (used to prevent multiple presses)
int interrupt_interval = 10000; // Interval (sample rate) for timer interrupt in microseconds
int scale = 10; // Scale factor for acceleration, currently not used

// Displays some basic information on the sensor from the unified sensor API sensor_t type (see Adafruit_Sensor for more information)
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
// Interrupt handler for timer
void ICACHE_RAM_ATTR TimerHandler(void)
{
  if (start_sampling) {
    Serial.println("ERROR: PREVIOUS SAMPLED WAS NOT FINISHED");
  }
  start_sampling = true;
}

void setup(void)
{
  Serial.begin(115200);
  Serial.println("Orientation Sensor Test"); Serial.println("");

  // Initialise the sensor
  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }

  pinMode(BUTTON_PIN, INPUT);
  
  delay(1000);

  /* Use external crystal for better accuracy */
  bno.setExtCrystalUse(true);
   
  /* Display some basic information on this sensor */
  displaySensorDetails();

  // Setup Timer
  i_timer.attachInterruptInterval(interrupt_interval, TimerHandler);
  i_timer.enableTimer();
}

void loop(void)
{
  if (digitalRead(BUTTON_PIN) == LOW) { // Check if button is pressed
    button_pressed = true;
  }
  else if (button_pressed_last) { // reset button_pressed_last to false once button is released
    button_pressed_last = false;
  }
  else {
    button_pressed = false;
  }
  if (start_sampling) {
    double vertical_accel = getVerticalAcceleration();
    start_sampling = false;
    // Print in format: timestamp, acceleration, event
    Serial.print(millis());
    Serial.print(",");
    Serial.print(vertical_accel);
    Serial.print(",");
    // Print event only if button is pressed, else print space
    if (button_pressed && !button_pressed_last){
      Serial.println("BUTTON");
      button_pressed = false;
      button_pressed_last = true;
    }
    else {
      Serial.println(" ");
    }
  }
}

double getVerticalAcceleration() {
  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  imu::Vector<3> grav = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
  grav.normalize();
  return accel.dot(grav);
}

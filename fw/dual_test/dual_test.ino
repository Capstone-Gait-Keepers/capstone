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

#define PIEZO_PIN A0 // A0 (ADC0)
#define BUTTON_PIN 0 // D3 (GPI00)
#define PIEZO_SAMPLE_RATE 200 // Hz
#define ACCEL_SAMPLE_RATE 100 // Hz
#define SAMPLE_RATE max(PIEZO_SAMPLE_RATE, ACCEL_SAMPLE_RATE) // Hz
#define MICROSECONDS 1000000 // Microseconds in a second
#define INTERRUPT_INTERVAL (int)(MICROSECONDS / SAMPLE_RATE) // us
#define ACCEL_INTERRUPTS_PER_SAMPLE (int)(SAMPLE_RATE / ACCEL_SAMPLE_RATE) // Number of interrupts per sample
#define PIEZO_INTERRUPTS_PER_SAMPLE (int)(SAMPLE_RATE / PIEZO_SAMPLE_RATE) // Number of interrupts per sample

// Check I2C device address and correct line below (by default address is 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
ESP8266Timer i_timer;  // Hardware Timer

volatile bool start_sampling = false; // Flag to indicate if a new sample should be taken based on timer interrupt
volatile double accel_sample = -1;
volatile double piezo_sample = -1;
volatile uint sample_count = 0; // Number of samples taken
volatile unsigned long timestamp = 0;

boolean button_pressed = false; // Flag to indicate if button is pressed
boolean button_pressed_last = false; // Flag to indicate if button was pressed last time (used to prevent multiple presses)

// Interrupt handler for timer
void ICACHE_RAM_ATTR TimerHandler(void)
{
  if (start_sampling) {
    Serial.println("ERROR: PREVIOUS SAMPLED WAS NOT FINISHED");
  } else {
    start_sampling = true;
    sample_count++;
    timestamp = millis();
    if (sample_count % PIEZO_INTERRUPTS_PER_SAMPLE == 0) {
      piezo_sample = getPiezoVoltage();
    }
    if (sample_count % ACCEL_INTERRUPTS_PER_SAMPLE == 0) {
      accel_sample = getVerticalAcceleration();
    }
  }
}

void setup(void)
{
  Serial.begin(115200);
  Serial.println("FW: Sample Collection\n");

  if(!bno.begin()) {
    Serial.println("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  delay(1000);
  bno.setExtCrystalUse(true);

  pinMode(BUTTON_PIN, INPUT);
  i_timer.attachInterruptInterval(INTERRUPT_INTERVAL, TimerHandler);
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
    // Print in format: timestamp,accel,piezo,event
    Serial.print(timestamp);
    Serial.print(",");
    if (sample_count % PIEZO_INTERRUPTS_PER_SAMPLE == 0) {
      Serial.print(piezo_sample);
    }
    Serial.print(",");
    if (sample_count % ACCEL_INTERRUPTS_PER_SAMPLE == 0) {
      Serial.print(accel_sample);
    }
    start_sampling = false;
    // Print event only if button is pressed, else print space
    if (button_pressed && !button_pressed_last){
      Serial.println(",BUTTON");
      button_pressed = false;
      button_pressed_last = true;
    } else {
      Serial.println(",");
    }
  }
}

double getVerticalAcceleration() {
  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  imu::Vector<3> grav = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
  grav.normalize();
  return accel.dot(grav);
}

double getPiezoVoltage() {
  int sensorValue = analogRead(PIEZO_PIN);
  double voltage = sensorValue * (3.3 / 1023.0);
  double zeroedVolts = (voltage - 1.65);
  return zeroedVolts;
}

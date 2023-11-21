#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "ESP8266TimerInterrupt.h"              //https://github.com/khoih-prog/ESP8266TimerInterrupt

/*
   Connections
   ===========
    Accelerometer:
    Connect SCL to D1 (GPI05)
    Connect SDA to D2 (GPI04)
    Connect VDD to 3.3-5V DC
    Connect GROUND to common ground
*/

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
ESP8266Timer i_timer;  // Hardware Timer

#define SAMPLE_RATE 100 // Sample rate for accelerometer in Hz
#define INTERRUPT_INTERVAL_US (1000000/SAMPLE_RATE) // Interval between samples in microseconds
#define START_BUFFER_TIME 1 // Size of buffer to store acceleration data prior to first step (in seconds)
#define END_BUFFER_TIME 2 // Size of buffer to store acceleration data after last step (in seconds)
#define START_BUFFER_SAMPLES START_BUFFER_TIME * SAMPLE_RATE  // In samples
#define END_BUFFER_SAMPLES END_BUFFER_TIME * SAMPLE_RATE // Number of bad samples to save before stopping saving data
#define AMP_THRESHOLD 0.5 // Threshold for amplitude of acceleration data to be considered "good" (in m/s^2)

volatile bool start_sampling = false; // Flag to indicate if a new sample should be taken based on timer interrupt

float sensor_data_buffer[START_BUFFER_SAMPLES]; // Circular buffer to store acceleration data
int buffer_index = 0; // Index of circular buffer
String post_data = ""; // String to store data to be sent to backend server
bool save_sample = false; // Flag to indicate if a sample should be saved to post_data
bool post_data_ready = false; // Flag to indicate if post_data is ready to be sent to backend server
int bad_data_count = 0; // Counter to logic to stop saving data to post_data if too many bad samples are received
bool good_sample = false; // Flag to indicate if a sample is considered "good" (above threshold)

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
  delay(1000);
  Serial.println("Orientation Sensor Test\n");

  // Initialise the sensor
  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("No BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  
  delay(1000);

  /* Use external crystal for better accuracy */
  bno.setExtCrystalUse(true);

  /* Display some basic information on this sensor */
  displaySensorDetails();

  // Setup Timer
  Serial.print("Interrupt period: ");
  Serial.println(INTERRUPT_INTERVAL_US);
  i_timer.attachInterruptInterval(INTERRUPT_INTERVAL_US, TimerHandler);
  i_timer.enableTimer();
}

void loop(void)
{
  if (start_sampling) 
  {
    // Get a new sensor event for linear acceleration
    sensors_event_t event;
    bno.getEvent(&event, Adafruit_BNO055::VECTOR_LINEARACCEL);
    start_sampling = false;

    // Update circular buffer
    sensor_data_buffer[buffer_index] = event.acceleration.z;
    good_sample = abs(event.acceleration.z) > AMP_THRESHOLD;

    // if sample is considered good (above threshold), save buffer to string and start saving data to post_data
    if (good_sample && !save_sample) {
      Serial.println("Begin saving data");
      // save buffer to string
      for (int i = 0; i < START_BUFFER_SAMPLES; i++) {
          post_data += String(sensor_data_buffer[i]) + ",";
      }
      save_sample = true;
    }
    if (good_sample && save_sample) {
      // save sample to string
      post_data += String(event.acceleration.z) + ",";
      bad_data_count = 0;
    }
    if (!good_sample && save_sample) {
        if (bad_data_count < END_BUFFER_SAMPLES) {
            post_data += String(event.acceleration.z) + ",";
            bad_data_count++;
        } else {
            // stop saving
            save_sample = false;
            post_data_ready = true;
            bad_data_count = 0;
        }
      }
    if (post_data_ready) {
        Serial.println("POST DATA READY");
        // Serial.println(post_data);
        // post_data = "";
        post_data_ready = false;
    }
    buffer_index = (buffer_index + 1) % START_BUFFER_SAMPLES; // Move to the next index, modulus handle wraparound
  }
}

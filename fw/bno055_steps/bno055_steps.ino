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

#define USER_ID 0 // User ID for this device

#define SAMPLE_RATE 100 // Sample rate for accelerometer in Hz
#define INTERRUPT_INTERVAL_US (1000000/SAMPLE_RATE) // Interval between samples in microseconds
#define START_BUFFER_TIME 1 // Size of buffer to store acceleration data prior to first step (in seconds)
#define END_BUFFER_TIME 2 // Size of buffer to store acceleration data after last step (in seconds)
#define START_BUFFER_SAMPLES START_BUFFER_TIME * SAMPLE_RATE  // In samples
#define END_BUFFER_SAMPLES END_BUFFER_TIME * SAMPLE_RATE // Number of bad samples to save before stopping saving data
#define CALIBRATION_TIME 10 // Time to run calibration for (in seconds)

// #define AMP_THRESHOLD 0.5 // Threshold for amplitude of acceleration data to be considered "good" (in m/s^2)
float amp_threshold = 0.5; // Threshold for amplitude of acceleration data to be considered "good" (in m/s^2)

volatile bool start_sampling = false; // Flag to indicate if a new sample should be taken based on timer interrupt

bool calibration_flag = false; // Flag to indicate if the device is in calibration mode
bool wifi_status = false; // Flag to indicate if the device is connected to wifi

float sensor_data_buffer[START_BUFFER_SAMPLES]; // Circular buffer to store acceleration data
int start_buffer_index = 0; // Index of circular buffer
String post_data = ""; // String to store data to be sent to backend server
bool save_sample_flag = false; // Flag to indicate if a sample should be saved to post_data
bool post_data_ready = false; // Flag to indicate if post_data is ready to be sent to backend server
int end_buffer_length = 0; // Counter to logic to stop saving data to post_data if too many bad samples are received
bool good_sample = false; // Flag to indicate if a sample is considered "good" (above threshold)


// Interrupt handler for timer
void ICACHE_RAM_ATTR TimerHandler(void)
{
  if (start_sampling) {
    Serial.println("ERROR: PREVIOUS SAMPLED WAS NOT FINISHED");
  }
  start_sampling = true;
}

void update_circular_buffer(float accel_z) {
  // Update circular buffer
  sensor_data_buffer[start_buffer_index] = accel_z;
  good_sample = abs(accel_z) > amp_threshold;
}

void save_circular_buffer() {
  Serial.println("Begin saving data");
  for (int i = 0; i < START_BUFFER_SAMPLES; i++) {
      post_data += String(sensor_data_buffer[i]) + ",";
  }
  save_sample_flag = true;
}
void save_sample(float accel_z) {
  post_data += String(accel_z) + ",";
}

void stop_saving_samples() {
  save_sample_flag = false;
  post_data_ready = true;
  end_buffer_length = 0;
}

void running_mode() {
  if (start_sampling) 
  {
    float accel_z = getVerticalAcceleration(); // Get z component of acceleration
    start_sampling = false; // Reset flag for interrupt handler

    update_circular_buffer(accel_z); // Update circular buffer with new sample

    // if sample is considered good (above threshold), save buffer to string and start saving data to post_data
    if (good_sample && !save_sample_flag) {
      save_circular_buffer();
      led_on();
    }
    if (good_sample && save_sample_flag) {
      save_sample(accel_z);
      end_buffer_length = 0;
    }
    if (!good_sample && save_sample_flag) {
        if (end_buffer_length < END_BUFFER_SAMPLES) {
          save_sample(accel_z);
          end_buffer_length++;
        } 
        else {
          stop_saving_samples(); // Stop saving samples & reset flags if too many samples don't meet threshold
        }
      }
    if (post_data_ready) {
        Serial.println("POST DATA READY");\
        i_timer.disableTimer();
        led_off();
        // remove last comma if it exists
        if (post_data.endsWith(",")) {
          post_data = post_data.substring(0, post_data.length() - 1);
        }
        String json = "{\"sensorid\":\"" + String(USER_ID) + "\",\"timestamp\":\"" + String(millis()) + "\",\"ts_data\":[" + post_data + "]}";
        Serial.println(json);
        send_data(json, 3);
        post_data_ready = false;
        i_timer.enableTimer();
    }
    start_buffer_index = (start_buffer_index + 1) % START_BUFFER_SAMPLES; // Move to the next index, modulus handle wraparound
  }
}

double getVerticalAcceleration() {
  imu::Vector<3> accel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
  imu::Vector<3> grav = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
  grav.normalize();
  return accel.dot(grav);
}


void setup(void)
{
  Serial.begin(115200);
  delay(100);
  Serial.println("FW: Step Detection\n");

  if(!bno.begin()) {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.println("No BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  delay(1000);
  bno.setExtCrystalUse(true);

  pinMode(LED_BUILTIN, OUTPUT);     // Initialize the LED_BUILTIN pin as an output
  digitalWrite(LED_BUILTIN, HIGH);   // Turn the LED on by making the voltage LOW
  wifi_status = initialize_wifi();

  Serial.print("Sample Rate (Hz): ");
  Serial.println(SAMPLE_RATE);
  i_timer.attachInterruptInterval(INTERRUPT_INTERVAL_US, TimerHandler);
  i_timer.enableTimer();

  calibration_flag = true;
}

void loop(void)
{
  if (!wifi_status) {
    sos_mode();
  }
  if (calibration_flag) {
    calibration_mode();
  }
  else {
    running_mode();
  }
}

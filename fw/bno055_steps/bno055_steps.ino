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

void update_circular_buffer(float accel_z) {
  // Update circular buffer
  sensor_data_buffer[start_buffer_index] = accel_z;
  good_sample = abs(accel_z) > amp_threshold;
}

void save_starting_buffer() {
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
    start_sampling = false; // Reset flag for interrupt handler

    sensors_event_t event; 
    bno.getEvent(&event, Adafruit_BNO055::VECTOR_LINEARACCEL); // Get a new sensor event for linear acceleration
    float accel_z = event.acceleration.z; // Get z component of acceleration

    update_circular_buffer(accel_z); // Update circular buffer with new sample

    // if sample is considered good (above threshold), save buffer to string and start saving data to post_data
    if (good_sample && !save_sample_flag) {
      save_starting_buffer();
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
        // TODO: Add code to send post_data to backend server
        Serial.println(post_data);
        // post_data = "";
        String fun_data = "{\"sensorid\": \"18\",\"timestamp\":\"2023-11-25 03:41:23.295\",\"ts_data\":[1.23, 4.56, 7.89, -0.81,0.00]}";
        Serial.println(fun_data);
        // remove last comma if it exists
        if (post_data.endsWith(",")) {
          post_data = post_data.substring(0, post_data.length() - 1);
        }
        String post_data_formatted = "{\"sensorid\": \"18\",\"timestamp\":\"2023-11-25 03:41:23.295\",\"ts_data\":[" + post_data + "]}";
        Serial.println(post_data_formatted);
        send_data(fun_data);
        post_data_ready = false;
        i_timer.enableTimer();
    }
    start_buffer_index = (start_buffer_index + 1) % START_BUFFER_SAMPLES; // Move to the next index, modulus handle wraparound
  }
}

void sos_mode() {
  i_timer.disableTimer();
  while (true) {
    // blink LED to indicate no wifi connection (SOS in morse code). Use a loop for ech letter to make it easier to read
    for (int i = 0; i < 3; i++) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(100);
      digitalWrite(LED_BUILTIN, LOW);  
      delay(100);                      
    }
    for (int i = 0; i < 3; i++) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(1000);
      digitalWrite(LED_BUILTIN, LOW); 
      delay(100);
    }
  }
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

  pinMode(LED_BUILTIN, OUTPUT);     // Initialize the LED_BUILTIN pin as an output
  digitalWrite(LED_BUILTIN, HIGH);   // Turn the LED on by making the voltage LOW
  wifi_status = initialize_wifi();

  // Setup Timer
  Serial.print("Interrupt period: ");
  Serial.println(INTERRUPT_INTERVAL_US);
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

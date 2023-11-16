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
*/
// Check I2C device address and correct line below (by default address is 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);
ESP8266Timer i_timer;  // Hardware Timer

#define BUFFER_SIZE 100 // Size of buffer to store acceleration data (in samples)
#define amp_threshold 0.5 // Threshold for amplitude of acceleration data to be considered "good"

volatile bool start_sampling = false; // Flag to indicate if a new sample should be taken based on timer interrupt
int interrupt_interval = 10000; // Interval (sample rate) for timer interrupt in microseconds

float sensor_data_buffer[BUFFER_SIZE];
int buffer_index = 0;
String post_data = "";
bool save_sample = false;
bool post_data_ready = false;
int bad_data_count = 0;
bool good_sample = false;

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
// Saves specificied window size of data to EEPROM
boolean is_post_data (float data[], int size) {
    // TODO: identify if data window is worth saving
    for (int i = 0; i < size; i++) {
        if (abs(data[i]) > amp_threshold) {
            return true;
        }
    }
    return false;
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
  if (start_sampling) {
    // Get a new sensor event for linear acceleration
    sensors_event_t event;
    bno.getEvent(&event, Adafruit_BNO055::VECTOR_LINEARACCEL);
    start_sampling = false;

    // Print in format: timestamp, acceleration, event

    // Serial.print(millis());
    // Serial.print(",");
    // Serial.print(event.acceleration.z);
    // Serial.print(",");
    // Serial.println(" ");

    // Update circular buffer
    sensor_data_buffer[buffer_index] = event.acceleration.z;
    good_sample = abs(event.acceleration.z) > amp_threshold;

    // if sample is considered good (above threshold), save buffer to string and start saving data to post_data
    if (good_sample)
    {
        if (!save_sample) 
        {
            Serial.println("Begin saving data");
            // save buffer to string
            for (int i = 0; i < BUFFER_SIZE; i++) {
                post_data += String(sensor_data_buffer[i]) + ",";
            }
            save_sample = true;
        } 
        else 
        {
            // save sample to string
            post_data += String(event.acceleration.z) + ",";
            bad_data_count = 0;
        }

    } else if(save_sample) {
        if (bad_data_count < 200) {
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

    // if (good_sample && !save_sample) {
    //     Serial.println("Begin saving data");
    //     // save buffer to string
    //     for (int i = 0; i < BUFFER_SIZE; i++) {
    //         post_data += String(sensor_data_buffer[i]) + ",";
    //     }
    //     save_sample = true;
    // } 
    // if (abs(event.acceleration.z) > amp_threshold )

    // if (save_sample) {
    //     // save sample to string
    //     post_data += String(event.acceleration.z) + ",";
    // }

    // if (buffer_index == BUFFER_SIZE - 1) {
    //     // if sensor_data_buffer contains great amplitude, SAVE WOOOOO
    //     if (is_post_data(sensor_data_buffer, BUFFER_SIZE)) {
    //         Serial.println("Good window");
    //         // save buffer to string
    //         for (int i = 0; i < BUFFER_SIZE; i++) {
    //             post_data += String(sensor_data_buffer[i]) + ",";
    //         }
    //     }
    // } 

    buffer_index = (buffer_index + 1) % BUFFER_SIZE; // Move to the next index, modulus handle wraparound
  }
}

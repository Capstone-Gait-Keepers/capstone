// File with all the calibration functions
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "ESP8266TimerInterrupt.h"              //https://github.com/khoih-prog/ESP8266TimerInterrupt

float max_amp = 0;
float noise_amp = 0.05;
float calibration_data[CALIBRATION_TIME * SAMPLE_RATE];
int calibration_index = 0;
int calibration_samples = 0;

void calibration_mode() {
    if (start_sampling)
    {
        start_sampling = false; // Reset flag for interrupt handler
        if (calibration_index == 0) {
            Serial.println("CALIBRATION MODE");
        }
        // Run calibration for 10 seconds. Gather data and set the thresholds for good samples to half the max amplitude
        if (calibration_samples < CALIBRATION_TIME * SAMPLE_RATE) {
            sensors_event_t event; 
            bno.getEvent(&event, Adafruit_BNO055::VECTOR_LINEARACCEL); // Get a new sensor event for linear acceleration
            float accel_z = event.acceleration.z; // Get z component of acceleration
            calibration_data[calibration_index] = accel_z;
            calibration_index++;
            if (abs(accel_z) > max_amp) {
            max_amp = abs(accel_z);
            }
            calibration_samples++;
        }
        if ((calibration_index == CALIBRATION_TIME * SAMPLE_RATE - 1) && (max_amp - noise_amp >= 0.1)) {
            amp_threshold = (max_amp - noise_amp) / 2;
            Serial.println("Calibration finished. Max amplitude: " + String(max_amp) + " Threshold: " + String(amp_threshold));
            calibration_flag = false;
        }
        else if ((calibration_index == CALIBRATION_TIME * SAMPLE_RATE - 1) && (max_amp - noise_amp < 0.1)) {
            Serial.println("Calibration failed. Max amplitude: " + String(max_amp) + " Threshold: " + String(amp_threshold));
            calibration_flag = false;
        }
    }
}
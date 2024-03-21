// File with all the calibration functions
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "ESP8266TimerInterrupt.h"              //https://github.com/khoih-prog/ESP8266TimerInterrupt

#define FAILED_THRESHOLD_AMP 0.1    // Minimum amplitude to consider calibration successful
#define NOISE_AMP 0.05    // Assumed noise amplitude
#define MAX_AMP_RATIO 0.5   // Ratio of max amplitude to set as threshold

float max_amp = 0;
int calibration_index = 0;

void calibration_mode() {
    if (start_sampling) {
        start_sampling = false; // Reset flag for interrupt handler
        if (calibration_index == 0) {
            Serial.println("CALIBRATION MODE");
        }
        if (calibration_index % 100 == 0) {
            led_on();
        }
        else if (calibration_index % 100 == 50) {
            led_off();
        }
        // Run calibration. Gather data and set the thresholds for good samples based on the max amplitude
        if (calibration_index < CALIBRATION_TIME * SAMPLE_RATE) {
            float accel_z = getVerticalAcceleration();
            if (abs(accel_z) > max_amp) { // Update max amplitude
                max_amp = abs(accel_z);
            }
            calibration_index++;
        }
        if ((calibration_index == CALIBRATION_TIME * SAMPLE_RATE - 1) && (max_amp - NOISE_AMP >= FAILED_THRESHOLD_AMP)) {
            // amp_threshold = (max_amp + NOISE_AMP) * MAX_AMP_RATIO;
            Serial.println("Calibration finished. Max amplitude: " + String(max_amp) + " Threshold: " + String(amp_threshold));
            calibration_flag = false;
        }
        else if ((calibration_index == CALIBRATION_TIME * SAMPLE_RATE - 1) && (max_amp - NOISE_AMP < FAILED_THRESHOLD_AMP)) {
            Serial.println("Calibration failed. Max amplitude: " + String(max_amp) + " Threshold: " + String(amp_threshold));
            calibration_flag = false;
            sos_mode(); // If calibration fails, go to SOS mode
        }
    }
}
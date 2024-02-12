#include "ESP8266TimerInterrupt.h" //https://github.com/khoih-prog/ESP8266TimerInterrupt


#define BUTTON_PIN 0 // D3 (GPI00)
#define PIEZO_PIN A0 // A0 (ADC0)
#define SAMPLE_RATE 500 // Hz
#define MICROSECONDS 1000000 // Microseconds in a second

ESP8266Timer i_timer;  // Hardware Timer


volatile bool start_sampling = false; // Flag to indicate if a new sample should be taken based on timer interrupt
boolean button_pressed = false; // Flag to indicate if button is pressed
boolean button_pressed_last = false; // Flag to indicate if button was pressed last time (used to prevent multiple presses)
int interrupt_interval = MICROSECONDS / SAMPLE_RATE; // Interval (sample rate) for timer interrupt in microseconds
int scale = 10; // Scale factor for acceleration, currently not used


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
  Serial.println("FW: Sample Collection\n");

  pinMode(BUTTON_PIN, INPUT);
  pinMode(PIEZO_PIN, INPUT);
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
  int sensorValue = analogRead(PIEZO_PIN);
  double voltage = sensorValue * (3.3 / 1023.0);
  double acceleration = (voltage - 1.65);
  return acceleration;
}

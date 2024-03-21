const int microphonePin = A0;
#define ALPHA 0.2

uint16_t output = 0;

void setup() {
  Serial.begin(115200);
}

uint16_t EWMA(uint16_t old_output, uint16_t reading) {
  return ALPHA * ((float)reading) + (1 - ALPHA) * ((float)old_output);
}

void loop() {
  uint16_t reading = analogRead(microphonePin);
  output = EWMA(output, reading);
  Serial.print(400);
  Serial.print(" ");
  Serial.print(output);
  Serial.print(" ");
  Serial.println(600);
}


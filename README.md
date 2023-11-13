# Capstone :)

## Folder Structure
- fw: Firmware. Embedded software a.k.a things that run on micros
- data_analysis: All postprocessing or data capturing. Currently includes simulations as well
- datasets: All recorded data
- frontend: TBD
- backend: TBD


## Data Collection Procedure
The purpose of data collection is to collect labelled data to develop the step detection algorithm and assess its accuracy. To standardize the procedure, all measurements should be taken in a single, controlled environment.

Required Equipment
- 2 People (Walker & Observer)
- Prototype: MCU with sensing element (accelerometer) and button
- Computer with USB port to connect to the MCU
- Micro-USB to USB cable (2 meter) (must be capable of data transmission)
- Tape measure
- Tape (electrical tape)
- Video recorder (e.g. phone)


Pre - Setup
1. Define a testing plan that identified which environmental variables will be controlled and varied.
    a. The goal of testing will be to collect data for all combinations of variables. This will follow 2^k rule. 

Setup
1. Setup the desired environment, including any intentionally placed obstacles.
2. Tape the sensing element to the floor located according to the testing plan. Measure the distance from the sensor to the nearest wall, as well as the nearest obstacle. Update `RecordingEnvironment` variable in the `data_capture.py` python script.
3. Use small pieces of tape to mark the walking path(s). Assuming a straight walk, tape should mark the start and end of the walk. Measure the distance from each tape marker (start and end) to the sensor, as well as the distance in between the start and end points. Update `RecordingEnvironment` variable in the `data_capture.py` python script.
4. Connect the MCU to a computer. Ideally the computer is 1-2 meters away, and on an elevated surface (table or chair)

Collecting Data
1. Upload the firmware to the MCU while it is connected to a computer (using Arduino IDE). The computer should start receiving serial data.
2. Run the `data_capture.py` python script. Input the desired duration of the recording.
3. Optionally, record a video of the walk for posterity.
4. Start walk. Observer should click button at the moment of each heel strike. Currently, each button is hardcoded to be a step.
5. To end the recording early, input a keyboard interrupt.

### Event Categories
- `step`: Initial heel contact with floor (user, pet, etc.)
- `fall`: Source fell to the floor (user, frying pan, etc.)
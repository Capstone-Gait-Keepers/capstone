# Capstone :)

## Folder Structure
- fw: Firmware. Embedded software a.k.a things that run on micros
- data_analysis: All postprocessing or data capturing. Currently includes simulations as well
- datasets: All recorded data
- frontend: TBD
- backend: TBD


## Data Collection Procedure
To standardize the procedure, all measurements should be taken in a single, controlled environment.
For now, this is our living room. In the future we will test in the gait analysis lab. 

Required Equipment
- 2 People (Walker & Observer)
- MCU with sensing element and button
- Computer with USB to connect to the MCU
- Tape measure
- Tape
- Video recorder (e.g. phone)

Setup
1. Tape the sensing element to the floor. Take note of its distance to obstacles, walls, and it's overall surrounding.
2. Connect the MCU to a computer. Ideally the computer is far away, and on an elevated surface.
3. Setup the desired environment around the sensor, including any intentionally placed obstacles.
4. Define the walk type (e.g. shuffle, limp, etc.) and path. Measure the starting and end distance from the sensor, as well as the total length of the walk.
5. Update `RecordingEnvironment` variable in the `data_capture.py` python script.

Running
1. Run the fw on the MCU while it is connected to a computer. The computer should start receiving serial data.
2. Run the `data_capture.py` python script. Input the desired duration of the recording.
3. Optionally, record a video of the walk for posterity.
4. Start walk. Observer should click button at the moment of each heel strike. Currently, each button is hardcoded to be a step.
5. To end the recording early, input a keyboard interrupt.

### Event Categories
- `step`: Initial heel contact with floor (user, pet, etc.)
- `fall`: Source fell to the floor (user, frying pan, etc.)
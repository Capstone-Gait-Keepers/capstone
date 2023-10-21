# Capstone :)

## Folder Structure
- fw: Firmware. Embedded software a.k.a things that run on micros
- data_analysis: All postprocessing or data capturing. Currently includes simulations as well
- datasets: All recorded data
- frontend: TBD
- backend: TBD


## Data Collection Method
Two people, one walking and another using a stopwatch to collect the source of truth

1. Define walk path. This can be used to geometrically determine the distance from the sensor for each step
2. Run `data_capture.py`, input environment variables
3. Start walk. Observer should click button at the moment of each heel strike. Currently, each button is hardcoded to be a step.
4. End recording.

### Event Categories
- `step`: Initial heel contact with floor (user, pet, etc.)
- `fall`: Source fell to the floor (user, frying pan, etc.)
# Capstone :)

## Folder Structure
- fw: Firmware. Embedded software a.k.a things that run on micros
- data_analysis: All postprocessing or data capturing. Currently includes simulations as well
- datasets: All recorded data
- frontend: Hosts a Vue app
- backend: Hosts a flask App

# Running the Project
## Firmware Setup
1. Add `https://arduino.esp8266.com/stable/package_esp8266com_index.json` as an additional board manager URL (in preferences)
2. Install `esp8266timerinterrupt` library
3. Select NodeMCU 1.0 as the board type
4. Compile and run :)

## Frontend Setup
1. Install [Node](https://nodejs.org/en)
   - Check box to automatically install all dependencies, if applicable 
   - Verify installation with `npm -v`
2. Add `frontend/.env` file (get it from Dan)
3. Install dependencies and run a local development server:
```
cd frontend
npm install
npm run dev
```

## Backend
### Installations
#### Locally hosted Backend
- Python 3.11.6 virtual environment (`python3.11 -m venv ./path/to/venv/folder`)
- Install dependencies in backend_requirements.txt
- `backend/.env` file from cybersecurity risk thread in slack

#### Locally hosted DB
- PostgreSQL https://www.postgresql.org/download/
    - Install with default settings 
- Docker https://docs.docker.com/desktop/install/windows-install/
    - PostgreSQL image 
        - Run 'docker pull postgres' to download the image 
        - Documentation: https://hub.docker.com/_/postgres
- DBeaver https://dbeaver.io/download/
    - or other DB IDE of choice, setup instructions may vary 
    - pgAdmin4 comes with the PostgreSQL installation but I found creating a connection confusing 

### Running the Backend
- If locally hosting the DB, point the `.env` file to the local url
- Run app.py, app will be live locally
- Run client.py in dedicated terminal to test

### Running the Database
#### Docker
- in terminal, run 'docker run --name some-postgres -e POSTGRES_PASSWORD=mysecretpassword -d postgres'
    - name = name of image, your choice
    - password = set a password, remember it preferably
    - 'd' means it runs in 'detached mode'
- To run your image, the command is 'docker start some-postgres' where some-postgres is your image name
- To stop your image, the command is 'docker stop some-postgres'

#### Connecting PostgreSql locally with Docker Image
Make sure your docker image is running first!

Tutorial link that I found helpful: https://www.youtube.com/watch?v=RdPYA-wDhTA&list=PLZDOU071E4v4S95kbGgRebjKYC5eqSGPM&index=5&ab_channel=DatabaseStar  (starts from Docker installation)
1. Open DBeaver
2. 'Create new connection' by clicking plug button in top left
3. Select PostgreSQL -> Next
4. Host: localhost
5. Port: 5432
6. Database: postgres
7. password: the password you set with the docker image
8. Click 'Test Connection' -> you should see a success image.
9. If connection works, hit finish

#### FLASK APP
- Run app.py, app will be live locally
- Run client.py in dedicated terminal to test
- Terminal should display success message + data should appear in 'sensors' db table

## Current backend + DB status:
- POST API endpoint set up for testing purposes in Flask app
- PostgreSQL locally hosted on Docker image
- Can receive data from POST endpoint and push it to said database.

### Viewing Frontend Pages from Flask
If you'd like to host the frontend using Flask, you need to build a static version of the frontend.
```
cd frontend
npm run build
```

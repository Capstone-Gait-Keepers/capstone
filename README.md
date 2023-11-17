# Capstone :)

## Folder Structure
- fw: Firmware. Embedded software a.k.a things that run on micros
- data_analysis: All postprocessing or data capturing. Currently includes simulations as well
- datasets: All recorded data
- frontend: TBD
- backend: TBD

## Data Collection Procedure
Refer to https://docs.google.com/document/d/1Wm_FNKl8EB9JSIcI2R2lFsQD-1NN7LGn7f8f7JlDUrw/edit 

### Event Categories
- `step`: Initial heel contact with floor (user, pet, etc.)

## Backend
### Necessary Tools:
- Python 3.11.6
- PostgreSQL https://www.postgresql.org/download/
    - install with default settings 
- Docker https://docs.docker.com/desktop/install/windows-install/
    - PostgreSQL image 
        - run 'docker pull postgres' to download the image 
        - Documentation: https://hub.docker.com/_/postgres
- Install dependencies in backend_requirements.txt 
- DBeaver https://dbeaver.io/download/
    - or other DB IDE of choice, setup instructions may vary 
    - pgAdmin4 comes with the PostgreSQL installation but I found creating a connection confusing 

### Setup:
#### DOCKER
- in terminal, run 'docker run --name some-postgres -e POSTGRES_PASSWORD=mysecretpassword -d postgres'
    - name = name of image, your choice
    - password = set a password, remember it preferably
    - 'd' means it runs in 'detached mode'
- To run your image, the command is 'docker start some-postgres' where some-postgres is your image name
- To stop your image, the command is 'docker stop some-postgres'

#### CONNECT POSTGRESQL LOCALLY WITH DOCKER IMAGE
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

### Next steps:
- Deploy backend from main
- Deploy DB
- Get local backend talking to deployed DB
- Get deployed backend talking to deployed DB

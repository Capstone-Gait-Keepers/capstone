#!/bin/bash
# Run frontend and backend in separate processes


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# Start the database, if -d flag is passed
# if [ "$1" = "-d" ]; then
#   docker-compose up -d
# fi

# Build the frontend instead of running it with dev if -b flag is passed
if [ "$1" = "-b" ]; then
  npm run build
else
  npm run dev &
fi

# Start the backend
cd ../backend
python app.py &

# Wait for both processes to finish
wait

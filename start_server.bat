@ECHO ON
cd ./server
call activate py36
START ngrok http 8080
timeout /t 5
cd ..
call python -W ignore serverMain.py
pause
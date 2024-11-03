# Web services: Intro to flask
## Writing a simple ping/pong app
## Querying it with `curl` abd browser
### Turning a python function into web service and access this function from some other process like terminal or browser

from flask import Flask

app = Flask('ping')

# add a decorator
@app.route('/ping', methods=['GET'])
def ping():
    return "PONG"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696) #curl http://localhost:9696/ping
    
    
    

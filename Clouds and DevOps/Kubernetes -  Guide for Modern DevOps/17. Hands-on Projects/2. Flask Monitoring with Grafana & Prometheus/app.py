from flask import Flask, jsonify
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST, start_http_server

app = Flask(__name__)

# Define a Prometheus Counter metric
REQUEST_COUNT = Counter('app_requests_total', 'Total number of requests', ['method', 'endpoint'])

@app.route("/")
def index():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    return "Welcome to the Flask app!"

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route("/hello")
def hello():
    REQUEST_COUNT.labels(method="GET", endpoint="/hello").inc()
    return jsonify({"message": "Hello, World!"})

if __name__ == "__main__":
    start_http_server(8000)
    app.run(host="0.0.0.0", port=5000)

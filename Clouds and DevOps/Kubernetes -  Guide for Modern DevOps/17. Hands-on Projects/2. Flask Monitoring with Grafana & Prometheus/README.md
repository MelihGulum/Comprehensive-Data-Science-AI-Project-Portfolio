# **Flask Monitoring with Grafana & Prometheus**

This project demonstrates setting up a simple Flask application deployed on Kubernetes with monitoring using Grafana and Prometheus. It's an excellent starting point for learning how to integrate monitoring solutions with a basic web application.

This script is a simple Flask web application that exposes Prometheus metrics. It defines a Prometheus Counter metric to track the total number of requests, categorized by HTTP method and endpoint. The application serves a home route (/), a /hello route, and a /metrics route, where Prometheus can scrape the metrics at http://localhost:8000/metrics.

⚠️ **Note**: Replace 'your-docker-hub-username' with your actual Docker Hub username in both the `kubernetes->flask->deployment.yaml` and `Makefile` files.

## **1. Prerequisites**
- Docker installed on your system
- Minikube installed and configured
- Kubernetes CLI (`kubectl`) installed
- Optional: Helm installation for managing Helm charts for Grafana and Prometheus. But Helm is not necessary for this project; however, it’s a choice. ([Source](https://semaphoreci.com/blog/prometheus-grafana-kubernetes-helm))
- A container registry to push your Docker image

## **2. Deployment Steps**
Refer to the provided Makefile for automation. Below are the key steps for deployment:

1. **Start Minikube**: Set up a local Kubernetes cluster.
```bash
make setup-minikube
```
2. **Create Namespace**: Create a Kubernetes namespace for monitoring tools and the Flask app.
 ```bash
make create-namespace
```
3. **Build and Push Docker Image**: Create a Docker image of the Flask application and push it to a container registry.
 ```bash
make docker-build
make docker-push
```
4. **Deploy Resources**: Deploy the Flask app and monitoring tools (Prometheus and Grafana) to Kubernetes.
 ```bash
make deploy
```
5. **Access Services**: Use port-forwarding to access the Flask app, Grafana, and Prometheus services locally.
   - **Grafana**:  Accessible at `http://localhost:3000`
     ```bash
     make port-forward-grafana
     ```  
   - **Prometheus**: Accessible at `http://localhost:9090`
     ```bash
     make port-forward-prometheus
     ```  
   - **Flask Application**: Accessible at `http://localhost:8080`
     ```bash
     make port-forward-flask
     ```  
6. **Cleanup Resources**: Remove all Kubernetes resources and namespaces created for this project.
```bash
make clean
```

### **b. Optional: Automate All Steps with Make**
You can automate the entire process by running all steps in sequence:
1. **Run All Steps**
```bash
make all
```
2. You need at least **three terminal windows** to run port-forwarding for each service:
- Terminal 1: `make port-forward-flask`
- Terminal 2: `make port-forward-grafana`
- Terminal 3: `make port-forward-prometheus`

### **c. Optional: Manual Command Execution** 
Alternatively, you can run individual steps manually by following the commands listed in the Makefile.

## **3. Create Grafana Dashboards**
Now you can create a simple Grafana dashboard to visualize your Flask app's metrics:
1. Go to Create > Dashboard in the Grafana interface.
2. Click Add Query and select Prometheus as the data source.
3. In the query section, enter:
```plaintext
app_requests_total
```
4. Create a panel and visualize the metric.
This dashboard will help you monitor the number of requests your Flask app has processed in real-time.
## **4. Final Project Overview**
After completing the setup:
- **Flask Application**: [http://localhost:8080](http://localhost:8080)  
  - A basic Flask app demonstrating monitoring capabilities.
- **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000)  
  - Default credentials: **Username**: `admin`, **Password**: `admin`  
  - Explore dashboards and visualize metrics.
- **Prometheus Metrics**: [http://localhost:9090](http://localhost:9090)  
  - Prometheus interface to query and analyze collected metrics.

You now have a functional Flask application monitored by Grafana and Prometheus, ready for further exploration and customization!

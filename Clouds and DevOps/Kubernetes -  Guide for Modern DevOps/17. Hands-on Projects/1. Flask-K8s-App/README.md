# **Flask-K8s-App**
Creating a Kubernetes project for a Flask application can help you understand how to deploy and manage web applications in Kubernetes. This guide outlines how to set up a simple Flask application and deploy it to Kubernetes.

## **Prerequisites**
* Docker installed and configured
* Kubernetes (kubectl) CLI installed
* Minikube installed
* (Optional) A Docker Hub account for pushing the image

## **Deployment Steps**
### **1. Manual Deployment**
Follow these steps to deploy the application manually:

* Build the Docker image:
```
docker build -t your-docker-hub-username/flask-k8s-app .
```
* Push the image to Docker Hub:
```
docker push your-docker-hub-username/flask-k8s-app
```
* Start Minikube:
```
minikube start
```
* Deploy the application to Kubernetes:
```
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```
* Access the Minikube dashboard (Optional): You can monitor your cluster's resources visually with the Minikube dashboard:
```
minikube dashboard
```
* Access the application: To access the application via the service:
```
minikube service flask-k8s-app-service
```
You should get an output similar to this:
```
|-----------|-----------------------|-------------|---------------------------|
| NAMESPACE |         NAME          | TARGET PORT |            URL            |
|-----------|-----------------------|-------------|---------------------------|
| default   | flask-k8s-app-service |          80 | http://192.168.49.2:31612 |
|-----------|-----------------------|-------------|---------------------------|
* Starting tunnel for service flask-k8s-app-service.
|-----------|-----------------------|-------------|------------------------|
| NAMESPACE |         NAME          | TARGET PORT |          URL           |
|-----------|-----------------------|-------------|------------------------|
| default   | flask-k8s-app-service |             | http://127.0.0.1:63084 |
|-----------|-----------------------|-------------|------------------------|
* Opening service default/flask-k8s-app-service in default browser...
! Because you are using a Docker driver on windows, the terminal needs to be open to run it.

```
* Scale the deployment (Optional): Kubernetes makes it simple to scale your application horizontally. For example, to scale your deployment to 5 replicas:
```
kubectl scale deployment flask-k8s-app --replicas=5
```
After scaling, check the updated status of the pods:
```
kubectl get pods
```
* Clean up resources: When you're done, clean up the resources to free up cluster resources:
```
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
```

### **2. Deployment Using Makefile (Optional)**
For convenience, you can automate the build, push, and deployment process using the provided Makefile. Below are the available targets:

* **Build and Deploy All**: Run the following command to build the image, push it to Docker Hub, and deploy it to Kubernetes:
```
make all
```
* **Access the Application**: Once deployed, expose the service in Minikube:
```
minikube service flask-k8s-app-service
```
* **Clean Up Resources**: To delete all Kubernetes resources and clean up your environment:
```
make clean
```

# **Final Project Overview**
* **Flask Application**: A basic web app built with Flask that returns "Hello, Kubernetes!" on the home page.
* **Docker**: The Flask app was containerized using Docker, making it portable and ready to run in any Kubernetes environment.
* **Kubernetes**: We deployed the Flask app on Kubernetes using YAML files for the deployment and service configuration. We also scaled the app and accessed it through Minikube.
* **Minikube Dashboard**: The Minikube dashboard provided a UI for monitoring and managing the cluster.
* **Makefile**: The project includes a Makefile to automate common tasks such as logging into Docker Hub, building and pushing the Docker image, starting Minikube, deploying the app to Kubernetes, and cleaning up resources.

By following this tutorial, you now have a simple Flask application running on Kubernetes with multiple replicas for scaling. You can use this as a base for more complex applications in the future.
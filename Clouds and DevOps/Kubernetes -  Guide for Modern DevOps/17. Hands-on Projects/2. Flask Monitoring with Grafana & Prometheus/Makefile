# Makefile for setting up Flask app on Kubernetes with Prometheus and Grafana

# Variables
DOCKER_IMAGE=your-docker-hub-username/flask-app
TAG=latest
NAMESPACE=monitoring

.PHONY: all setup-minikube create-namespace docker-build docker-push deploy port-forward-grafana port-forward-prometheus port-forward-flask clean

# Step 1: Minikube Setup
setup-minikube:
	@echo "[INFO] Starting Minikube..."
	minikube start
	@echo "[SUCCESS] Minikube is up and running."

# Step 2: Create Namespace
create-namespace: setup-minikube
	@echo "[INFO] Creating namespace $(NAMESPACE)..."
	kubectl get namespace $(NAMESPACE) || kubectl create namespace $(NAMESPACE)
	@echo "[SUCCESS] Namespace $(NAMESPACE) is ready."

# Step 3: Docker Build and Push
docker-build:
	@echo "[INFO] Building Docker image $(DOCKER_IMAGE):$(TAG)..."
	docker build -t $(DOCKER_IMAGE):$(TAG) .
	@echo "[SUCCESS] Docker image $(DOCKER_IMAGE):$(TAG) built successfully."

docker-push:
	@echo "[INFO] Pushing Docker image $(DOCKER_IMAGE):$(TAG) to registry..."
	docker push $(DOCKER_IMAGE):$(TAG)
	@echo "[SUCCESS] Docker image $(DOCKER_IMAGE):$(TAG) pushed successfully."

# Step 4: Deploy
deploy:
	@echo "[INFO] Deploying Flask app to Kubernetes..."
	kubectl apply -n $(NAMESPACE) -R -f kubernetes/
	@echo "[SUCCESS] Flask app deployed successfully."

# Step 5: Access Servers (Grafana, Prometheus, Flask)
port-forward-grafana:
	@echo "[INFO] Port-forwarding Grafana service to localhost:3000..."
	kubectl port-forward -n $(NAMESPACE) service/grafana 3000:3000

port-forward-prometheus:
	@echo "[INFO] Port-forwarding Prometheus service to localhost:9090..."
	kubectl port-forward -n $(NAMESPACE) service/prometheus 9090:9090

port-forward-flask:
	@echo "[INFO] Port-forwarding Flask app service to localhost:8080..."
	kubectl port-forward -n $(NAMESPACE) service/flask-app 8080:80

# Step 6: Cleanup
clean:
	@echo "[INFO] Cleaning up resources..."
	kubectl delete -n $(NAMESPACE) -R -f kubernetes|| echo "[INFO] No deployments to delete."
	kubectl delete namespace $(NAMESPACE) || echo "[INFO] Namespace $(NAMESPACE) already deleted."
	@echo "[SUCCESS] Cleanup complete."

# Step 7: Run All Steps
all: setup-minikube create-namespace docker-build docker-push deploy
	@echo "[SUCCESS] All steps completed successfully."

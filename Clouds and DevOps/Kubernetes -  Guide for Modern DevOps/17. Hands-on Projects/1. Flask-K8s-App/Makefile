# Variables
IMAGE_NAME=your-docker-hub-username/flask-k8s-app
KUBE_NAMESPACE=default

# Targets
all: login build push start deploy

login:
	@echo 🔑 Logging in to Docker Hub...
	@echo your-docker-hub-personal-access-token | docker login
	@echo ✅ Successfully logged in to Docker Hub.
	@echo ==========================================

build:
	@echo 🔨 Building Docker image: $(IMAGE_NAME)...
	docker build -t $(IMAGE_NAME) .
	@echo ✅ Docker image built successfully.
	@echo ==========================================

push:
	@echo 📤 Pushing Docker image to Docker Hub: $(IMAGE_NAME)...
	docker push $(IMAGE_NAME)
	@echo ✅ Docker image pushed successfully.
	@echo ==========================================

start:
	@echo 🚀 Starting Minikube...
	minikube start
	@echo ✅ Minikube started successfully.
	@echo ==========================================

deploy: start
	@echo 📦 Deploying application to Kubernetes in namespace: $(KUBE_NAMESPACE)...
	kubectl apply -f deployment.yaml --namespace $(KUBE_NAMESPACE)
	kubectl apply -f service.yaml --namespace $(KUBE_NAMESPACE)
	@echo ✅ Application deployed successfully.
	@echo ==========================================

clean:
	@echo 🧹 Cleaning up Kubernetes resources in namespace: $(KUBE_NAMESPACE)...
	kubectl delete -f deployment.yaml --namespace $(KUBE_NAMESPACE)
	kubectl delete -f service.yaml --namespace $(KUBE_NAMESPACE)
	@echo ✅ Cleanup completed.
	@echo ==========================================

.PHONY: all build push start deploy clean

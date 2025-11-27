
# **README.md**

## **1. Project Title and Overview**

**Pancreas Tumor Classification System**

This project implements an AI-based medical imaging system designed to classify pancreas CT slices as either *normal* or *tumor*.
The primary objective is to build a complete, production-ready AI pipeline including:

* Deep learning model development (ResNet50 + CBAM)
* A real-time inference interface using Gradio
* Grad-CAM visual explanations
* User feedback collection
* Containerized deployment using Docker
* System monitoring with Prometheus and Grafana
* Documentation and video demonstration for reproducibility

The system demonstrates end-to-end AI lifecycle management aligned with modern ML engineering practices.

---

## **2. Repository Contents**

### **src/**

Contains all core system files required for inference.

* `app.py` — Main application, including preprocessing, inference, Grad-CAM, metrics, and Gradio interface
* `model.py` — Implementation of the ResNet50-CBAM architecture
* `model.pth` — Trained model weights
* `data/` — *Small testing dataset*

  * Includes **5 normal** and **5 tumor** sample images for validating the running system
* `feedback_log.csv` — Automatically generated feedback log from user inputs
* `requirements.txt` — Python dependencies
* `utils/` (if added) — Helper functions

---

### **deployment/**

Contains all deployment components.

* `Dockerfile` — Containerization setup for the AI model and Gradio application
* `docker-compose.yml` — Orchestrates multi-container deployment including:

  * Model inference service
  * Prometheus metrics service
  * Grafana visualization service

---

### **monitoring/**

Configuration files for performance monitoring.

* `prometheus.yml` — Metric scraping configuration pointing to the running model service

---

### **documentation/**


* AI System Project Proposal Report

---

### **videos/**

Contains demo and supplementary visuals.

* Real-time demo screen recording showing:

  * Dockerized deployment
  * Model inference
  * Metrics tracking
  * Prometheus + Grafana dashboards
* `Feedback.jpg` — An additional image showing the feedback functionality in the UI

---
### **Colab notebooks/**
This contains the colab notebooks that has been done intially. It has the been done till the gradio implementaion.


---

## **3. System Entry Point**

**Main script:**

```
src/app.py
```

### **How to run locally (without Docker)**

```
pip install -r src/requirements.txt
python src/app.py
```

The Gradio app will start at:

```
http://localhost:7860
```

### **How to run fully containerized (recommended)**

```
docker compose up --build
```

This launches:

* Gradio UI (port 7860)
* Prometheus (port 8000 → scrape target, 9090 UI)
* Grafana (port 3000)

---

## **4. Video Demonstration**

A complete system demonstration is included under `videos/`.
The video shows:

* Starting the dockerized system
* Accessing the Gradio inference interface
* Uploading test images
* Grad-CAM visualization
* Submitting feedback
* Inspecting live Prometheus metrics
* Viewing dashboards in Grafana

The demonstration confirms the system’s functionality from deployment to monitoring.

---

## **5. Deployment Strategy**

The system uses **Docker-based containerization** and **Docker Compose** for multi-service orchestration.

Key components deployed:

* **AI Model Service** — Gradio interface + Prometheus metrics
* **Prometheus** — Metric scraper for inference latency and prediction counts
* **Grafana** — Dashboard visualization

Deployment ensures:

* Portability
* Reproducibility
* Isolated environments
* Scalability for multi-container setups

Refer to:

```
deployment/Dockerfile
deployment/docker-compose.yml
```

---

## **6. Monitoring and Metrics**

The system integrates:

* **Prometheus** for:

  * Inference latency histogram
  * Prediction counters
  * Feedback counters
  * Runtime resource metrics

* **Grafana** for:

  * Visualization dashboards
  * Monitoring live inference activity
  * Tracking system health and performance

To access dashboards:

* Prometheus → `http://localhost:9090/metrics`
* Grafana → `http://localhost:3000` (default login: admin / admin)

---

## **7. Project Documentation**

Included under `documentation/`:

* **AI System Project Proposal Template** — Completed and aligned with assignment requirements
* **Project Report** (optional but recommended)

All documentation files clearly describe:

* System objectives
* Lifecycle design
* Risk & trustworthiness considerations
* Deployment and monitoring plans
* Performance evaluation

---

## **8. Version Control and Team Collaboration**

This project follows standard Git best practices:

* Clean folder organization
* Meaningful commit messages
* Versioning through Git & GitHub
* Project re-uploaded using a structured workflow
* All development contained in a single-developer repository

---

## **9. Not Applicable Items**

Certain components such as Kubernetes, cloud deployment services, and large-scale orchestration tools were not used because the system does not require distributed infrastructure or cloud-based scaling. The project functions effectively with local containerization, so additional platforms were unnecessary for this deployment approach.


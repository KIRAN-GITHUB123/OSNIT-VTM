# OSINT-VTM: A Hybrid Classical-Deep Learning Framework for Automated Visual Threat Intelligence

> **Note:** This repository contains the raw, single-file research prototype developed for the Computer Vision Lab Project (CSE-3144) at Manipal Institute of Technology. It is provided as proof-of-execution for the underlying business logic, parallel compute architecture, and algorithmic heuristics. The system is currently being modularized into a production-ready microservice architecture.
> 
> Recorded Accuracy: 72-78%; varies by scene and quality of images sourced from Mapillary.
> 

## Executive Summary

The OSINT-VTM (Open-Source Intelligence Visual Threat Mapper) is a framework designed for the automated discovery and analysis of security and safety risks from geotagged street-level imagery.

Traditional manual analysis of platforms like Mapillary is infeasible at scale, while standard Deep Learning (DL) models are computationally expensive and limited by closed vocabularies. This framework solves these limitations by employing a novel hybrid detection pipeline. It synergizes the speed of classical computer vision algorithms for rapid candidate generation with the high-precision verification of modern, open-vocabulary deep learning models (YOLOv8x-Worldv2 and CLIP).

Beyond simple object detection, the system fuses low-level detections into high-level, actionable intelligence, outputting a complete intelligence product that includes a multi-page PDF threat report and an interactive map.


## System Architecture & Methodology

The pipeline transforms raw geotagged images into a structured intelligence report through a multi-stage process:

### 1. Hybrid Detection Pipeline

The core detection engine maximizes recall with classical methods before maximizing precision with DL models:

* **Pass 1: Single-Pass DL Caching:** A preprocessed image is fed once into a dynamically configured YOLOv8x-Worldv2 model to cache open-vocabulary detections (e.g., "security camera," "graffiti").

* **Pass 2: Classical Candidate Generation:** A suite of computationally cheap classical detectors (Haar cascades, HOG, Hough transforms, and contour-based shape analysis) rapidly identifies all potential regions of interest.

* **Pass 3: DL-Based Verification (The Filter):** High-recall candidates are validated against the DL cache. Candidates receive a confidence boost based on Intersection over Union (IoU) with YOLO boxes or cosine similarity via CLIP (ViT-B/32) image-text embeddings. A dynamic filtering logic then discards unverified false positives.


### 2. Advanced Contextual & Temporal Analysis

The system moves beyond simple bounding boxes to generate situational insights:

* **Temporal Scene Change Detection:** Images are grouped by geographic coordinates and sorted by timestamp. A Kanade-Lucas-Tomasi (KLT) tracker analyzes feature displacement across sequential images to flag significant scene changes (e.g., tampered infrastructure).

* **Contextual Risk Fusion (Sign Obstruction):** The system analyzes spatial overlaps between different finding types. If a "graffiti" detection overlaps heavily with a "stop sign," the system intelligently fuses these and re-classifies the event as a critical "sign_obstructed" threat.


## Actionable Intelligence & Reporting

Detections are translated into actionable, executive-ready outputs:

* **Time-Decayed Risk Scoring:** Detections are assigned a base risk score (e.g., 9.5 for obstructed signs) which is modulated by a time-decay function, reducing the risk weight of older imagery by 10% per year to prioritize recent threats.

* **Automated Intelligence Reports:** The pipeline automatically generates a comprehensive, multi-page PDF report featuring an executive summary, a mitigation matrix mapping threats to recommended actions, and a detailed findings table with visual snippets.

* **Interactive Geovisualization:** The system generates a standalone interactive HTML map using `folium`, plotting the precise location of each finding color-coded by its risk severity.



## Core Tech Stack

* **Data Acquisition:** Mapillary Graph API

* **Classical Computer Vision:** OpenCV (cv2)

* **Deep Learning & NLP:** PyTorch, Ultralytics (YOLOv8x-Worldv2), OpenCLIP (ViT-B/32), PyTesseract (OCR)

* **Reporting & Visualization:** ReportLab, Folium

* **Hardware Acceleration:** Native support for Apple Silicon (MPS) and standard CPU/GPU scaling

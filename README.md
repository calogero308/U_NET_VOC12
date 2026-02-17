# U_NET_VOC12
U_Net network designed for the VOC12 dataset, aimed at performing semantic image segmentation by accurately identifying and classifying each object into 21 classes.

U-Net Semantic Segmentation Project
Overview

This project implements a U-Net network for semantic segmentation on the VOC12 dataset. The goal is to segment objects into 21 classes, including background, using a combination of Cross-Entropy and Dice loss.

The project is intended as a portfolio/demonstration project, showcasing model design, training, evaluation, and prediction workflows, with code organized and documented for clarity. Note: model performance is limited due to dataset size, computational constraints, and training time.

Features:
  
  -U-Net architecture for semantic segmentation
  
  -Training loop with AdamW optimizer and combined CE + 0.5 * Dice loss
  
  -Data augmentation pipeline with albumentations
  
  -Evaluation using mean IoU (mIoU) and per-class IoU
  
  -Prediction script for single images
  
  -Checkpoints saved at multiple stages (e.g., 40, 45, 50 epochs) "I only loaded the 50 epochs
  
  -Interactive main script to choose between training, evaluation, and prediction

Note: Dataset VOC12 and large model checkpoints (>100MB) are not included. See instructions below to download or link externally. 

Model trained at epoch 50: https://www.dropbox.com/scl/fo/hc9uzb34gye8j76kgvo12/AMwYjQlW8PgqLb0dZEX7AdQ?rlkey=jejhh52syswsy4izsyakafge3&st=ro57wj0u&dl=0


Create a virtual environment (optional but recommended):
  -python -m venv venv
  
  -source venv/bin/activate  # Linux/macOS
  
  -venv\Scripts\activate     # Windows

Install dependencies:

  -pip install -r requirements.txt

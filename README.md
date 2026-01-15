# Simulation of a Bioinspired Robotic Arm Controlled Using Deep Learning Techniques / Original Thesis Title: Simulación de un brazo robótico bioinspirado controlado mediante técnicas de deep learning
A project that explores how electronics can move closer to the human body. This thesis presents the simulation of a bio-inspired robotic arm controlled through deep learning techniques and real EMG signals, enabling more natural, precise, and efficient movements. It proposes an approach oriented toward the future of prosthetic control and bionics.

# Content
## 1. data

This folder contains the surface EMG signals acquired using the **FREYA EMG Sensor** system.  
The dataset includes recordings of four upper limb movements. Each movement was captured using surface electrodes placed over specific muscle groups, and both raw signals and their corresponding FFT transformations are provided.

**Important:** The conventions are the initials of the name of the muscle and the correponding movement in Spanish.

---

### Elbow Flexion and Extension
- **Muscles recorded:**  
  - **Biceps brachii** (B)  
  - **Forearm / Antebrazo** (A) 
- **Signal naming convention:**  
  - **B** → Raw biceps EMG  
  - **B_fft** → FFT-transformed biceps EMG  
  - **A** → Raw forearm EMG  
  - **A_fft** → FFT-transformed forearm EMG  

---

### Front Shoulder 
*(Vertical Adduction and Abduction of the Shoulder)*

- **Muscles recorded:**  
  - **Anterior Deltoid Frontal / Deltoides Anterios Hombro Frontal** (DAHF)  
  - **Lateral Deltoid Frontal / Deltoides Externo Hombro Frontal** (DEHF)  
- **Signal naming convention:**  
  - **DAHF** → Raw EMG of the Anterior Deltoid (Frontal)  
  - **DAHF_fft** → FFT-transformed signal  
  - **DEHF** → Raw EMG of the Lateral Deltoid (Frontal)  
  - **DEHF_fft** → FFT-transformed signal  

---

## Saggital Shoulder
*(Flexion and Extension of the Shoulder)*

---

## Tranverse Shoulder
*(Horizontal Adduction and Abduction of the Shoulder)*



Each signal is stored in CSV format for easy integration with Python, MATLAB, and deep learning pipelines.  



## 2. simulation


## 3. modelues


## 4. figures


## 5. documents




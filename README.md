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
  - **Biceps** (B)  
  - **Forearm / Antebrazo** (A) 
- **Signal naming convention:**  
  - **B** → Raw EMG of the bicep  
  - **B_fft** → FFT transformed signal of B  
  - **A** → Raw EMG os the forearm
  - **A_fft** → FFT transformed signal of A  

---

### Frontal Shoulder 
*(Vertical Adduction and Abduction of the Shoulder)*

- **Muscles recorded:**  
  - **Anterior Deltoid Frontal Shoulder / Deltoides Anterior Hombro Frontal** (DAHF)  
  - **Lateral Deltoid Frontal Shoulder / Deltoides Externo Hombro Frontal** (DEHF)  
- **Signal naming convention:**  
  - **DAHF** → Raw EMG of the Anterior Deltoid on the Frontal Plane
  - **DAHF_fft** → FFT transformed signal of DAHF  
  - **DEHF** → Raw EMG of the Lateral Deltoid on the Frontal Plane
  - **DEHF_fft** → FFT transformed signal of DEHF

---

### Saggital Shoulder
*(Shoulder Flexion and Extension)*

- **Muscles recorded:**  
  - **Anterior Deltoid Sagittal Shoulder / Deltoides Anterior Hombro Sagital** (DAHS)  
  - **Posterior Deltoid Sagittal Shoulder / Deltoides Posterior Hombro Sagital** (DPHS)  
- **Signal naming convention:**  
  - **DAHS** → Raw EMG of the Anterior Deltoid on the Sagittal Plane
  - **DAHS_fft** → FFT transformed signal of the DAHS
  - **DPHS** → Raw EMG of the Posterior Deltoid on the Sagittal Plane
  - **DPHS_fft** → FFT transformed signal of DPHS


---

### Tranverse Shoulder
*(Horizontal Adduction and Abduction of the Shoulder)*

- **Muscles recorded:**  
  - **Lateral Deltoid Transverse Shoulder / Deltoides Externo Hombro Transversal** (DEHT)  
  - **Posterior Deltoid Transverse Shoulder / Deltoides Posterior Hombro Trasnversal** (DPHT)  
- **Signal naming convention:**  
  - **DEHT** → Raw EMG of the Lateral Deltoid on the Transverse Plane
  - **DEHT_fft** → FFT transformed signal of DEHT
  - **DPHT** → Raw EMG of the Posterior Deltoid on the Transverse Plane
  - **DPHT_fft** → FFT transformed signal of DPHT



Each signal is stored in CSV format for easy integration with Python, MATLAB, and deep learning pipelines.  



## 2. simulation

The robotic arm simulation was performed in Matlab. The .mlx file is attached in the folder under the name “Simulación de un brazo robótico bioinspirado”.

Transformation matrices based on Denavith-Hartenberg parameters were used to develop the simulation. Efforts were also made to ensure that the final position vector, referred to in the simulation as q_brazo_final, functioned similarly to the vector used by the Kinova 3Gen Lite arm, in order to guarantee that the simulation and the arm's operation would function identically.


## 3. models


## 4. figures


## 5. documents




Author: Postelnecu Ioana
Subject: Bachelor Thesis
Domain of Study for Bachelor: Cybersecurity Analysis with AI (IPS through ai w suricata and zeek), afterwards transferred onto a Raspberry Pi 5b
Faculty of Engineering in Foreign Languages
Electronics, Telecommunications and Information of Technology - Applied Electronics
Politehnica University of Bucharest

mai am ultima parte rezolvat cu predict in server.py si jucat cu interfata.. vazut ce grafice/detalii sa extrag din proiect...
gandit ce le zic lui Marian si Bujor
uitat dupa Raspberry!!!!! respectiv ce componente sa-i iau(daca tot fac asa ceva)
aranajat codul (comentarii, etc)
inteles cat folosesc din dataset si cum exact
eu vad ce se intampla, dar nu trebuie activat la un trafic in care se intampla activ ceva?
rezolvat documentatia pt nn
brainstorming : idei pt adaugiri? Ok este utilizabil, dar ce-i mai fac? din ce am eu pana-n punctul asta?!
Cum explic AI din titlul Licentei? trb ceva concret, ca altfel mi-o iau rau de tot

16.01
rezolvat cu predict
jucat cu threshold pt a avea anomalii(modul prin care trece prin ele?)
jucat cu interfata
luat la disecat variantele de dataseturi? csvurile?!?!
mai fac cu un dataset? sa vad cum se descurca in aia?!



teoria licentei pana-n mom de fata: 20.01.2026
 
 AI-ENHANCED NETWORK INTRUSION DETECTION SYSTEM
Neural Networks Course – Project Submission


PROJECT OVERVIEW

This project implements an AI-based Network Intrusion Detection System (NIDS) using a
Neural Network Autoencoder to identify anomalous network traffic.

The system is designed to:
- learn patterns of normal network behavior,
- detect deviations from this learned behavior,
- expose detection results through a real-time inference API and a monitoring dashboard.

The project represents a practical application of Neural Networks in the field of
cybersecurity and constitutes the Neural Network component of the author’s Bachelor-level
research project.


USE OF ARTIFICIAL INTELLIGENCE (NEURAL NETWORK COMPONENT)

Type of AI Used

The core Artificial Intelligence component of this project is a feed-forward Autoencoder
Neural Network.

The Autoencoder is trained using unsupervised learning, relying exclusively on normal
(non-attack) network traffic. No attack labels are used during training.

The model learns a compact internal representation of normal network behavior and is later
used to detect anomalies based on reconstruction error.


AUTOENCODER ARCHITECTURE AND TRAINING

An Autoencoder is a neural network composed of:
- an encoder, which compresses the input feature vector into a latent representation;
- a decoder, which reconstructs the original input from that representation.

Training is performed by minimizing a reconstruction loss function
(Mean Squared Error) using gradient-based optimization.

After training:
- normal traffic produces low reconstruction error;
- anomalous traffic produces high reconstruction error.


WHY THE LEARNING IS UNSUPERVISED

The Autoencoder is trained using only normal network traffic, without any attack labels.

This means:
- the model does not learn predefined attack signatures;
- anomalies are detected as deviations from learned normal behavior;
- the model can detect previously unseen (zero-day) attacks.

This strictly qualifies the neural network training stage as unsupervised learning.


HYBRID NATURE OF THE OVERALL SYSTEM

While the neural network training itself is unsupervised, the overall detection pipeline
is hybrid.

The complete system includes:
- unsupervised learning for representation learning (Autoencoder);
- statistical thresholding on reconstruction error;
- evaluation using labeled datasets for validation and performance measurement.

Thus:
- AI learning stage: unsupervised
- decision and evaluation logic: supervised/statistical

This hybrid design reflects common real-world intrusion detection systems.


WHY THIS SYSTEM QUALIFIES AS ARTIFICIAL INTELLIGENCE

This project qualifies as an AI system because:
- it uses a trainable neural network with learned weights and biases;
- detection decisions are based on model inference, not predefined rules;
- the system generalizes to unseen data;
- anomalous behavior emerges from learned representations.


DATASET USED

Primary Dataset

The system is trained and evaluated using the UNSW-NB15 network intrusion dataset.

Dataset characteristics:
- realistic modern network traffic;
- contains both normal and attack flows;
- widely used in academic intrusion detection research.

Dataset Source

The dataset can be obtained from:
- Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
- official UNSW dataset mirrors.

Dataset Usage in This Project

- Training: only normal traffic is used to train the Autoencoder.
- Validation and testing: mixed traffic (normal and attack) is used to evaluate detection.

Processed CSV feature files are located under:
data/processed/


PROJECT STRUCTURE (SIMPLIFIED)

Bachelor/
│
├── data/
│   ├── external/        Raw datasets (not included in submission)
│   ├── processed/       Processed CSV feature files
│   └── models/          Trained models and thresholds
│
├── src/
│   ├── features/        Feature extraction scripts
│   ├── models/          Neural network training scripts
│   └── realtime/        Real-time inference API and dashboard
│
├── logs/                Inference and metrics logs
├── requirements.txt
└── README.txt


ENVIRONMENT AND PLATFORM

Operating System:
- Linux (tested on Ubuntu)

Python Version:
- Python 3.10

Main Frameworks:
- TensorFlow / Keras (training)
- ONNX Runtime (inference)
- FastAPI (real-time API)
- Prometheus-compatible metrics

The project is Linux-based and is not officially supported on Windows.


VIRTUAL ENVIRONMENTS

Two Python virtual environments are used to separate training and inference concerns.

Training Environment (venv_tf):
Used for neural network training and TensorFlow dependencies.

Creation:
python3 -m venv venv_tf
source venv_tf/bin/activate
pip install -r requirements.txt

API Environment (venv_api):
Used for real-time inference, API, and dashboard.

Creation:
python3 -m venv venv_api
source venv_api/bin/activate
pip install -r requirements.txt

Virtual environments are not included in the submission archive.


RUNNING THE PROJECT

Start the real-time API and dashboard:

source venv_api/bin/activate
uvicorn src.realtime.server:app --host 127.0.0.1 --port 8000 --reload

Open in a browser:
http://127.0.0.1:8000


AVAILABLE ENDPOINTS

/predict  - run inference on a feature vector
/alerts   - view detected anomalies
/metrics  - performance and system metrics
/health   - system status


NOTES FOR EVALUATION

The Artificial Intelligence component of this project is the Autoencoder Neural Network.

The project demonstrates:
- unsupervised neural network training,
- anomaly detection based on reconstruction error,
- real-time inference,
- application of neural networks to cybersecurity.

This fulfills the requirements of a Neural Networks course project while also integrating
into a broader Bachelor-level research system.



23.02.2026
pun metadata de la dport si sport -> le pun pe web si incerc sa pun explicatii? are sens? sa vad logica si practic de ce? gen, o explicatie?

24.02.2025
nu are sens pe Raspberry, pot scana direct din laptop? 
interfata e ok. (poate doar niste explicatii ca la prosti,initial de inteles) - nu las 


ce face?
de ce?
cati virusi detecteaza?
criteriu de alegere? (daca este anomalie sau nu)
trafic intern sau live?
utilizeaza dataset?
cate pachete?

cum? 
brut si direct (pleaca de la x si se duce la y. ce scaneaza, de unde scaneaza, cum decide anomalia, de unde si pana unde considera ca anomalia este de tip x sau daca este anomalie sau nu).
cum am antrenat ai-ul?
de ce autoencoder?
schema efectiva de rulare + 
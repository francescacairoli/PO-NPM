# Neural Predictive Monitoring under Partial Observability (PO-NPM)
Neural Predictive Monitoring (NPM) for hybrid systems under noisy and partially observable measurements.

Datasets are uploaded here: https://mega.nz/folder/RGIlFQAR#jhu8Gu-zAaIcMMi0jkjYXA

Francesca Cairoli, Luca Bortolussi, Nicola Paoletti
Submitted to **Runtime Verification 21**

## Abstract
We consider the problem of predictive monitoring (PM), i.e., predicting at runtime future violations of a system from the current state. We work under the most realistic settings where only partial and noisy observations of the state are available at runtime. Such settings directly affect the accuracy and reliability of the reachability predictions, jeopardizing the safety of the system. 
In this work, we present a learning-based method for PM that produces accurate and reliable reachability predictions despite partial observability (PO).
We build on Neural Predictive Monitoring (NPM), a PM method that uses deep neural networks for approximating hybrid systems reachability, and extend it to the PO case. We propose and compare two solutions, an *end-to-end* approach, which directly operates on the rough observations, and a *two-step* approach, which introduces an intermediate state estimation step. Both solutions rely on conformal prediction to provide 1) probabilistic guarantees in the form of prediction regions and 2) sound estimates of predictive uncertainty. We use the latter to identify unreliable (and likely erroneous) predictions and to retrain and improve the monitors on these uncertain inputs (i.e., active learning). Our method results in highly accurate reachability predictions and error detection, as well as tight prediction regions with guaranteed coverage. 

## Code structure
`RV21/` contains the code for all the experiments submitted to RV21.

- **End-to-end experiments**:
`python execute_iterative_active_po_nsc.py --model_name "NAME"`

- **Two-step experiments**:
`python execute_iterative_active_comb_po_nsc.py --model_name "NAME"`

### Case studies:
- Inverted Pendulum: "NAME" = "IP"
- Spiking Neuron: "NAME" = "SN"
- LaubLoomis: "NAME" = "LALO"
- Coupled Vand Der Pol: "NAME" = "CVDP"
- Triple Water Tank: "NAME" = "TWT"
- Helicopter: "NAME" = "HC"


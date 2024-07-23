
## Contents


- [**MTS Anomaly Surveys**](#Surveys)
  [**MTS Anomaly ideas**](#Ideas)
  - [**Diffusio-Models**](#Diffusion Models)
  - [Differential Geometry + Lie Groups](#diffgeo)
  - [Information Geometry](#infogeo)
  - [Topology](#topology)


<br /><br />

<a name="Surveys" />

#### Links of MTS Anomaly Srveys
*[**TimeSe{AD}: Benchmarking Deep Multivariate Time-Series Anomaly Detection,Wagner, Dennis and Michels, Tobias and Schulz, Florian CF and Nair, Arjun and Rudolph, Maja and Kloft, Marius,2023**](https://github.com/wagner-d/TimeSeAD/tree/master)
*[Time Series Anomaly Detection using Diffusion-based Models,Florin Brad,2023]
*[Graph Anomaly Detection in Time Series: A Survey N. Armanfard 2023]
*[TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection Marius Kloft 2023]
*[Deep Learning for Time Series Anomaly Detection: A Survey Mahsa Salehi 2022]
*[Navigating the Metric Maze: A Taxonomy of Evaluation Metrics for Anomaly Detection in Time Series M. Ruocco 2023]

<br /><br />

<a name="Ideas" />

####  MTS Anomaly ideas
*[**Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network,Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, Dan Pei,2019**](https://github.com/NetManAIOps/OmniAnomaly)

The OmniAnomaly model is a stochastic recurrent neural network (RNN) designed for robust anomaly detection in multivariate time series (MTS). It learns normal patterns in MTS data by capturing the temporal dependence and stochasticity using key techniques such as:

Stochastic Variable Connection: Introduces uncertainty in the latent space, enabling the model to adapt to varying patterns in the data.
Planar Normalizing Flow: Provides a flexible and differentiable way to model complex distributions, allowing the model to effectively capture normal patterns.
The core idea is to:

Learn robust representations of normal patterns in MTS data.
Reconstruct input data using these representations.
Use reconstruction probabilities to determine anomalies.
OmniAnomaly demonstrates robust performance across various devices and datasets, including:

Server Machine Dataset (SMD): 28 machines with 38 dimensions, 13.13% anomaly ratio.
Mars Science Laboratory Rover (MSL) dataset: 27 machines with 55 dimensions, 10.72% anomaly ratio.
Soil Moisture Active Passive Satellite (SMAP) dataset: multivariate time series data from spacecraft, used for anomaly detection.
The model’s effectiveness lies in its ability to:

Detect anomalies from predictable, unpredictable, periodic, aperiodic, and quasi-periodic time-series.
Handle short and long time-series data (30-500 samples).
Provide interpretations for detected entity anomalies based on reconstruction probabilities of constituent univariate time series.
Overall, OmniAnomaly presents a robust solution for multivariate time series anomaly detection, leveraging stochastic recurrent neural networks to learn and utilize robust representations of normal patterns.

*[**BTAD: A binary transformer deep neural network model for anomaly detection in multivariate time series data,Mingrui Ma, Lansheng Han, Chunjie Zhou,2023** ](https://www.sciencedirect.com/science/article/abs/pii/S1474034623000770)

Key Features:

Binary Transformer Structure: BTAD employs a binary Transformer structure, which allows it to differentiate between unary and multivariate time series datasets. This adaptability enables the model to optimize its efficiency and accuracy for various dataset types.
Attention-Based Sequence Encoders: The model uses attention-based sequence encoders to swiftly perform inference, incorporating knowledge of broader temporal trends in the data. This mechanism enables BTAD to identify anomalies by recognizing deviations from normal patterns.
Improved Adaptive Multi-Head Attention Mechanism: BTAD’s attention mechanism is improved to infer trends in each meta-dimension of multivariate time series data in parallel, enhancing its ability to detect anomalies.

*[**AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme,Yungi Jeong, Eu-Hui Yang, Jung Hyun Ryu, Imseong Park, Myung-joo Kang,2023** ](https://github.com/Jhryu30/AnomalyBERT)

AnomalyBERT is a self-supervised transformer-based model for time series anomaly detection using a data degradation scheme. This approach surpasses previous state-of-the-art methods on five real-world benchmarks.

Key Features:

Data Degradation Scheme: The model degrades normal data to simulate anomalies, enabling self-supervised learning without labeled anomaly data.
Transformer Architecture: AnomalyBERT employs a transformer-based architecture, which effectively captures temporal dependencies and patterns in time series data.
Effective Anomaly Detection: The model demonstrates superior performance in detecting anomalies in complex time series data, outperforming previous state-of-the-art methods.
Availability:

The AnomalyBERT code is available on GitHub, allowing for easy reproduction and modification of the model.
The model has been trained on various datasets, including SMAP and UCR, and can be fine-tuned for specific use cases.
Comparison to Other Methods:

While other transformer-based models, such as AnoFormer and Donut, have been proposed for time series anomaly detection, AnomalyBERT’s self-supervised approach and data degradation scheme set it apart as a more effective and efficient method.

*[Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy,Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long,2021**](https://github.com/thuml/Anomaly-Transformer(
Unsupervised detection of anomaly points in time series is a challenging problem, which requires the model to learn informative representation and derive a distinguishable criterion. In this paper, we propose the Anomaly Transformer in these three folds:

An inherent distinguishable criterion as Association Discrepancy for detection.
A new Anomaly-Attention mechanism to compute the association discrepancy.
A minimax strategy to amplify the normal-abnormal distinguishability of the association discrepancy.

*[**Graph Neural Network-Based Anomaly Detection in Multivariate Time Series,Ailin Deng, Bryan Hooi,2021**](https://github.com/d-ailin/GDN)


*[**USAD: UnSupervised Anomaly Detection on Multivariate Time Series,Julien Audibert, Pietro Michiardi, F. Guyard, Sébastien Marti, Maria A. Zuluaga,2020**](https://hwk0702.github.io/treatise%20review/2021/02/15/USAD/)

*[**A time series anomaly detection method based on series-parallel transformers with spatial and temporal association discrepancies,Guangyao Zhang 2024**](https://www.sciencedirect.com/science/article/abs/pii/S0020025523015633)
*[**Global–Local Association Discrepancy for Multivariate Time Series Anomaly Detection in IIoT,Tie Qiu,2024**]
[Local-Adaptive Transformer for Multivariate Time Series Anomaly Detection and Diagnosis,Ruyi Zhang,2023]



A filter-augmented auto-encoder with learnable normalization for robust multivariate time series anomaly detection,Chun Xiao 2023







Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction D. Boning 2023




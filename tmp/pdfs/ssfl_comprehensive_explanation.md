# Semisupervised Federated Learning for IoT Intrusion Detection

## A comprehensive explanation of Zhao et al. (IEEE Internet of Things Journal, 2023)

This report explains the paper "Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things" by Ruijie Zhao, Yijun Wang, Zhi Xue, Tomoaki Ohtsuki, Bamidele Adebisi, and Guan Gui. It is intended to be readable without constant reference to the original article while still preserving the paper's technical logic, experimental details, numerical results, and important limitations.

## Executive summary

The paper proposes a federated-learning intrusion-detection system called SSFL, short for semisupervised federated learning. Its goal is to let many Internet of Things devices collaboratively train an attack detector without transferring their private network traffic or repeatedly transferring complete neural-network models. Each participating device, called a client, trains a classifier on its own labeled traffic. It then predicts labels for a common unlabeled traffic dataset. Because a client may have seen only a few attack categories, the method introduces a second neural network called a discriminator. The discriminator estimates whether a shared example is familiar to that client. Predictions on unfamiliar examples are rejected rather than sent as if they were reliable.

Every accepted prediction is converted to a hard class label. The client sends these compact labels to a central server. For each shared example, the server ignores abstentions, counts the remaining votes, and selects the majority class. The resulting global pseudo-labels are broadcast to all clients. Clients then train on the common examples with those global labels. The process repeats, allowing knowledge possessed by one client to reach clients that never observed the same attack category privately.

This design aims to solve three problems simultaneously. First, ordinary federated learning can still leak information through gradients or model parameters. SSFL avoids sending those objects. Second, IoT data are strongly non-IID: devices see different traffic, class proportions, and attacks. SSFL uses familiarity filtering and voting to prevent uninformed clients from corrupting global supervision. Third, transferring a large model in every round is expensive. SSFL transmits small integer labels rather than millions of floating-point parameters.

The authors evaluate SSFL on a reduced version of the N-BaIoT botnet traffic dataset under three non-IID client partitions. SSFL reaches accuracies of 87.40%, 86.70%, and 84.22%. It outperforms conventional federated learning, federated distillation, and an earlier distillation-based semisupervised approach. Its reported communication cost at top accuracy is approximately 0.55 MB, 0.49 MB, and 0.47 MB in the three scenarios, compared with 711.43 MB, 1745.06 MB, and 1700.12 MB for conventional federated learning. Ablation experiments show that the familiarity discriminator is the critical component: removing it can reduce accuracy to roughly 10%-40%.

The paper therefore demonstrates a compelling filtered pseudo-labeling mechanism for communication-efficient federated intrusion detection. It does not, however, prove formal privacy, test malicious clients, evaluate real IoT hardware, or establish generalization beyond one source dataset. Its strongest defensible conclusion is that sharing carefully filtered labels can be substantially more effective than sharing every client's raw predictions when local data are severely heterogeneous.

## 1. The problem the paper is trying to solve

The work lies at the intersection of Internet of Things security, machine-learning intrusion detection, and privacy-preserving collaborative learning. IoT networks contain devices such as cameras, doorbells, routers, thermostats, sensors, and appliances. These devices often have limited processors, restricted memory, long deployment lifetimes, inconsistent patching, and weak built-in defenses. Attackers can exploit those properties to compromise devices, create botnets, scan networks, steal information, or launch denial-of-service attacks.

An intrusion detection system monitors traffic and decides whether a record is benign or malicious. A multiclass detector may also identify a specific attack behavior. Modern deep neural networks can learn this task from numerical traffic features, often more effectively than manually designed rules. The difficulty is that a good neural model needs representative training data from many devices and attack conditions.

Centralized deep learning collects all traffic on one server. Although technically convenient, centralization exposes private information. Network activity can reveal when a device is used, which services it contacts, its internal configuration, communication rhythms, and characteristics of the people or organization operating it. Uploading all records also creates an attractive central target. Data protection rules and organizational policies may prohibit such collection.

Federated learning offers a partial solution. Every device keeps its raw records, trains a model locally, and sends a model update to a server. The server aggregates updates and returns a global model. Raw traffic no longer needs to leave the device. Yet the paper emphasizes that this arrangement has three remaining weaknesses.

### 1.1 Model updates can leak information

Gradients and parameter changes are functions of the examples used to calculate them. Gradient-inversion and reconstruction attacks can sometimes recover properties of training data from an uploaded update. Therefore, "the raw data were not sent" is not equivalent to "the data cannot leak." SSFL responds by avoiding parameter and gradient exchange entirely. Clients communicate predictions on shared nonprivate examples.

This choice reduces exposure to attacks that specifically require model updates. It is not a complete privacy proof. Labels and abstention patterns can still reveal something about a client's expertise or local distribution. The distinction between reduced attack surface and formal privacy is important throughout the paper.

### 1.2 IoT data are non-IID

IID means independent and identically distributed. An IID federation would give every client approximately the same distribution of benign traffic and attack classes. Real IoT networks are not like that. A camera produces different traffic from a doorbell. One device may observe mostly benign use, another may experience a Mirai attack, and another may never encounter most attack classes. Clients can also have very different quantities of data.

This statistical heterogeneity creates label skew, feature skew, device-domain skew, and quantity skew. A local model becomes a specialist. If it has seen only benign traffic and one attack, its output on an unfamiliar attack is not trustworthy. Neural networks nevertheless always produce a class probability. A naive federated distillation method may mistake that forced guess for knowledge and average it with valid predictions.

The principal logic of SSFL is to permit abstention. Before a client contributes a label, a discriminator asks whether the example resembles the client's area of competence.

### 1.3 Model exchange is expensive

Deep models can contain millions of parameters. Transmitting the full model from every client during hundreds of rounds consumes bandwidth and energy. It also creates latency. Communication becomes especially expensive as the number of clients grows. In conventional federated learning, choosing a larger neural architecture automatically increases network cost.

SSFL sends one discrete label per common example. Its communication burden depends primarily on the size of the shared dataset and number of clients, not the number of model parameters. This separation means a client can theoretically use a larger or even different architecture without changing the label message format.

## 2. What the authors aim to achieve

The authors organize the desired solution around security, accuracy, and efficiency.

Security means that private traffic remains local and that the protocol avoids exchanging gradients or model parameters that could be used in known reconstruction attacks. Accuracy means the collaboration must remain useful when local distributions are severely non-IID and clients lack entire classes. Efficiency means the messages must be small enough to make repeated training plausible in an IoT setting.

The proposed answer is not a single trick. It combines four ideas: shared unlabeled examples, client-side classifiers, learned familiarity discrimination, and server-side hard-label voting. The common dataset gives every client a shared reference. Classifiers attach tentative knowledge to that reference. Discriminators remove likely uninformed predictions. Voting converts the surviving opinions into global pseudo-labels. These labels then become training targets for all participants.

## 3. Concepts needed to understand the paper

### 3.1 Supervised learning

In supervised learning, the model receives an example x_i and a correct label y_i. Here, x_i is a numerical description of network traffic and y_i identifies benign traffic or an attack category. A classifier F(x_i | w) outputs probabilities, where w represents model parameters. Cross-entropy measures disagreement between the output and true label. An optimizer changes w to reduce that loss.

Each SSFL client performs ordinary supervised training on its private labeled dataset. This creates the initial local expertise from which collaboration begins.

### 3.2 Semisupervised learning and pseudo-labels

Semisupervised learning combines labeled and unlabeled data. SSFL has private labeled data at the clients and a common unlabeled dataset available to the federation. The open samples initially have no usable training labels. Clients collaboratively manufacture labels for them. A machine-generated target is called a pseudo-label.

Pseudo-labeling can be powerful because it turns inexpensive unlabeled data into extra supervision. It can also create a feedback loop of errors. If incorrect pseudo-labels are accepted, the model trains itself to reproduce those mistakes, becomes more confident in them, and may generate even worse labels in the next round. SSFL's filtering and voting mechanisms are designed to control that risk.

### 3.3 Conventional federated learning

In a conventional FedAvg cycle, the server initializes a model, clients download it, each client trains on local data, clients upload updated parameters, and the server computes a weighted average. If client k has parameters w_k and N_k samples, the update has the general form:

    w_server = sum over k of (N_k / N_total) * w_k

Clients with more data have more influence. The global parameters are distributed again and the loop repeats.

FedAvg shares parameter tensors. SSFL shares labels associated with common examples. That change is responsible for both the privacy and communication differences.

### 3.4 Knowledge distillation

Knowledge distillation transfers behavior from a teacher to a student. The teacher predicts outputs on examples, and the student learns to match those outputs. Soft labels retain a probability for every class, such as [0.03, 0.82, 0.10, 0.05]. A hard label retains only the winning class.

Soft labels contain uncertainty and similarities between classes, sometimes called dark knowledge, but require multiple floating-point numbers. Hard labels contain less information but are cheap to send.

In SSFL, the teacher is effectively an ensemble of clients. Their filtered votes produce global labels. Every client and a server model can then act as students. Because the final targets are hard labels, the method is also naturally described as filtered federated pseudo-labeling.

### 3.5 Non-IID learning

A non-IID federation violates the assumption that every client is a random sample of the same population. In this paper the clients differ in class coverage, class proportions, sample counts, and originating device. These differences cause local models to move toward different solutions. They also make local predictions unreliable on portions of the shared dataset.

The severity of this problem explains why the discriminator is central rather than optional. A client must estimate not only what label it predicts, but whether it has enough relevant experience to deserve a vote.

## 4. Baseline approaches

### 4.1 Conventional FL

Conventional FL directly averages uploaded model parameters. It is a strong accuracy baseline but has high communication cost, slow convergence under heterogeneity, and exposure to parameter- or gradient-based leakage attacks. In the paper it ultimately achieves reasonable accuracy, but its payload is hundreds or thousands of megabytes.

### 4.2 Federated distillation

Federated distillation, abbreviated FD, avoids model transfer. Each client divides its private data by class and computes an average model output for each locally available class. It uploads these class-average logits. The server aggregates them and returns global averages.

This is efficient, but it works poorly when clients lack many classes. A client's averaged knowledge remains tied to its local labels, so collaboration may provide little information about missing categories. The paper's FD results are extremely weak under the tested non-IID partitions.

### 4.3 DS-FL

DS-FL is an earlier distillation-based semisupervised federated method. Every client predicts a probability vector for every example in a common unlabeled dataset. The server averages those vectors and applies a temperature-adjusted softmax. A temperature below one sharpens the distribution, lowers entropy, and makes the largest probabilities more decisive. Global logits are returned to clients for training.

The common examples improve on FD because they allow knowledge to be attached to specific inputs rather than only to class averages. However, DS-FL trusts all clients on all examples. When local data are strongly non-IID, many predictions come from models that never learned the relevant class or region. Averaging those outputs can create a bad teacher. SSFL's discriminator and abstention value are designed specifically to correct this failure.

## 5. The main idea of SSFL

Imagine the clients as specialists. All specialists inspect the same unlabeled traffic record. Before giving an answer, each asks whether the record resembles anything it understands. Unfamiliar specialists abstain. The remaining specialists vote. Their majority answer becomes a temporary global label, and everyone studies from it.

The complete information path is:

    private labels -> local expertise -> predictions on shared data
    -> familiarity filtering -> hard-label voting
    -> global pseudo-labels -> additional training

This loop repeats. Ideally, each round improves the classifiers, which improves later votes, which creates better pseudo-labels. The discriminator is the gate that prevents this positive feedback loop from turning into error amplification.

## 6. Detailed SSFL procedure

Assume K clients and L traffic classes. Client k owns a private labeled dataset D_private_k, a classifier with parameters w_classifier_k, and a discriminator with parameters w_discriminator_k. All clients can access the same unlabeled open dataset D_open.

### 6.1 Train the local classifier

Every client trains its classifier on private labeled data using a standard gradient update:

    w_classifier_k <- w_classifier_k
        - learning_rate * gradient(cross_entropy(prediction, true_label))

This creates a model that is competent on its own local distribution. Because the distribution is non-IID, it may be excellent on a few classes and unreliable elsewhere.

### 6.2 Predict open examples and measure confidence

The classifier processes every shared example x_open_j. It produces a vector of class probabilities. The confidence score is the largest probability:

    confidence(k,j) = max(classifier_k(x_open_j))

If the output is [0.02, 0.04, 0.84, 0.03, 0.07], confidence is 0.84. A low maximum suggests uncertainty or unfamiliarity. Confidence alone is imperfect because neural networks can be confidently wrong, so the score is used to construct a discriminator rather than serving as the final rejection rule.

### 6.3 Construct familiarity training data

The discriminator has two classes: familiar and unfamiliar. Every private labeled example is marked familiar because it belongs to the client's known distribution. Shared examples whose classifier confidence is below a boundary theta are provisionally marked unfamiliar.

Instead of a universal fixed theta, the main implementation uses the median of that client's confidence values over all open examples. This adapts the threshold to each local model. A fixed value of 0.9 may reject almost everything for an underconfident model and almost nothing for an overconfident model. The median supplies a relative boundary even when clients have very different calibration.

These labels are heuristic. A difficult but familiar example may have low confidence, while an unfamiliar example may receive a confidently wrong answer. The discriminator is trained from noisy self-generated targets.

### 6.4 Train the discriminator

The client trains a binary CNN to separate familiar from unfamiliar examples. It uses almost the same backbone as the multiclass classifier, but its final layer has two outputs.

The intended benefit is that a neural discriminator can learn a feature-space boundary more informative than a single confidence number. It may recognize that an example resembles the client's known traffic even if the classifier's confidence is moderate, or reject an example whose raw confidence would otherwise pass.

This resembles learned in-distribution versus out-of-distribution detection. It is not a perfect OOD detector, because its familiarity labels are not ground truth and the private and open data may differ for reasons unrelated to class knowledge.

### 6.5 Filter and abstain

The discriminator classifies every open sample. If the sample is familiar, the client keeps its classifier's most likely class. If it is unfamiliar, the client assigns -1.

The value -1 is not a network-traffic class. It means "I abstain because this example is outside my trusted expertise." This small design choice is the conceptual heart of the method. A neural network always returns some class, but SSFL does not force the federation to interpret every output as knowledge.

### 6.6 Upload hard labels

Each client sends one hard value for each open example: a class ID from 0 through L-1, or -1 for abstention. It does not send private examples, model parameters, gradients, or full probability vectors.

Hard labels greatly reduce payload size. With eleven classes, a soft output contains eleven floating-point numbers per example. A hard output can be encoded as a compact integer. The ablation results later show that soft labels provide nearly the same accuracy in this experiment, making their additional communication difficult to justify.

### 6.7 Vote on the server

For each open sample, the server ignores -1 entries and counts votes for each real class. The class with the most votes becomes the global hard label:

    global_label(j) = argmax_l(number of accepted votes for class l)

Voting reduces isolated errors after the discriminator has removed many unreliable contributions. If the survivors are usually correct and their errors are not perfectly correlated, majority agreement is a useful teacher signal.

Voting can also amplify a systematic error. If most participating clients are wrong in the same way, the majority is wrong. This explains the important interaction observed in the ablation study: voting is helpful with discrimination but can be harmful when uninformed clients are allowed to participate.

The paper does not fully specify how ties, all-abstention cases, or very small vote counts are handled. A production implementation should define a minimum vote count and agreement threshold.

### 6.8 Broadcast and pseudo-label training

The server broadcasts the global label sequence. Every client trains its classifier on the shared examples paired with those labels. The server also trains a model on the pseudo-labeled open data for evaluation.

This stage transfers class knowledge. A client that never privately observed Mirai SYN traffic may learn from open examples labeled by clients familiar with that behavior. The raw private Mirai records never move; only knowledge expressed on the shared examples moves.

### 6.9 Repeat for multiple rounds

The system alternates private supervised training and global pseudo-label training. Better client models should produce more accurate familiarity assessments and votes. Better global labels should then improve every client. The experiments run up to 200 communication rounds and show that SSFL obtains most of its performance by approximately rounds 100-150.

## 7. CNN architecture

Every N-BaIoT record contains 115 numerical features calculated over five time windows: 100 milliseconds, 500 milliseconds, 1.5 seconds, 10 seconds, and one minute. The feature vector is rearranged into a 23 by 5 matrix. The five columns represent the temporal windows, while the rows represent measurements repeated across those windows.

The model uses eight one-dimensional convolutional layers. Layers 1-4 use 64 filters with kernel size 3 and stride 1. Layers 5-6 use 128 filters with kernel size 3 and stride 1. Layers 7-8 use 128 filters with kernel size 3 and stride 2. The convolutional representation is flattened into a 128-unit fully connected representation and an output layer.

The classifier has eleven output neurons for the full traffic-class task. The discriminator has two outputs. Apart from the final layer, they share the same general architecture.

The authors argue that convolution promotes interaction among adjacent time windows and traffic features. Their comparison supports the CNN choice. When trained with SSFL, the CNN reaches 87.40%, 86.70%, and 84.22% accuracy. The MLP reaches 82.78%, 82.85%, and 81.28%. The LSTM reaches 72.24%, 69.69%, and 60.80%.

Because SSFL does not average model tensors, clients could theoretically use different architectures. They only need to predict the same label space for the same open examples. The paper mentions this flexibility but does not test heterogeneous client models.

There are minor reporting inconsistencies. Table I identifies an input batch dimension of 80, while the experimental setup says batch size 100. The prose also alternates between two and three fully connected layers, while the architecture table appears to show one hidden fully connected layer plus the output. These ambiguities affect exact reproduction but not the logic of the communication protocol.

## 8. Dataset and preprocessing

The experiments use N-BaIoT, a public IoT botnet traffic dataset collected from nine devices, including devices such as cameras and doorbells. Seven devices contain eleven categories: one benign class and ten attacks. The other two contain one benign class and five attacks.

The attack labels cover Gafgyt and Mirai behaviors. The full set includes Gafgyt combo, junk, scan, TCP, and UDP, together with Mirai ACK, scan, SYN, UDP, and UDP-plain behavior.

### 8.1 Mini-N-BaIoT

The original dataset is large, so the authors construct mini-N-BaIoT. They select 1,000 records from every available device-class combination. Seven devices have eleven classes and two have six, giving:

    7 * 11 + 2 * 6 = 89 device-class combinations

This implies approximately 89,000 selected records if every combination contributes the stated 1,000 examples. The records are divided into 70% private labeled data, 10% shared open data, and 20% test data. The sets are disjoint. The implied approximate sizes are 62,300 private records, 8,900 open records, and 17,800 test records. These counts are inferred from the selection description rather than reported as a separate table.

The open examples originally come from the same N-BaIoT source but their labels are withheld from training. This produces good domain alignment between private, open, and test sets. It is also a favorable condition compared with deployment, where a public open dataset may differ from the current network.

### 8.2 Normalization and reshaping

Features are min-max normalized to the range zero to one. This prevents measurements with large numerical ranges from dominating those with smaller ranges and stabilizes optimization. The paper does not clearly state whether scaling bounds are learned only from training data. If test data influence normalization, that would introduce a mild form of test-distribution leakage.

After normalization, each 115-element vector is reshaped into the 23 by 5 representation used by the CNN.

## 9. Three non-IID scenarios

### 9.1 Scenario 1: 27 shard-based clients

For each device, private data are sorted by class and divided into twice as many shards as clients. Each client receives two shards. There are three clients per device, producing 9 * 3 = 27 clients. Sorting before sharding creates narrow class coverage. Every client still receives data from only one physical device.

### 9.2 Scenario 2: 89 highly fragmented clients

The second scenario uses a number of clients equal to the number of classes available for each device. Seven devices contribute eleven clients and two contribute six, for 89 total. The same data are spread across more clients, so each client has fewer examples and generally fewer categories. This creates severe label fragmentation and tests scalability with a larger federation.

### 9.3 Scenario 3: 89 Dirichlet clients

The third scenario also has 89 clients but uses a Dirichlet allocation with concentration alpha = 0.1. A small alpha creates highly uneven class proportions and quantities. Some clients become dominated by a few labels, while others receive different and imbalanced mixtures. The paper considers this the most challenging scenario.

Across all scenarios, private, open, and test data are disjoint. Clients differ in both the number and categories of records, and no client mixes traffic from different physical devices.

## 10. Training and evaluation setup

The implementation uses Python 3.7 and PyTorch 1.9.0. Training uses Adam, a learning rate of 0.0001, and five local epochs per communication round. The main setup reports batch size 100, although the architecture table reports 80. The experiments run on an Intel Core i9-11900K, 64 GB RAM, and an NVIDIA RTX 3090 GPU.

The paper reports accuracy, precision, recall, and F1 score. It gives the conventional definitions:

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = 2 * precision * recall / (precision + recall)

The classification problem is multiclass, so precision and F1 require micro, macro, or weighted averaging. The paper does not clearly state which rule is used, making exact metric reproduction harder.

## 11. Main detection results

### 11.1 Scenario 1

| Method | Accuracy | F1 score | Precision |
|---|---:|---:|---:|
| Conventional FL | 86.11% | 85.13% | 91.29% |
| FD | 48.54% | 35.76% | 33.69% |
| DS-FL | 50.49% | 40.85% | 48.27% |
| MLP with SSFL | 82.78% | 81.29% | 87.45% |
| LSTM with SSFL | 72.24% | 66.77% | 74.50% |
| SSFL with CNN | 87.40% | 86.50% | 92.33% |

SSFL improves accuracy over conventional FL by 1.29 percentage points. This is a modest but consistent gain. Its much larger practical advantage is that it reaches the result with a tiny fraction of FL's communication. FD and DS-FL remain near 50%, showing that compact prediction exchange is not sufficient unless bad predictions are controlled.

### 11.2 Scenario 2

| Method | Accuracy | F1 score | Precision |
|---|---:|---:|---:|
| Conventional FL | 81.38% | 81.92% | 87.34% |
| FD | 20.12% | 8.53% | 10.37% |
| DS-FL | 53.53% | 43.95% | 57.45% |
| MLP with SSFL | 82.85% | 81.31% | 87.57% |
| LSTM with SSFL | 69.69% | 65.20% | 64.37% |
| SSFL with CNN | 86.70% | 84.95% | 91.73% |

SSFL's accuracy advantage over conventional FL grows to 5.32 points. FD collapses to 20.12% accuracy and 8.53% F1. DS-FL benefits from common example-level data but remains far behind SSFL. The result supports the need to distinguish informed from uninformed predictions when each client has little local class coverage.

### 11.3 Scenario 3

| Method | Accuracy | F1 score | Precision |
|---|---:|---:|---:|
| Conventional FL | 81.13% | 81.64% | 86.70% |
| FD | 53.06% | 43.27% | 44.31% |
| DS-FL | 20.01% | 7.31% | 10.96% |
| MLP with SSFL | 81.28% | 79.61% | 87.14% |
| LSTM with SSFL | 60.80% | 58.08% | 60.86% |
| SSFL with CNN | 84.22% | 82.47% | 90.17% |

SSFL exceeds FL by 3.09 accuracy points and DS-FL by 64.21 points. Its own result is lower than in Scenarios 1 and 2, confirming that it is not immune to extreme heterogeneity. It is substantially more robust than the tested distillation alternatives.

## 12. Confusion-matrix interpretation

The confusion matrices show more than 99% correct detection for benign traffic in every scenario. This matters because false alarms can make an IDS unusable. Most attack classes also have strong diagonal values in Scenarios 1 and 2.

The clearest confusion is between Gafgyt TCP and Gafgyt UDP. Both are denial-of-service behaviors and can produce similar statistical traffic patterns, so their feature representations overlap. Scenario 3 has more class-specific degradation, including a Mirai UDP-like category with noticeably lower recall. This is consistent with its extreme client imbalance.

The matrices indicate that aggregate accuracy is not produced only by correctly recognizing the benign class. At the same time, closely related attack subtypes remain difficult, suggesting that hierarchical classification or richer temporal features might help.

## 13. Training speed

The authors compare top-1 accuracy at rounds 10, 50, 100, 150, and 200. SSFL progresses quickly.

| Scenario | Round 10 | Round 50 | Round 100 | Round 150 | Round 200 |
|---|---:|---:|---:|---:|---:|
| SSFL Scenario 1 | 77.90% | 83.81% | 84.90% | 87.19% | 87.40% |
| SSFL Scenario 2 | 75.26% | 80.43% | 85.09% | 86.31% | 86.70% |
| SSFL Scenario 3 | 72.16% | 78.39% | 83.28% | 83.84% | 84.22% |

At only ten rounds, accuracy is already between 72% and 78%. Most final performance arrives by rounds 100-150.

Conventional FL is much slower. At round 200 it reaches 73.79%, 57.96%, and 63.35% in the three scenarios. These figures are lower than the final FL accuracies in the main result table. The likely explanation is that the main table reports the eventual top performance after longer training, while the round table stops at 200. The authors explicitly state that FL remains far from its maximum at that point.

FD and DS-FL often plateau very early at a poor value. Rapid convergence is beneficial only when the converged solution is useful. Their flat curves indicate that repeated communication cannot repair low-quality global supervision.

## 14. Communication results

### 14.1 Scenario 1

| Method | Cost at 50% | Cost at 75% | Cost at top accuracy | Top accuracy |
|---|---:|---:|---:|---:|
| Conventional FL | 15.81 MB | 216.29 MB | 711.43 MB | 86.11% |
| FD | - | - | 0.13 MB | 48.54% |
| DS-FL | 5.04 MB | - | 5.04 MB | 50.49% |
| SSFL | 0.01 MB | 0.02 MB | 0.55 MB | 87.40% |

At their reported top accuracies, FL uses approximately 711.43 / 0.55 = 1,294 times as much communication as SSFL.

### 14.2 Scenario 2

| Method | Cost at 50% | Cost at 75% | Cost at top accuracy | Top accuracy |
|---|---:|---:|---:|---:|
| Conventional FL | 137.04 MB | 514.14 MB | 1745.06 MB | 81.38% |
| FD | - | - | 0.02 MB | 20.12% |
| DS-FL | - | - | 22.63 MB | 53.53% |
| SSFL | 0.01 MB | 0.02 MB | 0.49 MB | 86.70% |

The top-performance ratio is approximately 3,561 to one. The large client population greatly increases repeated parameter uploads, while each SSFL vote remains compact.

### 14.3 Scenario 3

| Method | Cost at 50% | Cost at 75% | Cost at top accuracy | Top accuracy |
|---|---:|---:|---:|---:|
| Conventional FL | 110.69 MB | 473.75 MB | 1700.12 MB | 81.13% |
| FD | - | - | 0.19 MB | 53.06% |
| DS-FL | - | - | 46.35 MB | 20.01% |
| SSFL | 0.01 MB | 0.04 MB | 0.47 MB | 84.22% |

The corresponding top-performance ratio is approximately 3,617 to one.

The paper separately lists about 0.96 MB for distributing the open dataset. FL and FD do not require that initial distribution. Real deployments would also incur authentication, encryption, message headers, retransmissions, client coordination, and onboarding costs. The reported absolute numbers are therefore idealized payload accounting. The relative advantage should still remain large because a class ID is intrinsically much smaller than a neural model.

FD sometimes sends even less than SSFL, but its accuracy can be unusable. Communication efficiency must always be interpreted jointly with task utility. SSFL's contribution is the combination of low cost and competitive accuracy.

## 15. Ablation study

### 15.1 Removing the discriminator

Removing discrimination causes the largest collapse. Approximate final accuracies fall to about 40% in Scenario 1, 19% in Scenario 2, and 10% in Scenario 3. Clients then label examples from classes they do not understand, and those errors contaminate the global teacher.

This is the strongest evidence for the paper's central contribution. The discriminator is not a minor enhancement added to ordinary distillation; it is what makes shared pseudo-labeling viable under these partitions.

### 15.2 Removing voting

Without voting, results remain near 81%, 79%, and 79%-80%. The decline is meaningful but much smaller than removing the discriminator. Filtering is therefore the dominant contributor, while voting supplies additional correction and consensus.

### 15.3 Removing both

Without discrimination or voting, performance stays around 48%, 49%, and 20%. These values resemble failed naive distillation. They reinforce the conclusion that unlabeled data alone do not solve non-IID learning.

### 15.4 Simple confidence filtering

The authors replace the learned discriminator with a direct rule: reject predictions below the client's median confidence. This reaches only about 39%, 40%, and 10%. Therefore, the gain is not explained by mechanically discarding half the votes. The trained discriminator appears to learn useful information from the feature distribution.

### 15.5 Threshold choice

The paper compares fixed confidence thresholds 0.7, 0.8, and 0.9 against the client-specific median. The median performs best overall, with a particularly large advantage when clients have fewer examples and classes. A relative threshold adapts to differences in local confidence calibration.

The median also has a weakness: it imposes a relative split even if almost all predictions are reliable or almost all are bad. A calibrated uncertainty or conformal approach could adapt more meaningfully to absolute evidence.

### 15.6 Hard versus soft labels

Soft labels rounded to eight, six, four, or two decimal places achieve nearly the same accuracy as hard labels. Their communication costs differ dramatically. By round 200, eight-decimal soft labels require roughly 26 MB, 48 MB, and 13 MB across the three scenarios, while hard labels remain around half a megabyte.

The motivation for hard labels is therefore efficiency rather than an accuracy increase. In this dataset, uncertainty information contributes too little to justify its payload.

## 16. Why the method works

First, the open dataset creates a common language. Models with different parameters cannot be meaningfully averaged without architectural compatibility, but predictions on the same examples are directly comparable.

Second, clients contribute specialized rather than universal knowledge. A client does not need all eleven classes. It needs to be correct where it participates.

Third, abstention prevents negative transfer. A conventional classifier always outputs something, even outside its training distribution. SSFL introduces a mechanism for saying "I do not know."

Fourth, majority voting reduces individual mistakes after low-quality predictions have been removed. Filtering improves the voters; voting then improves the label.

Fifth, global pseudo-labels broaden every client's effective training distribution. Knowledge about a class can travel through shared examples without the private examples themselves moving.

These mechanisms form a positive loop. The discriminator protects early rounds from excessive noise. Global labels broaden local knowledge. Broader knowledge improves later predictions, which produces better labels.

## 17. What the paper achieves

The paper provides evidence for six main achievements. SSFL obtains the best accuracy, F1, and precision in all three tested scenarios. It substantially reduces communication relative to FedAvg. It learns useful classifiers much earlier in the round sequence. It remains effective under both shard-based and Dirichlet heterogeneity. Its ablation study identifies the discriminator as genuinely necessary. Finally, its prediction-sharing protocol is theoretically compatible with heterogeneous client architectures.

The most impressive result is not the small accuracy improvement over FL in Scenario 1. It is retaining or improving accuracy while replacing hundreds or thousands of megabytes of parameter traffic with roughly half a megabyte of labels.

## 18. Limitations and critical analysis

### 18.1 No formal privacy guarantee

SSFL avoids the gradients used by standard reconstruction attacks, but it does not quantify privacy. There is no differential privacy, secure aggregation, encryption analysis, information-theoretic bound, membership-inference test, or formal threat model. Labels and abstentions may reveal a client's class familiarity. The correct claim is reduced gradient-leakage exposure, not complete privacy.

### 18.2 Dependence on representative open data

The method assumes a common unlabeled dataset that is safe and practical to distribute. It must cover important traffic regions and attack categories. If it omits a rare attack, clients cannot transfer that knowledge through the shared examples. If its distribution differs sharply from deployment, the discriminator may reject it or the pseudo-labels may be wrong.

The experiment constructs open data from the same mini-N-BaIoT pool as private and test records. Although the sets are disjoint, they are well aligned. A public dataset collected elsewhere would present a harder problem.

### 18.3 One source dataset

All results derive from N-BaIoT. The paper does not establish performance on other networks, device populations, capture procedures, or modern attacks. Cross-dataset evaluation would be especially valuable because intrusion detectors often exploit dataset-specific artifacts.

### 18.4 Simulated rather than deployed federation

Experiments run on one workstation with an RTX 3090. They do not measure wireless latency, energy, slow clients, dropout, packet loss, asynchronous participation, constrained processors, or end-to-end training time. Communication payload is important but is only one part of deployment cost.

### 18.5 Noisy discriminator supervision

The discriminator's familiar and unfamiliar labels are heuristics. Private examples are declared familiar and low-confidence open examples unfamiliar. Confidence is not correctness. Neural models may be overconfident outside their domain. The discriminator could also learn device or dataset-origin artifacts rather than genuine epistemic familiarity.

### 18.6 Equal voting ignores reliability

Every accepted vote appears to count equally even though clients differ in sample quantity, label quality, class expertise, and calibration. Per-class reliability weights could improve aggregation. The authors identify contribution scoring as future work.

### 18.7 Malicious clients are not handled

A malicious client can send intentionally incorrect labels. Coordinated attackers could control a majority for selected examples. The local discriminator does not help because a malicious participant need not follow it. Byzantine-robust aggregation, reputation, secure count aggregation, and poisoning detection remain open requirements.

### 18.8 Edge cases are underspecified

The paper does not clearly define what happens when all clients abstain, votes tie, only one client votes, or clients are offline. A production design should require minimum participation and agreement, preserve uncertainty, and specify fallback behavior.

### 18.9 Limited statistical reporting

The results are point estimates without clear random-seed repetitions, standard deviations, confidence intervals, or significance tests. Federated results can depend strongly on initialization and partitions. A one-point gain over FL may or may not be stable across repeated runs.

### 18.10 Reproducibility gaps

The batch-size discrepancy, ambiguous fully connected layer count, unspecified multiclass averaging, unclear normalization fitting, and missing details about ties, initialization, padding, activations, server training, and client selection complicate exact reproduction.

### 18.11 Terminology

Classic knowledge distillation emphasizes soft teacher distributions. SSFL ultimately uses hard voted labels. "Distillation" is reasonable because knowledge moves through model outputs, but "filtered federated ensemble pseudo-labeling" more precisely describes the operational mechanism.

## 19. Possible improvements

A stronger system could weight each vote according to per-class expertise, historical accuracy, or agreement with trusted validation data. The server could require a minimum number of votes and a minimum winning margin before assigning a pseudo-label. Uncertain samples could remain unlabeled rather than receiving forced targets.

Familiarity estimation could use calibrated entropy, energy-based scores, deep ensembles, Monte Carlo dropout, Mahalanobis distance, conformal prediction, or contrastive embeddings. These methods could be compared with the current confidence-seeded discriminator.

Privacy could be strengthened through secure aggregation of class counts, shuffled messages, randomized response, differential privacy, or cryptographic protocols. Robust voting and client reputation could address poisoning and Byzantine behavior.

An active-learning mechanism could select a smaller set of informative shared examples instead of repeatedly processing the full open pool. Continual learning and open-set recognition would be needed to detect new attacks outside the fixed eleven-class vocabulary.

Finally, evaluation should include multiple datasets, cross-domain tests, real edge devices, client dropout, heterogeneous neural architectures, noisy labels, adversarial clients, class absence from the open pool, and multiple random seeds.

## 20. Final assessment

The paper's most important contribution is selective knowledge sharing. Each client is treated as a limited expert rather than a universal authority. The common dataset provides a reference surface, the classifier supplies tentative expertise, the discriminator rejects predictions outside that expertise, and majority voting turns the surviving opinions into cheap global supervision.

SSFL reaches 87.40%, 86.70%, and 84.22% accuracy across three difficult non-IID scenarios. It outperforms all tested baselines and reduces reported communication at top accuracy to roughly 0.47-0.55 MB. The ablation study shows that removing the discriminator can reduce performance from the mid-80% range to approximately 10%-40%, making the central mechanism unusually clear.

The method should not be interpreted as a complete solution to federated security. It lacks formal privacy, poisoning resistance, real-device validation, and broad dataset testing. Within the experimental scope, however, it convincingly demonstrates that carefully filtering who is allowed to teach can transform prediction-based federated learning from a fragile technique into an accurate and communication-efficient intrusion detector.

The fairest overall conclusion is that SSFL is an effective filtered pseudo-labeling framework for federated IoT intrusion detection. It substantially improves the reliability of prediction sharing under non-IID label distributions and dramatically reduces communicated information. It is a strong research prototype whose central idea is worth extending, while its real-world privacy, robustness, and deployment properties remain to be demonstrated.

## Reference

R. Zhao, Y. Wang, Z. Xue, T. Ohtsuki, B. Adebisi, and G. Gui, "Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things," IEEE Internet of Things Journal, vol. 10, no. 10, pp. 8645-8657, May 2023. DOI: 10.1109/JIOT.2022.3175918.

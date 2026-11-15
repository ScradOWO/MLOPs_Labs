# MLOps Labs

My lab submissions for the MLOps course. Each lab is in its own folder with its own README and requirements.

| Lab | Topic | Base lab |
|---|---|---|
| [TensorBoard_HParams_Lab](TensorBoard_HParams_Lab/) | Hyperparameter tuning a CIFAR-10 CNN with the TensorBoard HParams dashboard | Tensorflow_Labs / TensorBoard / Lab4 |
| [Snorkel_Data_Labeling_Lab](Snorkel_Data_Labeling_Lab/) | Weak-supervision spam labeling with Snorkel, plus a labeling-function quality gate (dev-accuracy and duplicate checks) | Data_Labs / Data_Labeling_Labs |
| [Docker_lab](Docker_lab/) | Multi-stage Docker build with a build-time model quality gate, slim non-root health-checked serving image, and a Compose train → api → smoke-test pipeline (Wine classifier) | Docker_Labs / Lab2 |
| [MLMD_Lineage_Lab](MLMD_Lineage_Lab/) | ML Metadata lineage for a 3-step training pipeline: persistent store, cached Trainer step, recorded validation failures, and a lineage CLI (trace / impact / run history) | MLMD_Labs / Lab1 |
| [GitHub_Actions_Lab](GitHub_Actions_Lab/) | CI/CD for a calibrated digits classifier: Python-matrix tests, train + quality gate against the latest released model, manual promotion to GitHub Releases ([workflow](.github/workflows/github_actions_lab.yml)) | Github_Labs / Lab2 |
| [ELK_Monitoring_Lab](ELK_Monitoring_Lab/) | Monitoring a deployed model with Elasticsearch, Logstash and Kibana: structured prediction logs, PSI drift detection on live traffic, auto-provisioned Kibana dashboard | ELK_Labs / ELK_Docker |

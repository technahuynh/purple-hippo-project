# purple-hippo-project

## Overview

In our final submission of the project, we (team `purple hippos`) got an F1 score of 0.8163 on the public dataset. Although we tried many different approaches throughout the week, our best models came from the very first and most simple approach: using GIN + virtual nodes with normal cross-entropy (CE) loss and a dropout ratio of 0.5.

GIN + virtual nodes is most effective (from the baseline) for graph classification with noisy labels thanks to its expressivity and good information preservation. We used a high dropout ratio to further prevent the model from overfitting on noise instead of real features. We implemented early stopping and saved the model with the best validation loss (20% of the training dataset).

### Final model accuracies:
- A: 71.41%  
- B: 56.79%  
- C: 84.05%  
- D: 78.02%  

## Other approaches we have tried

- Using GIN-virtual with Noisy Cross Entropy and the following noise probabilities and dropout ratios:  
  - A: noise 0.3, dropout 0.2  
  - B: tried noise 0.5 / 0.7, dropout 0.4 / 0.5  
  - C: noise 0.3, dropout 0.3  
  - D: noise 0.5 / 0.4, dropout 0.4 / 0.3  

- Using GIN-virtual with NCOD loss from https://arxiv.org/abs/2303.09470  
- Applied Variational Graph Autoencoder with NormalCE and NoisyCE and GCN backbone (we’d love to try GIN as well)  
- Using the mean of edge features as a node feature  
- Pretraining with contrastive loss + training with NoisyCE  
- Filtering samples based on high loss, then fine-tuning on the clean dataset

You can find these approaches in our active branches.

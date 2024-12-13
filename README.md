# RAnGE: Reachability Analysis for Guaranteed Ergodicity
### [📄 Paper](https://arxiv.org/abs/2404.03186)<br>

[Henry Berger](mailto:henry.berger@yale.edu) and [Ian Abraham](mailto:ian.abraham@yale.edu)<br>
Yale University

![Figure_1](https://github.com/user-attachments/assets/a1d99304-d85a-4e9c-a6e4-3442a771183e)

## Qickstart: Reproducing Figures from the Paper

_Note: Because training with `pytorch` is nondeterministic, the instructions below will produce figures that are similar but not identical to those in the paper._

### Training Models

```
cd RAnGE
source ./train.sh
```
- This trains two models: one for a uniform distribution and one for a bimodal distribution

### 2. Generate Figures

#### a.  Evaluate many trajectories for RAnGE, reMPC, and MPC
- This is only necessary for Figure 4, and it takes a while. If you only want the other figures, you can skip this step and then comment out the lines for Figure 4 in `evaluation/plot_all`.
```
cd evaluation/batch_evaluation
source ./batch_evaluate_RAnGE
source ./batch_evaluate_reMPC
source ./batch_evaluate_MPC
```

#### b. Generate plots

```
cd evaluation
source ./plot_all
```

## Contact
If you have any questions, please feel free to email the authors.
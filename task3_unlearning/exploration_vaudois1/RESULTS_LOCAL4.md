| tag | config | alpha | clean CE m/s | mild CE | heldout CE | jpeg CE (jamais vu) | KL cible | ancres KL/accord | holdout KL | retain_est | KL vues medium peu confiantes | retain_est2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P9b_all | all m0 lr1e-05 ep5 wa5 wf1 wk5 l20 vmild/shift mu0 targets_g9A8_Borig | 1 | 0.008/0.080 acc 0.998 | 0.134/0.708 | 0.209/0.960 | 0.014/0.123 | 1.505 | 0.012/0.938 | nan | 0.927 | 0.035 | 0.978 |
| P9b_all | all m0 lr1e-05 ep5 wa5 wf1 wk5 l20 vmild/shift mu0 targets_g9A8_Borig | 1.25 | 0.021/0.163 acc 0.993 | 0.168/0.785 | 0.249/1.037 | 0.032/0.227 | 1.327 | 0.018/0.922 | nan | 0.906 | 0.053 | 0.967 |
| P9a_local | mlp6-11 m0 lr0.0001 ep10 wa5 wf1 wk5 l20 vmild/shift mu0 targets_g9A8_Borig | 1.25 | 0.720/1.304 acc 0.759 | 0.726/1.554 | 0.846/1.738 | 0.773/1.399 | 0.348 | 0.030/0.895 | nan | 0.868 | 0.110 | 0.938 |
| P9a_local | mlp6-11 m0 lr0.0001 ep10 wa5 wf1 wk5 l20 vmild/shift mu0 targets_g9A8_Borig | 1.5 | 1.087/1.820 acc 0.711 | 0.999/1.870 | 1.125/2.034 | 1.133/1.895 | 0.308 | 0.047/0.877 | nan | 0.838 | 0.156 | 0.912 |
| P9c_keep2 | mlp6-11 m0 lr0.0001 ep10 wa5 wf1 wk2 l20 vmild/shift mu0 targets_g9A8_Borig | 1.5 | 1.226/1.970 acc 0.691 | 1.182/2.070 | 1.295/2.208 | 1.264/2.030 | 0.275 | 0.051/0.876 | nan | 0.833 | 0.212 | 0.890 |
| P9c_keep2 | mlp6-11 m0 lr0.0001 ep10 wa5 wf1 wk2 l20 vmild/shift mu0 targets_g9A8_Borig | 2 | 1.681/2.552 acc 0.671 | 1.612/2.497 | 1.718/2.589 | 1.712/2.584 | 0.391 | 0.114/0.834 | nan | 0.749 | 0.362 | 0.816 |
| P9d_mlp4 | mlp4-11 m0 lr0.0001 ep10 wa5 wf1 wk5 l20 vmild/shift mu0 targets_g9A8_Borig | 1.5 | 1.086/1.816 acc 0.710 | 0.997/1.862 | 1.126/2.030 | 1.132/1.889 | 0.307 | 0.048/0.879 | nan | 0.839 | 0.157 | 0.912 |
| P9d_mlp4 | mlp4-11 m0 lr0.0001 ep10 wa5 wf1 wk5 l20 vmild/shift mu0 targets_g9A8_Borig | 2 | 1.586/2.450 acc 0.676 | 1.452/2.327 | 1.576/2.446 | 1.623/2.490 | 0.384 | 0.107/0.828 | nan | 0.748 | 0.278 | 0.845 |

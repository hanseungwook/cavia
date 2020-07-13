## CAVIA

Code for "[Fast Context Adaptation via Meta-Learning](https://arxiv.org/abs/1810.03642)" - 
Luisa M Zintgraf, Kyriacos Shiarlis, Vitaly Kurin, Katja Hofmann, Shimon Whiteson
(ICML 2019).

I used Python 3.7 and PyTorch 1.0.1 for these experiments.

### Regression

- Running experiments:
    
    To run the experiment with default settings (2-level), execute
    ```
    python3 regression/main_huh.py
    ```
    
    This will run the regression of different task families (sine, quadratic, cubic, ...) experiment. 
    To run the CelebA image completion experiment, run 

    ```
    python3 main_huh.py --n_iters 2 3 1000 --n_contexts 2 1 --log_interval=20
    ```

    To change the number of context parameters, use the flag `--n_contexts`.

    To change the result log name, use the flag `--log_name`.

    To change the meta-learner to 1-level, change the `--n_iters` to `0 3 1000`. (Setting the most inner loop number of iterations to 0)

    For default settings and other argument options, see `regression/arguments.py`
    
- CelebA dataset:

    If you want to use the code for the CelebA dataset, you have to download it 
    (`http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`) and change the path in 
    `tasks_celeba.py`.

### Classification

- Running experiments:

    To run the experiment with default settings, execute in your command line:
    ```
    python3 classification/cavia.py
    ```

    Use the `--num_filters` flag to set the number of filters. 
    For default settings and other argument options, see `arguments.py`.

- Retrieving Mini-Imagenet:
   
    You need the Mini-Imagenet dataset to run these experiments. 
    See e.g. `https://github.com/y2l/mini-imagenet-tools` for how to retrieve it.
    Put them in the folder `classification/data/miniimagenet/images/` (the label files are already in there).

### Reinforcement Learning

This code is an extended version of Tristan Deleu's PyTorch MAML implementation: `https://github.com/tristandeleu/pytorch-maml-rl`.

- Prerequisites:

    For the MuJoCo experiments you need [`mujoco-py`](https://github.com/openai/mujoco-py) 
and [OpenAI gym](https://github.com/openai/gym).

- Running experiments:

    To run an experiment on the 2D navigation, use the following command:

    ```
    python3 main.py --env-name 2DNavigation-v0 --fast-lr 1.0 --phi-size 5 0  --output-folder results
    ```

#### Acknowledgements

Special thanks to 
Chelsea Finn, 
Jackie Loong and 
Tristan Deleu for their open-sourced MAML implementations.
This was of great help to us, 
and parts of our implementation are based on the PyTorch code from:
- Jackie Loong's implementation of MAML, `https://github.com/dragen1860/MAML-Pytorch`
- Tristan Deleu's implementation of MAML-RL, `https://github.com/tristandeleu/pytorch-maml-rl`

#### BibTex

```
@article{zintgraf2018cavia,
  title={Fast Context Adaptation via Meta-Learning},
  author={Zintgraf, Luisa M and Shiarlis, Kyriacos and Kurin, Vitaly and Hofmann, Katja and Whiteson, Shimon},
  conference={Thirty-sixth International Conference on Machine Learning (ICML 2019)},
  year={2019}
}
```
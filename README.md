# SafetyRacecarButton2 Project

The project is split into three folders:

- `final_sac/`: SAC training and local rendering.
- `final_td3/`: TD3 training and local rendering.
- `final_dockerized/`: Docker version used for controller evaluation, comparison and video rendering.

## TD3 training

Install the required packages once:

```powershell
cd final_td3
pip install -r requirements.txt
```

Run training:

```powershell
python train.py
```

The main experiment settings are in `config.py`. Models, plots and training logs are saved under `results/`.

To render a saved TD3 model locally:

```powershell
python render.py
```

The model path used by the renderer can be changed in the small configuration block at the top of `render.py`.


## SAC training

Install the required packages once:

```powershell
cd final_sac
pip install -r requirements.txt
```

Run training:

```powershell
python main_train.py
```

The main experiment settings can be altered in `def train()` in `main_train.py` file. Models are saved under `models/`, while plots and training logs are saved under `results/`.

In `main_train.py` use `train_2()` for fixed episode length and `train_5()` for episode termination in first correct button during the training process .


To render a saved SAC model locally:

```powershell
python play.py
```
At the end of `play.py` comment the `render_policy()` block for render with goal termination, or comment `render_policy_single_b()` for full episode length .

The desired model for render must be typed at line 47 of `play.py` inside `agent.load()` .

Training comparison :

```powershell
python compare_training_models.py
```

Will create plots comparing the training procedure of choosen models in file `results/training_comparisons/`. Write desired models in first block of `compare_training_models.py` in `MODELS=[]`

Evaluation comparisons:

```powershell
python compare_evaluation_models.py
```

Runs for choosen number of episodes all desired sac models and compares results in plots saved in desired folder. Works with and without goal termination.

## Docker evaluation

Open a terminal in `final_dockerized/` and build the image once:

```powershell
cd final_dockerized
docker compose build
```

Compare the configured controllers over 50 episodes each:

```powershell
docker compose run --rm compare 50
```

Render one episode of a controller:

```powershell
docker compose run --rm render td3 --episodes 1
docker compose run --rm render sac --episodes 1
```

The controller list for comparison is in `models/comparison_models.json`. Generated graphs, CSV files and videos are saved in `outputs/`.




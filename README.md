# End-to-End Visual Obstacle Avoidance for a Robotic Manipulator using Deep Reinforcement Learning

<!-- ABOUT THE PROJECT -->
## About The Project

This project uses a DRL algorithm (TD3) to perform a goal-reaching task for a robotic arm, while avoiding obstacles using camera images. This is the result of my master thesis, which can be found [here](https://www.teses.usp.br/teses/disponiveis/55/55134/tde-30082021-100712/en.php).

### Built With

* [Unity]()
* [ML-Agents]()
* [pytorch]()



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

This project was tested using python versions 3.7.10 and 3.8.8, but any recent python version should work.

1. Clone the repo
   ```sh
   git clone https://github.com/fpadula/visualcollisionarm
   ```

2. Navigate to the **Python Scripts** folder and create a new virtualenv
   ```sh
   cd python_scripts/
   python -m virtualenv venv
   ```

3. Enable the virtualenv and install all the necessary python packages
   ```sh
   source venv/bin/activate
   pip install -r python_packages.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

1. Run a pre-trained model with a single agent using visual and scalar inputs, and enable input visualization
   ```sh
   python src/trainer.py --run_id VisualModel --config_file configs/base_config_visual_aug.yaml --device cuda --exec_type eval --env_location simu_envs/SingleAgentVisualization/scene.x86_64 --simu_spd 1 --eval_episodes 10 --seed 1 --visualize_input true
   ```

2. Run a pre-trained model using multiple agents that uses visual and scalar values
   ```sh
   python src/trainer.py --run_id VisualModel --config_file configs/base_config_visual_aug.yaml --device cuda --exec_type eval --env_location simu_envs/AllAgentsVisual/scene.x86_64 --simu_spd 1
   ```
  
3. Run a pre-trained model using multiple agents that uses only scalar values
   ```sh
   python src/trainer.py --run_id ScalarModel --config_file configs/base_config.yaml --device cuda --exec_type eval --env_location simu_envs/AllAgentsScalar/scene.x86_64 --simu_spd 1
   ```

4. Gym-Wrapper example; running a random policy:
   ```sh
   python examples/gym_api.py
   ```

<!-- ROADMAP -->
## To-do

- Organize the code a little bit better 
- Add more usage examples
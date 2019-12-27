# AWS DeepRacer Custom Framework (Amazon SageMaker RL and AWS RoboMaker service)

This repository contains examples of how to use RL to train an autonomous deepracer. The original soure code can be found here: [AWS DeepRacer SageMaker Example](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning/rl_deepracer_robomaker_coach_gazebo). The original SageMaker repository provided a sample framework for running an AWS DeepRacer simulation and evaluation using fixed neural network architectures, hyperparameters, race tracks, and evaluation metrics.

In this repository, the original SageMaker example has been extended to provide a configurable environment for tuning different aspects of the DeepRacer. 

- Scenario 1: Single Model Training with Configurable Hyperparameters (Simulation and Training)
  
- Scenario 2: Parallel Mult-Model Training with Configurable Hyperparameters (Simulation and Training)

- Scenario 3: Parallel Multi-Track Training with Configurable Hyperparameters (Simulation and Training)



## Contents

Original Files 

* `deepracer_rl.ipynb`: notebook for training autonomous race car.

* `Dockerfile`: Custom docker instead of using SageMaker default docker

* `src/`
  * `training_worker.py`: Main entrypoint for starting distributed training job
  * `markov/`: Helper files for S3 upload/download
   * `presets/default.py`: Preset (configuration) for DeepRacer
   * `rewards/default.py`: Custom reward function
   * `environments/deepracer_racetrack_env.py`: Gym environment file for DeepRacer
   * `actions/model_metadata_10_state.json`: JSON file to customize your action space & the speed
  * `lib/`: redis configuration file and ppo_head.py customized tensorflow file copied to sagemaker container.

* `common/`: helper function to build docker files.

Additional Files 

* `DeepRacer_Framework.ipynb`: SageMaker Notebook providing difference scenarios for training and evaluating DeepRacer simulations
* `common/`: helper function to build docker files.
    * `constant.py`: Constant file containing fixed constants used in the DeepRacer Engine Class
* `src/`: helper function to build docker files.
    * `core/`:
        * `DeepRacerEngine.py`: Wrapper Class for all DeepRacer Related Operations
    * `markov/presets/`: 
        * `preset_hyperparameters.json`: Preset hyperparameters for DeepRacer Model
    * `markov/rewards`: 
        * `complex_reward.py`: Template for Advanced reward function for all reward parameter mappings
    

## How to use the notebook

1. Login to your AWS account - SageMaker service ([SageMaker Link](https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/dashboard))
2. On the left tab select `Notebook instances`
3. Select `Create notebook instance`
4. Fill up the notebook instance name. In the Additional configuration select atleast 25GB. This is because docker gets installed and takes up space.
5. Create a new IAM role. Give root permission
6. Select the git repository and clone this repository.
7. Then click create notebook instance button at the button
8. This takes like 2 min to create your notebook instance. Then click on the newly created instance and click on the juypter notebook.
9. First Run the `DeepRacer_configure.ipynb` Notebook in order to ensure thre correct IAM group and acess controls have been configured 
10. You will see all the github files and now run `DeepRacer_Framework.ipynb`


Note: Run clean robomaker & sagemaker commands only when you are done with training. These can be found in the `DeepRacer_configure.ipynb` Notebook.


## DeepRacer Paper

A techncial paper for AWS DeepRacer is available at https://arxiv.org/abs/1911.01562. Below is a BibTeX entry for citations:
```
@misc{deepracer2019,  
	title={DeepRacer: Educational Autonomous Racing Platform for Experimentation with Sim2Real Reinforcement Learning},
	author={Bharathan Balaji and Sunil Mallya and Sahika Genc and Saurabh Gupta and Leo Dirac and Vineet Khare and Gourav Roy and Tao Sun and Yunzhe Tao and Brian Townsend and Eddie Calleja and Sunil Muralidhara and Dhanasekar Karuppasamy},
	year={2019},  
	eprint={1911.01562},  
	archivePrefix={arXiv},  
	primaryClass={cs.LG}  
}
```

# AWS DeepRacer Custom Framework (Amazon SageMaker RL and AWS RoboMaker service)

This repository contains examples of how to use RL to train an autonomous deepracer. The original soure code can be found here: [AWS DeepRacer SageMaker Example](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/reinforcement_learning/rl_deepracer_robomaker_coach_gazebo). The original SageMaker repository provided a sample framework for running an AWS DeepRacer simulation and evaluation using fixed neural network architectures, hyperparameters, race tracks, and evaluation metrics.

In this repository, the original SageMaker example has been extended to provide a configurable environment for tuning different aspects of the DeepRacer. 

- Scenario 1: Single Model Training with Configurable Hyperparameters (Simulation and Training)
  
- Scenario 2: Parallel Mult-Model Training with Configurable Hyperparameters (Simulation and Training)

- Scenario 3: Parallel Multi-Track Training with Configurable Hyperparameters (Simulation and Training)



## Contents

Original Files 


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

* `DeepRacer_Configure.ipynb`: Configure DeepRacer Backend and IAM Access Rights

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

## DeepRacer Engine Configuration

### Parameters

The ```DeepRacerEngine``` Class requires input parameters when instantiating a new Object. 

For example, if we want to instantiate a new simulation, we provide the ```DeepRacerEngine``` with the following parameters such as:

```python

params = {
    'job_name': 'optimal-path',
    'track_name':'reinvent_base',
    'reward_policy':'src/markov/rewards/reward_estimator_optimal_path.py',
    'job_duration': 600,
    'batch_size':256,
    'evaluation_trials':5
}
```

The Engine accepts the following parameters.

Basic Parameters:
* ```job_name``` - The name of the job. Only Required param
* ```track_name``` - Optional. Default: ```reinvent_base```
* ```instance_type```: The type of compute instance
* ```instance_pool_count```: The number of instances for training
* ```job_duration``` - The duration in seconds of the training. Default 3600s
* ```racetrack_env``` - The environment python file for the racetrack.
* ```reward_policy``` - The reward policy pythonf file for the simulatiomn
* ```meta_file``` - The action space metadata file for the similatiom
* ```presets``` - The preset file for the simulation. This is where changes to the model architecture can be made

Action Space Params:
* ```custom_action_space``` - True/False. Default False. If set, the following parameters are required.
* ```min_speed``` - Minimum speed in the action space
* ```max_speed``` - Maximum Speed in the action Space
* ```min_steering_angle``` - Minimum Steering Angle
* ```max_steering_angle``` - Maximium Steering Angle
* ```speed_interval``` - The speed intervals to calculate the number of entries in the action space
* ```steering_angle_interval``` - the steering angle intervals to calculate the number of entries in the action space.

Hyperparameter Params:
* ```learning_rate``` - The learning rate of the optimization function. Default 0.0003
* ```batch_size``` - The batch size for each training epoch. Default 64
* ```optimizer_epsilon``` - The optimizater Epsilon. Default 0.00001
* ```optimization_epochs``` - The optimzation Epochs - Default 10
* ```discount_factor``` - The discount Factor:  Default 0.999
* ```beta_entropy``` - Beta Entropy. Default 0.01


## Hyperparamter Details

Below provides some details on the hyperparameters within the AWS DeepRacer Framework:
* Learning rate — Behind the scene, gradient descent function comes into picture to find the maxima(reward). Learning rate is the step function of the gradient descent. Learning rate is the rate at which you alter the parameters to check if you have maxima of the reward function.
* Batch size — The batch size determines how many data points or sample size is taken into consideration for updating the training model. So if the total sample space is 1200, and the batch size is 120, then there will be 1200/120 = 10 batches.
* Optimization Epocs - indicates the number of times the training data set will be processed in loop to update the learning parameters. Increase the number of epochs if you have a small training data set to process to get stable results. A larger data set has a smaller value to reduce the time for the training.
* Discount factor determines the weightage of the future events in order to make a current decision. All events in future will vary, however, how much of that needs to be taken into consideration is provided by the discount factor. E.g. You get $100 today, and it will have $100 value, getting $100 (tomorrow — day +1)will have a value of $90 today with discount factor of 0.9, getting $100 (day after tomorrow — day+2) will have a discount factor of Rs 81 at a discount factor of 0.9 ( 0.9 square X 100).
* Entropy — This is the measure of the randomness or the impurity in the data. As the AWS DeepRacer uses AWS DeepLense, the data can be fairly clean and free from randomness. That is why we have a default value of 0.01, meaning 1 out of 100 data inputs may not classify the right action.
* Optimization Epsilon - Gives a balance between exploitation and exploration. Exploitation is when the algorithm makes decision based on information it already has where as exploration is when the algorithm gathers additional information beyond what is already been collected. For instace – exploration helps us find a high reward path that may not have been discovered before. Exploration can be categorical and epsilon. Categorical exploration has a discrete action space. Epsilon exploration has a continuous action space. Epsilon greedy – will decrease the exploration value over time so that it explores a lot at first but later will perform actions more and more based on experience while still allowing for some random exploration.


## How to use the notebook

1. Login to your AWS account - SageMaker service ([SageMaker Link](https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/dashboard))
2. On the left tab select `Notebook instances`
3. Select `Create notebook instance`
4. Fill up the notebook instance name. In the Additional configuration select atleast 25GB. This is because docker gets installed and takes up space.
5. Create a new IAM role. Give root permission.
6. Then click create notebook instance button at the button
7. This takes like 2 min to create your notebook instance. Then click on the newly created instance and click on the juypter notebook.
8. In the Jupyter Notebook Webpage, click the New Button, and then Select Terminal
9. In the terminal window type the following commands:
```bash
git clone https://github.com/raminetinati/DeepRacer-Framework.git
```
10. Return to the Jupyter Notebook Webpage. A new folder should appear named ```DeepRacer-Framework```
11. First Run the `DeepRacer_Configure.ipynb` Notebook in order to ensure thre correct IAM group and acess controls have been configured 
12. You will see all the github files and now run `DeepRacer_Framework.ipynb`


Note: Run clean robomaker & sagemaker commands only when you are done with training. These can be found in the `DeepRacer_Configure.ipynb` Notebook.


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

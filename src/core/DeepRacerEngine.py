import boto3
import sagemaker
import sys
import os
import re
import numpy as np
import subprocess

sys.path.append("common")
import constant as const
from misc import get_execution_role, wait_for_s3_object
from docker_utils import build_and_push_docker_image
from sagemaker.rl import RLEstimator, RLToolkit, RLFramework
from copy_to_sagemaker_container import get_sagemaker_docker, copy_to_sagemaker_container, get_custom_image_name
from time import gmtime, strftime
import time
from IPython.display import Markdown
from markdown_helper import *

import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, clear_output

import warnings
warnings.filterwarnings("ignore")



class DeepRacerEngine:
    # Select the instance type
    instance_type = None

    # Starting SageMaker session
    sage_session = None

    # Create unique job name.
    job_name_prefix = None

    # Duration of job in seconds (1 hours)
    job_duration_in_seconds = None

    # Track name
    track_name = None

    # IAM Params
    sagemaker_role = None

    # Metric Defs
    metric_definitions = None

    # Docker Image Varaibles
    custom_image_name = None
    repository_short_name = None
    sagemaker_docker_id = None
    docker_build_args = None

    # VPC params
    ec2 = None
    deepracer_security_groups = None
    deepracer_vpc = None
    deepracer_subnets = None
    route_tables = None

    # S3 variavles
    s3_location = None
    s3_output_path = None

    # robmaker parms
    estimator = None
    job_name = None

    # param Kinesis Video Stream
    kvs_stream_name = None

    # robomaker Parsm
    robomaker = None
    robomaker_s3_key = None
    robomaker_source = None
    simulation_software_suite = None
    robot_software_suite = None
    rendering_engine = None
    app_name = None
    response = None
    simulation_app_arn = None

    # Simulationb Parsms
    num_simulation_workers = None
    envriron_vars = None
    simulation_application = None
    vpcConfig = None
    responses = None
    job_arns = None
    simulation_application_bundle_location = None

    # output Parsm
    tmp_dir = None
    training_metrics_file = None
    training_metrics_path = None

    # evaluate parsms:
    eval_envriron_vars = None
    eval_simulation_application = None
    eval_vpcConfig = None
    eval_responses = None

    # Eval output
    evaluation_metrics_file = None
    evaluation_metrics_path = None
    evaluation_trials = None

    
    # AWS Region
    aws_region = None

    ''' 
    INIT
    '''
    def __init__(self,kwargs):

        print('***Deep Racer Engine Backend***')
        
        if 'job_name' in kwargs:
            self.job_name = kwargs['job_name']
        else:
            raise Exception("A Job Name MUST be provided. Stopping execution.")
            
        if 'instance_type' in kwargs:
            self.instance_type = kwargs['instance_type']
        else:
            self.instance_type = const.default_instance_type

        if 'job_duration' in kwargs:
            self.job_duration_in_seconds = kwargs['job_duration']
        else:
            self.job_duration_in_seconds = const.default_job_duration

        if 'track_name' in kwargs:
            self.track_name = kwargs['track_name']
        else:
            self.track_name = const.default_track_name

        if 'racetrack_env' in kwargs:
            self.envir_file_local = kwargs['racetrack_env']
        else:
            self.envir_file_local = const.envir_file_local

        if 'reward_policy' in kwargs:
            self.reward_file_local = kwargs['reward_policy']
        else:
            self.reward_file_local = const.reward_file_local

        if 'meta_file' in kwargs:
            self.model_meta_file_local = kwargs['meta_file']
        else:
            self.model_meta_file_local = const.model_meta_file_local

        if 'presets' in kwargs:
            self.presets_file_local = kwargs['presets']
        else:
            self.presets_file_local = const.presets_file_local
            
        if 'evaluation_trials' in kwargs:
            self.evaluation_trials = kwargs['evaluation_trials']
        else:
            self.evaluation_trials = const.evaluation_trials

        #local file where hyperparams will be saved..
        self.presets_hyperp_local = const.tmp_hyperparam_preset
         
        self.create_hyperparams(kwargs)

        self.sage_session = sagemaker.session.Session()
        self.s3 = boto3.resource('s3')
        self.job_name_prefix = self.job_name

        if self.track_name not in const.track_name:
            raise Exception("The track name provded does not exist. Please provide a trackname which matches an"
                            "available track")

        self.aws_region = self.sage_session.boto_region_name
        if self.aws_region not in ["us-west-2", "us-east-1", "eu-west-1"]:
            raise Exception("This notebook uses RoboMaker which is available only in US East (N. Virginia),"
                            "US West (Oregon) and EU (Ireland). Please switch to one of these regions.")


    def create_hyperparams(self, kwargs):

        #first we're going to get all the global variables
        with open(const.default_hyperparam_preset) as fp:
            data = json.load(fp)
            if 'learning_rate' in kwargs:
                data['learning_rate'] = kwargs['learning_rate']

            if 'batch_size' in kwargs:
                data['batch_size'] = kwargs['batch_size']

            if 'optimizer_epsilon' in kwargs:
                data['optimizer_epsilon'] = kwargs['optimizer_epsilon']

            if 'optimization_epochs' in kwargs:
                data['learning_rate'] = kwargs['learning_rate']

            #now write these key,values to file
            with open(const.tmp_hyperparam_preset, 'w') as filew:
                for k,v in data.items():
                    c = '{}={}\n'.format(k,v)
                    filew.write(c)



    ''' 
    TO-ADD
    '''
    def start_training_testing_process(self):

        print('********************************')
        print('PERFORMING ALL DOCKER, VPC, AND ROUTING TABLE WORK....')
        ## Configure The S3 Bucket which this job will work for
        self.configure_s3_bucket()
        ## Configure the IAM role and ensure that the correct access priv are available
        self.create_iam_role()

        self.build_docker_container()

        self.configure_vpc()

        self.create_routing_tables()

        self.upload_environments_and_rewards_to_s3()

        self.configure_metrics()

        self.configure_estimator()

        self.configure_kinesis_stream()

        self.start_robo_maker()

        self.create_simulation_application()

        self.start_simulation_job()

        self.plot_training_output()
        

    def start_model_evaluation(self):

        self.configure_evaluation_process()

        self.plot_evaluation_output()


    '''
    Set up the linkage and authentication to the S3 bucket that we want to use for checkpoint and metadata.    
    '''

    def configure_s3_bucket(self):

        # S3 bucket
        self.s3_bucket = self.sage_session.default_bucket()

        # SDK appends the job name and output folder
        self.s3_output_path = 's3://{}/'.format(self.s3_bucket)

        # Ensure that the S3 prefix contains the keyword 'sagemaker'
        self.s3_prefix = self.job_name_prefix + "-sagemaker-" + strftime("%y%m%d-%H%M%S", gmtime())

        # Get the AWS account id of this account
        self.sts = boto3.client("sts")
        self.account_id = self.sts.get_caller_identity()['Account']

        print("Using s3 bucket {}".format(self.s3_bucket))
        print("Model checkpoints and other metadata will be stored at: \ns3://{}/{}".format(self.s3_bucket,
                                                                                            self.s3_prefix))

    def create_iam_role(self):
        try:
            self.sagemaker_role = sagemaker.get_execution_role()
        except:
            self.sagemaker_role = get_execution_role('sagemaker')

        print("Using Sagemaker IAM role arn: \n{}".format(self.sagemaker_role))

    def build_docker_container(self):
        cpu_or_gpu = 'gpu' if self.instance_type.startswith('ml.p') else 'cpu'
        self.repository_short_name = "sagemaker-docker-%s" % cpu_or_gpu
        self.custom_image_name = get_custom_image_name(self.repository_short_name)
        try:
            print("Copying files from your notebook to existing sagemaker container")
            self.sagemaker_docker_id = get_sagemaker_docker(self.repository_short_name)
            copy_to_sagemaker_container(self.sagemaker_docker_id, self.repository_short_name)
        except Exception as e:
            print("Creating sagemaker container")
            self.docker_build_args = {
                'CPU_OR_GPU': cpu_or_gpu,
                'AWS_REGION': boto3.Session().region_name,
            }
            self.custom_image_name = build_and_push_docker_image(self.repository_short_name,
                                                                 build_args=self.docker_build_args)
            print("Using ECR image %s" % self.custom_image_name)

    def configure_vpc(self):
        self.ec2 = boto3.client('ec2')

        #
        # Check if the user has Deepracer-VPC and use that if its present. This will have all permission.
        # This VPC will be created when you have used the Deepracer console and created one model atleast
        # If this is not present. Use the default VPC connnection
        #
        self.deepracer_security_groups = [group["GroupId"] for group in
                                          self.ec2.describe_security_groups()['SecurityGroups'] \
                                          if group['GroupName'].startswith("aws-deepracer-")]

        # deepracer_security_groups = False
        if (self.deepracer_security_groups):
            print("Using the DeepRacer VPC stacks. This will be created if you run one training job from console.")
            self.deepracer_vpc = [vpc['VpcId'] for vpc in self.ec2.describe_vpcs()['Vpcs'] \
                                  if "Tags" in vpc for val in vpc['Tags'] \
                                  if val['Value'] == 'deepracer-vpc'][0]
            self.deepracer_subnets = [subnet["SubnetId"] for subnet in self.ec2.describe_subnets()["Subnets"] \
                                      if subnet["VpcId"] == self.deepracer_vpc]
        else:
            print("Using the default VPC stacks")
            self.deepracer_vpc = [vpc['VpcId'] for vpc in self.ec2.describe_vpcs()['Vpcs'] if vpc["IsDefault"] == True][
                0]

            self.deepracer_security_groups = [group["GroupId"] for group in
                                              self.ec2.describe_security_groups()['SecurityGroups'] \
                                              if 'VpcId' in group and group["GroupName"] == "default" and group[
                                                  "VpcId"] == self.deepracer_vpc]

            self.deepracer_subnets = [subnet["SubnetId"] for subnet in self.ec2.describe_subnets()["Subnets"] \
                                      if subnet["VpcId"] == self.deepracer_vpc and subnet['DefaultForAz'] == True]

        print("Using VPC:", self.deepracer_vpc)
        print("Using security group:", self.deepracer_security_groups)
        print("Using subnets:", self.deepracer_subnets)

    '''
    A SageMaker job running in VPC mode cannot access S3 resourcs. 
    So, we need to create a VPC S3 endpoint to allow S3 access from SageMaker container. 
    To learn more about the VPC mode, 
    please visit [this link.](https://docs.aws.amazon.com/sagemaker/latest/dg/train-vpc.html)
    '''

    def create_routing_tables(self):

        print("Creating Routing Tables")
        try:
            self.route_tables = [route_table["RouteTableId"] for route_table in
                                 self.ec2.describe_route_tables()['RouteTables'] \
                                 if route_table['VpcId'] == self.deepracer_vpc]
        except Exception as e:
            if "UnauthorizedOperation" in str(e):
                # display(Markdown(generate_help_for_s3_endpoint_permissions(self.sagemaker_role)))
                print(e, 'UnauthorizedOperation')
            else:
                print('EE')
                # display(Markdown(create_s3_endpoint_manually(self.aws_region, self.deepracer_vpc)))
            raise e

        print("Trying to attach S3 endpoints to the following route tables:", self.route_tables)

        if not self.route_tables:
            raise Exception(("No route tables were found. Please follow the VPC S3 endpoint creation "
                             "guide by clicking the above link."))
        try:
            self.ec2.create_vpc_endpoint(DryRun=False,
                                         VpcEndpointType="Gateway",
                                         VpcId=self.deepracer_vpc,
                                         ServiceName="com.amazonaws.{}.s3".format(self.aws_region),
                                         RouteTableIds=self.route_tables)
            print("S3 endpoint created successfully!")
        except Exception as e:
            if "RouteAlreadyExists" in str(e):
                print("S3 endpoint already exists.")
            elif "UnauthorizedOperation" in str(e):
                # display(Markdown(generate_help_for_s3_endpoint_permissions(role)))
                raise e
            else:
                # display(Markdown(create_s3_endpoint_manually(aws_region, deepracer_vpc)))
                raise e

    def upload_environments_and_rewards_to_s3(self):

        self.s3_location = self.s3_prefix
        print(self.s3_location)

        # Clean up the previously uploaded files
        bucket = self.s3.Bucket(self.s3_bucket)
        bucket.objects.filter(Prefix=self.s3_prefix).delete()
        # !aws s3 rm --recursive {s3_location}

        # Make any changes to the environment and preset files below and upload these files
        # !aws s3 cp src/markov/environments/deepracer_racetrack_env.py {self.s3_location}/environments/deepracer_racetrack_env.py
        envir_file_s3 = self.s3_location + '/environments/deepracer_racetrack_env.py'
        bucket.upload_file(self.envir_file_local, envir_file_s3)

        # !aws s3 cp src/markov/rewards/complex_reward.py {s3_location}/rewards/reward_function.py
        reward_file_s3 = self.s3_location + '/rewards/reward_function.py'
        bucket.upload_file(self.reward_file_local, reward_file_s3)

        # !aws s3 cp src/markov/actions/model_metadata_10_state.json {s3_location}/model_metadata.json
        model_meta_file_s3 = self.s3_location + '/model_metadata.json'
        bucket.upload_file(self.model_meta_file_local, model_meta_file_s3)

        # !aws s3 cp src/markov/presets/default.py {s3_location}/presets/preset.py
        presets_file_s3 = self.s3_location + '/presets/preset.py'
        bucket.upload_file(self.presets_file_local, presets_file_s3)

        presets_hyperparams_file_s3 = self.s3_location + '/presets/preset_hyperparams.py'
        bucket.upload_file(const.tmp_hyperparam_preset, presets_hyperparams_file_s3)
        print('Cleaning Up Tmp HyperParam file')
        os.remove(const.tmp_hyperparam_preset)

        
    def configure_metrics(self):

        self.metric_definitions = [
            # Training> Name=main_level/agent, Worker=0, Episode=19, Total reward=-102.88, Steps=19019, Training iteration=1
            {'Name': 'reward-training',
             'Regex': '^Training>.*Total reward=(.*?),'},

            # Policy training> Surrogate loss=-0.32664725184440613, KL divergence=7.255815035023261e-06, Entropy=2.83156156539917, training epoch=0, learning_rate=0.00025
            {'Name': 'ppo-surrogate-loss',
             'Regex': '^Policy training>.*Surrogate loss=(.*?),'},
            {'Name': 'ppo-entropy',
             'Regex': '^Policy training>.*Entropy=(.*?),'},

            # Testing> Name=main_level/agent, Worker=0, Episode=19, Total reward=1359.12, Steps=20015, Training iteration=2
            {'Name': 'reward-testing',
             'Regex': '^Testing>.*Total reward=(.*?),'},
        ]

    def configure_estimator(self):
        self.estimator = RLEstimator(entry_point=const.entry_point,
                                     source_dir=const.source_dir,
                                     image_name=self.custom_image_name,
                                     dependencies=["common/"],
                                     role=self.sagemaker_role,
                                     train_instance_type=self.instance_type,
                                     train_instance_count=1,
                                     output_path=self.s3_output_path,
                                     base_job_name=self.job_name_prefix,
                                     metric_definitions=self.metric_definitions,
                                     train_max_run=self.job_duration_in_seconds,
                                     hyperparameters={
                                         "s3_bucket": self.s3_bucket,
                                         "s3_prefix": self.s3_prefix,
                                         "aws_region": self.aws_region,
                                         "preset_s3_key": "%s/presets/preset.py" % self.s3_prefix,
                                         "model_metadata_s3_key": "%s/model_metadata.json" % self.s3_prefix,
                                         "environment_s3_key": "%s/environments/deepracer_racetrack_env.py" % self.s3_prefix,
                                     },
                                     subnets=self.deepracer_subnets,
                                     security_group_ids=self.deepracer_security_groups,
                                     )

        self.estimator.fit(wait=False)
        self.job_name = self.estimator.latest_training_job.job_name
        print("Training job: %s" % self.job_name)

    def configure_kinesis_stream(self):

        self.kvs_stream_name = "dr-kvs-{}".format(self.job_name)

        # !aws --region {aws_region} kinesisvideo create-stream --stream-name {kvs_stream_name} --media-type video/h264 --data-retention-in-hours 24
        print ("Created kinesis video stream {}".format(self.kvs_stream_name))

    def start_robo_maker(self):
        self.robomaker = boto3.client("robomaker")

    def create_simulation_application(self):
        self.robomaker_s3_key = 'robomaker/simulation_ws.tar.gz'
        self.robomaker_source = {'s3Bucket': self.s3_bucket,
                                 's3Key': self.robomaker_s3_key,
                                 'architecture': "X86_64"}
        self.simulation_software_suite = {'name': 'Gazebo',
                                          'version': '7'}
        self.robot_software_suite = {'name': 'ROS',
                                     'version': 'Kinetic'}
        self.rendering_engine = {'name': 'OGRE',
                                 'version': '1.x'}

        bucket = self.s3.Bucket(self.s3_bucket)

        if not os.path.exists('./build/output.tar.gz'):
            print("Using the latest simapp from public s3 bucket")

            # Download Robomaker simApp for the deepracer public s3 bucket
            # !aws s3 cp {self.simulation_application_bundle_location} ./
            # simulation_application_bundle_location = "s3://deepracer-managed-resources-us-east-1/deepracer-simapp-notebook.tar.gz"
            copy_source = {
                'Bucket': 'deepracer-managed-resources-us-east-1',
                'Key': 'deepracer-simapp-notebook.tar.gz'
            }

            simulation_application_bundle_s3 = './'
            self.s3.Bucket('deepracer-managed-resources-us-east-1').download_file('deepracer-simapp-notebook.tar.gz',
                                                                                  './deepracer-simapp-notebook.tar.gz')

            # Remove if the Robomaker sim-app is present in s3 bucket
            sim_app_filename = self.robomaker_s3_key
            bucket.delete_objects(
                Delete={
                    'Objects': [
                        {
                            'Key': sim_app_filename
                        },
                    ],
                    'Quiet': True
                })

            # Uploading the Robomaker SimApp to your S3 bucket
            simulation_application_bundle_location = "./deepracer-simapp-notebook.tar.gz"
            simulation_application_bundle_s3 = self.robomaker_s3_key
            bucket.upload_file(simulation_application_bundle_location, simulation_application_bundle_s3)

            # Cleanup the locally downloaded version of SimApp
            sim_app_filename = './deepracer-simapp-notebook.tar.gz'
            os.remove(sim_app_filename)

        else:
            print("Using the simapp from build directory")
            # !aws s3 cp ./build/output.tar.gz s3://{self.s3_bucket}/{self.robomaker_s3_key}
            sim_app_build_location = "./build/output.tar.gz"
            sim_app_build_s3 = self.robomaker_s3_key
            bucket.upload_file(sim_app_build_location, sim_app_build_s3)

        self.app_name = "deepracer-notebook-application" + strftime("%y%m%d-%H%M%S", gmtime())

        print('App Name: {}'.format(self.app_name))
        try:
            self.response = self.robomaker.create_simulation_application(name=self.app_name,
                                                                         sources=[self.robomaker_source],
                                                                         simulationSoftwareSuite=self.simulation_software_suite,
                                                                         robotSoftwareSuite=self.robot_software_suite,
                                                                         renderingEngine=self.rendering_engine)
            self.simulation_app_arn = self.response["arn"]
            print("Created a new simulation app with ARN:", self.simulation_app_arn)
        except Exception as e:
            if "AccessDeniedException" in str(e):
                # display(Markdown(generate_help_for_robomaker_all_permissions(role)))
                raise e
            else:
                raise e

    def start_simulation_job(self):
        self.num_simulation_workers = 1

        self.envriron_vars = {
            "WORLD_NAME": self.track_name,
            "KINESIS_VIDEO_STREAM_NAME": self.kvs_stream_name,
            "SAGEMAKER_SHARED_S3_BUCKET": self.s3_bucket,
            "SAGEMAKER_SHARED_S3_PREFIX": self.s3_prefix,
            "TRAINING_JOB_ARN": self.job_name,
            "APP_REGION": self.aws_region,
            "METRIC_NAME": "TrainingRewardScore",
            "METRIC_NAMESPACE": "AWSDeepRacer",
            "REWARD_FILE_S3_KEY": "%s/rewards/reward_function.py" % self.s3_prefix,
            "MODEL_METADATA_FILE_S3_KEY": "%s/model_metadata.json" % self.s3_prefix,
            "METRICS_S3_BUCKET": self.s3_bucket,
            "METRICS_S3_OBJECT_KEY": self.s3_bucket + "/training_metrics-"+self.job_name+".json",
            "TARGET_REWARD_SCORE": "None",
            "NUMBER_OF_EPISODES": "0",
            "ROBOMAKER_SIMULATION_JOB_ACCOUNT_ID": self.account_id
        }

        self.simulation_application = {"application": self.simulation_app_arn,
                                       "launchConfig": {"packageName": "deepracer_simulation_environment",
                                                        "launchFile": "distributed_training.launch",
                                                        "environmentVariables": self.envriron_vars}
                                       }

        self.vpcConfig = {"subnets": self.deepracer_subnets,
                          "securityGroups": self.deepracer_security_groups,
                          "assignPublicIp": True}

        self.responses = []
        for job_no in range(self.num_simulation_workers):
            client_request_token = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
            response = self.robomaker.create_simulation_job(iamRole=self.sagemaker_role,
                                                            clientRequestToken=client_request_token,
                                                            maxJobDurationInSeconds=self.job_duration_in_seconds,
                                                            failureBehavior="Continue",
                                                            simulationApplications=[self.simulation_application],
                                                            vpcConfig=self.vpcConfig
                                                            )
            self.responses.append(response)

        print("Created the following jobs:")
        job_arns = [response["arn"] for response in self.responses]
        for response in self.responses:
            print("Job ARN", response["arn"])

    def plot_training_output(self):
        self.tmp_root = 'tmp/'
        os.system("mkdir {}".format(self.tmp_root))
        self.tmp_dir = "tmp/{}".format(self.job_name)
        os.system("mkdir {}".format(self.tmp_dir))
        print("Create local folder {}".format(self.tmp_dir))

        self.training_metrics_file = "training_metrics-"+self.job_name+".json"
        self.training_metrics_path = "{}/{}".format(self.s3_bucket, self.training_metrics_file)

#         # Disable
#         def blockPrint():
#             sys.stdout = open(os.devnull, 'w')

#         # Restore
#         def enablePrint():
#             sys.stdout.close()
#             sys.stdout = self._original_stdout

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        ax = fig.add_subplot(1, 2, 1)

        x_axis = 'episode'
        y_axis = 'reward_score'
        ytwo_axis = 'completion_percentage'
        
        for i in range(200):
            #     print(i)
#             blockPrint()
            wait_for_s3_object(self.s3_bucket, self.training_metrics_path, self.tmp_dir)

            json_file = "{}/{}".format(self.tmp_dir, self.training_metrics_file)
            with open(json_file) as fp:
                data = json.load(fp)
                data = pd.DataFrame(data['metrics'])

                x = data[x_axis].values
                y = data[y_axis].values
                y2 = data[ytwo_axis].values
                
                ax.set_xlim(0, np.max(x))
                ax[0].plot(x, y)
                ax[1].plot(x, y2)
                fig.tight_layout()
                display(fig)
                clear_output(wait=True)
                plt.pause(0.5)
#             enablePrint()



    def configure_evaluation_process(self):
        sys.path.append("./src")

        self.num_simulation_workers = 1

        self.eval_envriron_vars = {
            "WORLD_NAME": self.track_name,
            "KINESIS_VIDEO_STREAM_NAME": "SilverstoneStream",
            "MODEL_S3_BUCKET": self.s3_bucket,
            "MODEL_S3_PREFIX": self.s3_prefix,
            "APP_REGION": self.aws_region,
            "MODEL_METADATA_FILE_S3_KEY": "%s/model_metadata.json" % self.s3_prefix,
            "METRICS_S3_BUCKET": self.s3_bucket,
            "METRICS_S3_OBJECT_KEY": self.s3_bucket + "/evaluation_metrics-"+self.job_name+".json",
            "NUMBER_OF_TRIALS": self.evaluation_trials,
            "ROBOMAKER_SIMULATION_JOB_ACCOUNT_ID": self.account_id
        }

        self.eval_simulation_application = {
            "application": self.simulation_app_arn,
            "launchConfig": {
                "packageName": "deepracer_simulation_environment",
                "launchFile": "evaluation.launch",
                "environmentVariables": self.eval_envriron_vars
            }
        }

        self.eval_vpcConfig = {"subnets": self.deepracer_subnets,
                               "securityGroups": self.deepracer_security_groups,
                               "assignPublicIp": True}

        responses = []
        for job_no in range(self.num_simulation_workers):
            response = self.robomaker.create_simulation_job(clientRequestToken=strftime("%Y-%m-%d-%H-%M-%S", gmtime()),
                                                            outputLocation={
                                                                "s3Bucket": self.s3_bucket,
                                                                "s3Prefix": self.s3_prefix
                                                            },
                                                            maxJobDurationInSeconds=self.job_duration_in_seconds,
                                                            iamRole=self.sagemaker_role,
                                                            failureBehavior="Continue",
                                                            simulationApplications=[self.eval_simulation_application],
                                                            vpcConfig=self.eval_vpcConfig)
            responses.append(response)

        # print("Created the following jobs:")
        for response in responses:
            print("Job ARN", response["arn"])

    def plot_evaluation_output(self):
        evaluation_metrics_file = "evaluation_metrics-"+self.job_name+".json"
        evaluation_metrics_path = "{}/{}".format(self.s3_bucket, evaluation_metrics_file)
        wait_for_s3_object(self.s3_bucket, evaluation_metrics_path, self.tmp_dir)

        json_file = "{}/{}".format(self.tmp_dir, evaluation_metrics_file)
        with open(json_file) as fp:
            data = json.load(fp)

        df = pd.DataFrame(data['metrics'])
        # Converting milliseconds to seconds
        df['elapsed_time'] = df['elapsed_time_in_milliseconds'] / 1000
        df = df[['trial', 'completion_percentage', 'elapsed_time']]

        display(df)

























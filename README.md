# ChargingEnvironment
ChargingEnvironment is a Python library for training a deep reinforcement learning agent to maximize its profit in a decentralized electric vehicle smart charging process. The program architecture was developed as part of a master's thesis at the Technical University of Berlin. 

## Installation
Download the repository, navigate into the folder 'ChargingEnvironment' and use the package manager [pip](https://pip.pypa.io/en/stable/) to install the charging environment.

```bash
pip install -e .
```

## File Structure
In the folder are three Jupyter notebooks, that are central for the use of the underlying classes:

**main_RL_model.ipynb:** Allows training and testing of a Deep Q-learning agent. In addition, various benchmark strategies can be tested within the main method.

**main_sup_model.ipynb:** Allows training of k-nearest neighbors (knn) and MLP classifiers to make charging decisions in the same environment. To do so, a labeled training and test dataset must be compiled first using *main_benchmark.ipynb*. The trained agents can then be tested in *main_RL_model.ipynb*.

**main_benchmark.ipynb:** Allows the computation of the theoretical optimum assuming complete information. In addition, labeled training and test sets for k-NN and MLP classifiers can be compiled here.

The folders **dqn_models/**, **knn_models/** and **mlp_models/** contain selected pre-trained agents. When new agents are trained, they are stored in the corresponding folders.

The folder **data/** contains the underlying data. For the time being, several .csv files are included here, since individual columns in individual files unfortunately cause formatting problems. An aggregated file will follow.

The folder **/envs/custom_env_dir/** contains the charging environment, the agent, the deep Q-network, as well as classes for the replay memory, the two supervised learning models knn and mlp, convex optimization, and data processing:

**charging_env.py:** Here, the charging environment is defined. If new input feature combinations are to be tested, they must be incorporated in this file. The class is called by *main_RL_model.ipynb* among others.

**conv_optim.py:** This class is accessed by *main_benchmark.ipynb*. Here, optimal charging decisions are calculated using CVXPY under the assumption of complete information. It is used to calculate the theoretical optimum and to compile labeled datasets for supervised learning models.

**custom_dqn.py:** The architecture of the Q-network as a function approximator is defined here. This is also used by the target network. The class is called by *dqn_agent.py*, because the Q-network is part of the agent.

**data_handler.py:** This class contains various functions for loading and storing data from training and test runs.

**dqn_agent.py:** Here, the implemented deep Q-learning agent is defined. This agent has a Q-network, a target network and a replay memory. Besides, the action selection and the learning step are defined. The class is called by *main_RL_model.ipynb*.

**replay_memory.py:** This defines the replay memory, which is called by *dqn_agent.py*.

**sup_model.py:** This file contains the k-nearest neighbors and mlp classifier implementations. It is called by *main_sup_model.ipynb*.

**utils.py:** This class contains functions to create the charging environment.

## Contributing
Pull requests are welcome. For major changes, please open an issue to discuss what you would like to change. 
Questions regarding the program or results of the thesis are welcome. 
Contact: joh.ruetten@gmail.com

## License
[MIT](https://choosealicense.com/licenses/mit/)

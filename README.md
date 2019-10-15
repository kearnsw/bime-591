# Artificial Intelligence Methods for Conversational Agents in Healthcare

## Facilitators: 
Will Kearns, Aakash Sur, BHI PhD Students and Trevor Cohen, BHI Faculty

## Details: 
Tuesdays, Autumn Quarter: 11:30 am-12:20 pm, Health Sciences Building, T478

## Course Description:
Through this course, students will be introduced to reinforcement learning methods and how to apply them to train health dialog systems to address specific problems in healthcare. We will cover a range of machine learning methods including tree search, tree pruning, Markov decision processes, and Q-learning. We will explore both classical methods and recent advances in the development of dialog system components including natural language understanding, dialog management, and natural language generation. The course structure will be a mixture of lectures and interactive coding sessions culminating in the deployment of a health dialog system.

We welcome questions during the class as others might share the same questions. 
If you need individual help, please see one of the instructors after class or send a question to the group on slack. 

## Course Reading:
[Neural Approaches to Conversational AI Question Answering, Task-Oriented Dialogues and Social Chatbots](https://arxiv.org/pdf/1809.08267.pdf)

[Designing Voice User Interfaces: Principles of Conversational Experiences](https://www.cathypearl.com/book-1)

## Week 1 - 10/1/19

### Title: 
Introduction to Conversational Agents and Reinforcement Learning

### Description: 
We will introduce conversational agents operating within a natural language environment, an ideal context for employing reinforcement learning (RL). We will survey the methods of RL and its applications of NLP. Finally, we will cover the software requirements for the class, and ensure students can interactively follow along in coding exercises. 

### Lecture:
[Week 1 Slides](https://github.com/kearnsw/bime-591/raw/master/lectures/Week%201.pdf)

### Reading: 
[Write an NLU model](https://rasa.com/docs/rasa/nlu/training-data-format/)

### Coding: 
None

## Week 2 - 10/8/19

### Title: 
Natural Language Understanding


### Description: 
We will explore how we can train agents to understand their conversational environments.

### Lecture:
[Week 2 Slides](https://github.com/kearnsw/bime-591/blob/master/lectures/bime-591-week-2-nlu.pdf)

### Reading:
[ONENET:Joint Domain, Intent, Slot Prediction for Spoken Language Understanding](https://www.microsoft.com/en-us/research/publication/onenet-joint-domain-intent-slot-prediction-spoken-language-understanding/)

[Dynamic Integration of Background Knowledge in Neural NLU Systems](https://arxiv.org/pdf/1706.02596v3.pdf)

### Coding: 
[Train NLU model: with a Pipeline in Rasa](https://rasa.com/docs/rasa/nlu/using-nlu-only/)


## Week 3 - 10/15/19

### Title: 
Tree Search

### Description: 
We will model decisions as trees and learn to efficiently search them using classic algorithms such as breadth-first search and depth-first search. In addition, we will introduce heuristic based searches, including A* search. 

### Lecture:
[Week 3 Slides](https://github.com/kearnsw/bime-591/blob/master/lectures/Week%203.pdf)

### Reading:
None
### Coding:
[Week 3 Coding](https://github.com/kearnsw/bime-591/blob/master/notebooks/week_3.ipynb)

## Week 4 - 10/22/19

### Title: 
Advanced Tree Searches

### Description: 
We will cover how to model two player games as trees, and how the optimal strategy can be recovered from these trees. In addition, we will cover how to prune these trees to limit the total search space using alpha-beta pruning, and heuristic pruning. 

### Reading:
TBD
### Coding:
TBD

## Week 5 - 10/29/19

### Title: 
Dialog Management

### Description:


### Reading:
TBD

### Coding: 
[Train Rasa DM with Interactive Learning](https://rasa.com/docs/rasa/core/interactive-learning/)


## Week 6 - 11/5/2019

### Title: 
Markov Decision Processes

### Description: 
In this class, we will extend our tree based decision models to graphs with Markov models. We will learn how to calculate the best route through a Markov decision process (MDPs) using the Bellman equations. Finally, we will extend these ideas to conversational agents using partially observable Markov decision processes (POMDPs).

### Reading:
[POMDP-Based StatisticalSpoken Dialog Systems:A Review](http://cs.brown.edu/courses/csci2951-k/papers/young13.pdf)

[Training a real-world POMDP-based Dialogue System](https://pdfs.semanticscholar.org/c746/fe146789142262b749d362f5a0f38f3bf8ad.pdf)

### Coding:
TBD


## Week 7 - 11/12/2019

### Title: 
Q-Learning

### Description: 
Here we will introduce one of the key concepts in RL, Q-learning. This approach overcomes the limitations of MDPs and allows us to conduct on-line or off-line learning without complete information.

### Reading:
TBD

### Coding:
TBD

## Week 8 - 11/19/2019

### Title: 
Deep Q-Networks

### Description: 
Moving past basic tabular Q-learning, we will cover current approaches which revolve around Deep Q-Networks (DQN). We will cover popular examples of DQNs used to master video games, and conversations. Finally, we will cover how to efficiently train these models using experience replay. 

### Reading: 
[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

### Coding: 
[Train dialog policy w/ episodic replay](https://rasa.com/docs/rasa/core/policies/#configuring-policies)


## Week 9 - 11/26/2019

### Title: 
Advanced Neural Methods for Dialog Systems

### Description: 
We will cover advanced neural architectures for training dialog systems, e.g. A2C and MemNN models.

### Reading:
[Sample-efficient Actor-Critic Reinforcement Learning with Supervised Data for Dialogue Management](https://arxiv.org/pdf/1707.00130.pdf)

[Understanding Actor Critic Methods and A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)

### Coding: 
TBD

## Week 10 - 12/3/2019

### Title: 
Ethics and NLG

### Description: 
We will finish off the course with a discussion of ethics in the development of dialog systems using two case studies and then focus the discussion on health dialog systems in particular.

### Reading:
[The Design and Implementation of XiaoIce, an Empathetic Social Chatbot](https://arxiv.org/abs/1812.08989)

[Twitter taught Microsoft’s AI chatbot to be a racist asshole in less than a day](https://www.theverge.com/2016/3/24/11297050/tay-microsoft-chatbot-racist)

### Coding:
TBD

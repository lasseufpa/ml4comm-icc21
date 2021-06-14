# Installation of MimoRL-simple-1-v0 RL agent

Besides the dependencies described for the Jupyter notebooks of this tutorial, in order to run the RL code you will need 
[Stable-Baselines](https://stable-baselines.readthedocs.io/en/master/) (we used stable-baselines version 2.10.2), OpenAI gym (we used gym==0.18.0), pygame (we used pygame==2.0.1) and bidict (we used bidict==0.21.2). 

You can install bidict, for instance, with:

```
pip install bidict
```

The RL agent is executed at a base station (BS) with an antenna array and serves single-antenna users on downlink using an analog MIMO architecture with Nb beam vector indices. The BS and users live in a M x M grid world in which there are M2 invariant channels depending only on position. An episode lasts Ne time slots, and for each episode, a user moves left/right/up/down. It is an episodic (not continuing) task. The reward is a normalized throughput (in fact, thee magnitude of the combined channel) and a penalyy of -100 is added if a user is not allocated for Na=3 consecutive slots.

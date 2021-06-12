Nu = 2
Nb = 64

action = (user, beam_index) = 3
action space dim = Nu * Nb ==> 128 possible actions.

M = 6 ==> 6 x 6 grid
Na = 3 
state = [position_user1, ..., position_user_Nu, previously_scheduled_users]
state space dim = (M^2)^Nu*(Nu)^(Na-1) ==> 5184 possible states.

Example:
Ok: 0 0 1 0 0 1 0 1 0 1
Não Ok: 0 0 0 0 0 1 1 1 1 1

env
   previously_scheduled_users[Na-1]
   penalty = 100

step()
previously_scheduled_users=[0,1]
at time t=0  action = (0, 23)    
    throughput = combined_gain #codigo do Ailton
    scheduled_user = 0
    if len(np.unique([previously_scheduled_users scheduled_user])) == Nu:
    	reward = throughput
    else:
	reward = throughput - penalty 
    #update for next iteration
    loop to shift to the left
    	previously_scheduled_users    

at time t=1  action = (1, 12)
previously_scheduled_users=[1, 0]

at time t=2  action = (1, 3)
previously_scheduled_users=[1, 1]


### Nao sei se util:
git_lasse\ak-py\akpy
FiniteMDP.py
e
FiniteMDP2.py
GridWorldFiniteMDP.py
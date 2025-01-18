import numpy as np
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent
from typing import List
import pdb

class MultiModalPreds:
    def __init__(self,a_lat=0.2,dt=0.1):
        self.a_lat = a_lat #m/s^2
        self.dt = dt

    def predict(self, preds,ego_state,visualize=False):
        '''
        Input:
            preds: List[List] where outer list is the prediction horizon length and inner list is the [x, y] prediction
        Output:
            mm_preds: List[List[List]] where Inner most list is the [x, y] prediction of a lane change mode
        '''

        mm_preds = preds
        
        for t in range(len(preds)):
            for i, agent in enumerate(preds[t]):
                assert isinstance(agent, IDMAgent)
                #check if the TV is relative
                if t ==0 :
                    dot_product = np.dot( (np.array([preds[0][i].to_se2().x,preds[0][i].to_se2().y])-np.array([ego_state.center.point.x,ego_state.center.point.y])), np.array([np.cos(ego_state.center.heading),np.sin(ego_state.center.heading)])) 
                else:
                    dot_product = np.dot( (np.array([preds[0][i][0].to_se2().x,preds[0][i][0].to_se2().y])-np.array([ego_state.center.point.x,ego_state.center.point.y])), np.array([np.cos(ego_state.center.heading),np.sin(ego_state.center.heading)]))
                if dot_product> 0:
                    if t == 0:
                        x_p= preds[0][i].to_se2().x + 0.5*np.sin(preds[0][i].to_se2().heading)
                        y_p= preds[0][i].to_se2().y + 0.5*np.cos(preds[0][i].to_se2().heading)
                        dist_p = np.linalg.norm([x_p-ego_state.center.x,y_p-ego_state.center.y])
                        dist_0 = np.linalg.norm([preds[0][i].to_se2().x-ego_state.center.x,preds[0][i].to_se2().y-ego_state.center.y])
                        sign = -np.sign(dist_p-dist_0)
                        sign1 = np.sign(ego_state.center.x-preds[0][i].to_se2().x) * np.sign(np.sin(preds[0][i].to_se2().heading))
                        sign2 = np.sign(ego_state.center.y-preds[0][i].to_se2().y) * np.sign(np.cos(preds[0][i].to_se2().heading))
                    else:
                        x_p= preds[0][i][0].to_se2().x + 0.5*np.sin(preds[0][i][0].to_se2().heading)
                        y_p= preds[0][i][0].to_se2().y + 0.5*np.cos(preds[0][i][0].to_se2().heading)
                        dist_p = np.linalg.norm([x_p-ego_state.center.x,y_p-ego_state.center.y])
                        dist_0 = np.linalg.norm([preds[0][i][0].to_se2().x-ego_state.center.x,preds[0][i][0].to_se2().y-ego_state.center.y])
                        sign = -np.sign(dist_p-dist_0)
                        sign1 = np.sign(ego_state.center.x-preds[0][i][0].to_se2().x) * np.sign(np.sin(preds[0][i][0].to_se2().heading))
                        sign2 = np.sign(ego_state.center.y-preds[0][i][0].to_se2().y) * np.sign(np.cos(preds[0][i][0].to_se2().heading))
                    #get x and y coordinates of the idm agent

                    ey_t = 0.5*self.a_lat*(t*self.dt)**2

                    x_t = agent.to_se2().x + sign*ey_t*np.sin(agent.to_se2().heading)
                    y_t = agent.to_se2().y + sign*ey_t*np.cos(agent.to_se2().heading)

                    s_t = agent.progress
                    v_t = agent.velocity
                    
                    #copy the existing prediction
                    mm_preds[t][i] = [agent]

                    #make agent copy
                    mm_preds[t][i].append(agent.copy())

                    #Set state of the agent copy
                    mm_preds[t][i][-1].set_state(x_t,y_t,s_t,v_t)                    
                else:
                    mm_preds[t][i] = [agent]      
        if visualize:
            self.visualize(mm_preds,ego_state)
        return self.prune_mm_preds(mm_preds,ego_state)

    def prune_mm_preds(self,mm_preds,ego_state):
        temp_mm_preds = [[[None for _ in agent] for agent in mm_preds[t]] for t in range(len(mm_preds))] #same structure as mm_preds filled with filler None
        dist_arr = []
        ind_arr = []
        for i, agent in enumerate(mm_preds[0]):
            if isinstance(agent,List) and len(agent)>1:
                # print('This agent has lane change mode')
                dist = np.linalg.norm([agent[0].to_se2().x-ego_state.center.x,agent[0].to_se2().y-ego_state.center.y])
                dist_arr.append(dist)
                ind_arr.append(i)
        min_ind = np.argsort(dist_arr)
        two_agents_ind = [ind_arr[ind] for ind in min_ind.tolist()[:2]]

        #set the unselected indices to single mode predictions
        for ind in min_ind.tolist():
            if ind not in two_agents_ind:
                for t in range(len(mm_preds)):
                    temp_mm_preds[t][ind] = [mm_preds[t][ind][0]]

        #Modify mm_preds to move the selected agents with multi modal predictions to the first two indices in mm_preds
        for t in range(len(mm_preds)):
            for m, rel_agent_ind in enumerate(two_agents_ind):
                temp_mm_preds[t][m] = mm_preds[t][rel_agent_ind]
            k = len(two_agents_ind)
            for i, agent in enumerate(mm_preds[t]):
                if i not in two_agents_ind:
                    temp_mm_preds[t][k] = agent
                    k+=1
                else:
                    pass
        return temp_mm_preds

    def visualize(self,mm_preds,ego_state):
        import matplotlib.pyplot as plt
        N = len(mm_preds)  
        for i, mm_agent in enumerate(mm_preds[0]):
            print(f'Agent {i}')
            if isinstance(mm_agent,List) and len(mm_agent)>1:     
                print(f'Agent {i} has lane change mode')
                time_series_x, time_series_y = [], []
                time_series_lc_mode_x, time_series_lc_mode_y = [], []
                for k in range(N):
                    time_series_x.append(mm_preds[k][i][0].to_se2().x)
                    time_series_y.append(mm_preds[k][i][0].to_se2().y)
                    time_series_lc_mode_x.append(mm_preds[k][i][1].to_se2().x)
                    time_series_lc_mode_y.append(mm_preds[k][i][1].to_se2().y)
                plt.figure()
                plt.plot(time_series_x,time_series_y,'ro',label='original')
                plt.plot(time_series_lc_mode_x,time_series_lc_mode_y,'bo',label='lc mode')
                plt.plot(ego_state.center.point.x,ego_state.center.point.y,'gx',label='ego')
                plt.axis('equal')
                plt.legend()
                plt.show()
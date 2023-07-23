# new attempt to buffer demand
class QlearningRL():
    def __init__(self, states, proxy_reward=0):
        self.states = states
        self.n_states = len(self.states)
        self.w = 0.1
        self.discount_factor = 0.8
        self.threshold = 0
        self.max_allocate_in_one_step = 100
        self.n = 20 # number of states per province
        self.max_vent_requirement = 6000 # number that can be sent in a single time step. Blown way up and only positive impact observed so df
        self.ratio_step = 0.1
        self.offset_abs_cap = 3
        self.offset_scale = 0.25

        self.reward_state = {}
        self.q_vals = {}
        self.q_offset_dict = {}
        for state in states:
            self.q_offset_dict[state] = 0
            
        self.ratio_dict_init = False
        buffer_len = 5
        self.init_venti_required_buffer(buffer_len)

    
    def init_venti_required_buffer(self, buffer_len):
        self.venti_required_buffer_dict = {}
        for state in self.states:
            self.venti_required_buffer_dict[state] = [0] * buffer_len
            
    def update_venti_required_buffer_and_rtn(self, venti_required):
        venti_required_buffered = []
        for i, vents in enumerate(venti_required):
            next_state = self.states[i]
            tmp = self.venti_required_buffer_dict[next_state]
            tmp.append(vents)
            self.venti_required_buffer_dict[next_state] = tmp[1:]
            venti_required_buffered.append(max(self.venti_required_buffer_dict[next_state]))
        return venti_required_buffered
            
    
    def update_proxy_rewards(self, largest_observed, state_status_dict, vent_need_dict):
        for state in self.states:
            if state_status_dict[state].get_vh() > vent_need_dict[state]:
                q_offset_adjust = -1
            elif state_status_dict[state].get_vh() < vent_need_dict[state]:
                q_offset_adjust = 1
            else:
                q_offset_adjust = 0
                
            self.q_offset_dict[state] = np.clip((self.q_offset_dict[state] + q_offset_adjust), self.offset_abs_cap*-1, self.offset_abs_cap)
            proxy_val = largest_observed[state] + largest_observed[state] * self.offset_scale * self.q_offset_dict[state]
            #self.reward_state[state] = self.create_reward_proxy(proxy_val, self.n)
            self.reward_state[state] = self.create_reward_proxy_ln(proxy_val,self.n)
            self.q_vals[state] = self.q_learning(self.max_vent_requirement, self.max_allocate_in_one_step, self.n, self.ratio_step, self.reward_state[state])
            
    def get_action(self, venti_available, venti_required, state_allocation_limit, federal_available, federal_transfer_limit):
        #print("GET ACTION CALLED")
        #print("Stage 2")
        #state_to_state = self.optimal_policy(venti_available, venti_required, self.n, self.ratio_step, self.max_allocate_in_one_step, self.n_states, vals1, vals2, vals3, vals4, state_allocation_limit)
        #state_to_state = self.q_policy(venti_available, venti_required, self.n, self.ratio_step, self.max_allocate_in_one_step, self.n_states, self.q1, self.q2, self.q3, self.q4, state_allocation_limit)
        venti_required_buffered = self.update_venti_required_buffer_and_rtn(venti_required)
        state_to_state = self.q_policy_fed(venti_available, venti_required_buffered, self.n, self.ratio_step, self.max_allocate_in_one_step, self.n_states, state_allocation_limit, federal_available, federal_transfer_limit)
        return state_to_state[0]
                
    def next_state(self, venti_avail, venti_need):
        state = venti_avail/venti_needed
        return state
    
    def find_where_it_fits(self,ratio, max_states, increment):
        max_val = increment*max_states
        min_val = 0
        i = 0
        ratio_steps = []
        while i < max_states:
            ratio_steps.append(i*increment)
            i = i + 1
        ratio_steps = np.asarray(ratio_steps)
        min_diff = float('inf')
        index = -1
        for j in range(0,max_states):
            diff = abs(ratio - ratio_steps[j])
            if diff < min_diff:
                min_diff = diff
                index = j
        return index
    
    def create_reward_proxy(self,alpha,n):
        #https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population_density#2015_density_(states,_territories_and_DC)
        reward_state = np.zeros(n)
        for i in range(0,n):
            reward_state[i] = 0.1*i*alpha - 2 * alpha
            if reward_state[i] > -alpha:
                reward_state[i] = -alpha
        return reward_state
    
    def create_reward_proxy_ln(self,alpha,n):
        reward_state = np.zeros(n)
        for i in range(1,n+1):
            #reward_state[i-1] = (math.log(0.1*i) - math.log(0.1*n)) * alpha #postive offset
            #const = (math.log(1.5) - math.log(2)) * alpha
            reward_state[i-1] = (math.log(0.1*i) - math.log(0.1*n)) * math.log(alpha) #postive offset
            const = (math.log(1.5) - math.log(2)) * math.log(alpha)
            #reward_state[i-1] = (math.log(0.1*i) - math.log(0.1*(n+1))) * alpha #postive offset
            #const = (math.log(0.1*n) - math.log(0.1*(n+1))) * alpha
            #const = 0
            if reward_state[i-1] > const:
                reward_state[i-1] = const
        #print(reward_state)
        return reward_state

    def q_learning(self, max_vent_requirement, max_allocate_in_one_step, n, ratio_step, reward, verbose=0):
        #i = 0
        #vent_steps = []
        #while i < max_vent_requirement/max_allocate_in_one_step:
        #    vent_steps.append(i*max_allocate_in_one_step)
        #    i = i + 1
        #vent_steps = np.asarray(vent_steps)
        #i = 0
        #ratio_steps = []
        #while i < n:
        #    ratio_steps.append(i*ratio_step)
        #    i = i + 1
        #ratio_steps = np.asarray(ratio_steps)
        values = np.zeros((n))
        q_func = np.zeros((n,3))
        delta = float('inf')
        while delta > self.threshold:
            delta = 0
            for j in range(0, n):
                v = values[j]
                if j == 0:
                    reward_backward = reward[j]
                    reward_stay = reward[j]
                    reward_forward = reward[j+1]
                    val_backward = values[j]
                    val_stay = values[j]
                    val_forward = values[j+1]
                elif j == n-1:
                    reward_backward = reward[j-1]
                    reward_stay = reward[j]
                    reward_forward = reward[j]
                    val_backward = values[j-1]
                    val_stay = values[j]
                    val_forward = values[j]
                else:
                    reward_backward = reward[j-1]
                    reward_stay = reward[j]
                    reward_forward = reward[j+1]
                    val_backward = values[j-1]
                    val_stay = values[j]
                    val_forward = values[j+1]
                total_reward_backward = reward_backward + self.discount_factor*val_backward
                total_reward_forward = reward_forward + self.discount_factor*val_forward
                total_reward_stay = reward_stay + self.discount_factor*val_stay
                values[j] = max(total_reward_backward, total_reward_stay, total_reward_forward)
                rew_array = [total_reward_backward, total_reward_stay, total_reward_forward]
                rew_array = np.asarray(rew_array)
                rew_arg = np.argmax(rew_array)
                q_func[j,0] = total_reward_backward
                q_func[j,1] = total_reward_stay
                q_func[j,2] = total_reward_forward
                delta = max(delta, abs(v - values[j]))
        if verbose == 1:
            print((q_func))
        return(q_func)
    

    def q_policy_fed(self,venti_available, venti_required, max_states, increment, max_allocate_in_one_step, n_states, state_allocation_limit, fed_pile, fed_pile_limit):
        
        state_to_state = np.zeros((n_states+1,n_states+1)) # state_to_state transfer matrix
        state_state_dict = {}
        where_state_dict = {}
        q_array_dict = {}
        max_q_dict = {}
        for i in range(len(self.states)):
            state = self.states[i]
            if venti_required[i] == 0:
                venti_req = 1
            else:
                venti_req = venti_required[i]
            state_state_dict[state] = venti_available[i]/venti_req
            where_state_dict[state] = self.find_where_it_fits(state_state_dict[state], max_states, increment)
            q_array_tmp = [self.q_vals[state][where_state_dict[state], 0], self.q_vals[state][where_state_dict[state], 1], self.q_vals[state][where_state_dict[state], 2]]
            q_array_dict[state] = np.asarray(q_array_tmp)
            max_q_dict[state] = np.argmax(q_array_dict[state]) 
        
        max_array = []
        for state in self.states:
            max_array.append(max_q_dict[state])

        givers = []
        geters = []
        stayers = []

        for i in range(0, len(max_array)):
            if max_array[i] == 0:
                givers.append(i)
            elif max_array[i] == 1:
                stayers.append(i)
            elif max_array[i] == 2:
                geters.append(i)

#         if len(givers) == 0 or len(geters) == 0:
#             return np.floor(state_to_state), fed_pile
#         elif len(givers) >= 1 and len(geters) >= 1 and len(givers) >= len(geters):
#             for i in givers:
#                 for j in geters:
#                     fed_flag = 0
#                     state_to_state[i, j] = state_allocation_limit[i]
#                     if fed_flag == 0 and fed_pile > fed_pile_limit:
#                         state_to_state[n_states, j] = state_to_state[n_states, j] + fed_pile_limit
#                         fed_pile = fed_pile - fed_pile_limit
#                         fed_flag = 1
#                     elif fed_flag == 0 and fed_pile < fed_pile_limit:
#                         state_to_state[n_states, j] = state_to_state[n_states, j] + fed_pile
#                         fed_pile = 0
#                         fed_flag = 1
#         elif len(givers) >= 1 and len(geters) >= 1 and len(givers) < len(geters):
#             for j in geters:
#                 fed_flag = 0
#                 for i in givers:
#                     state_to_state[i, j] = state_allocation_limit[i]
#                     if fed_flag == 0 and fed_pile > fed_pile_limit:
#                         state_to_state[n_states, j] = state_to_state[n_states, j] + fed_pile_limit
#                         fed_pile = fed_pile - fed_pile_limit
#                         fed_flag = 1
#                     elif fed_flag == 0 and fed_pile < fed_pile_limit:
#                         state_to_state[n_states, j] = state_to_state[n_states, j] + fed_pile
#                         fed_pile = 0
#                         fed_flag = 1

        #print(givers)
        #print(stayers)
        #print(geters)
        
        if self.ratio_dict_init is False:
            self.ratio_dict = self.init_ratio_dict(venti_available, venti_required)
            self.ratio_dict_init = True
        else:
            self.ratio_dict = self.update_ratio_dict_required(self.ratio_dict, venti_required)
    
#         if len(givers) == 0:
#             potential_givers = []
#             potential_geters = []
#             if len(geters) != 0:
#                 sorted_all_highlow = self.sort_givers_geters(list(range(self.n_states)), venti_available, venti_required, True)
#                 potential_givers.append(sorted_all_highlow[0])
#             else: 
#                 potential_givers = []
#                 potential_geters = []
#                 for i in range(len(self.states)):
#                     (_, _, state_ratio) = self.ratio_dict[i]
#                     if state_ratio > 1.0:
#                         potential_givers.append(i)
#                     if state_ratio < 1.0:
#                         potential_geters.append(i)

#             for next_giver in potential_givers:
#                 givers.append(next_giver)
#             for next_geter in potential_geters:
#                 geters.append(next_geter)
        
        # unique entries only
        #givers = set(givers)
        #geters = set(geters)
            
        # order givers by their ratio
        #print('\n')
        #print('givers', givers)
        #print('geters', geters)
        sorted_givers = self.sort_givers_geters(givers, venti_available, venti_required, True) # highest ratio first
        sorted_geters = self.sort_givers_geters(geters, venti_available, venti_required, False) # lowest ratio first
        #print('sorted_givers', sorted_givers)
        #print('sorted_geters', sorted_geters)
        max_ratio = self.ratio_step * self.n
        #print("max ratio:", self.ratio_step * self.n)
        
        if len(givers) == 0 or len(geters) == 0:
            return np.floor(state_to_state), fed_pile
        elif len(givers) >= 1 and len(geters) >= 1 and len(givers) >= len(geters):
            for i in sorted_givers:
                for j in sorted_geters:
                    fed_flag = 0
                    (geter_avail, geter_req, geter_ratio) = self.ratio_dict[j]
                    (giver_avail, giver_req, giver_ratio) = self.ratio_dict[i]
                    if geter_ratio < max_ratio:
                        #if j == 0:
                        #    print("geter ratio", self.ratio_dict[j])
                        geter2x = math.ceil(2 * geter_req - geter_avail)
                        giver10p = state_allocation_limit[i]
                        giver1x = math.floor(giver_avail - giver_req)
                        #print('giver:', self.states[i])
                        #print('geter2x:', geter2x)
                        #print('giver10p:', giver10p)
                        #print('giver1x:', giver1x)
                        giver_amt = min([geter2x, giver10p, giver1x])
                        if giver_amt < 0:
                            giver_amt = 0
                    else:
                        giver_amt = 0
                    state_to_state[i, j] = giver_amt
                    self.ratio_dict = self.update_ratio_dict_transaction(self.ratio_dict, i, giver_amt, j)
                    if fed_flag == 0 and fed_pile > fed_pile_limit:
                        state_to_state[n_states, j] = state_to_state[n_states, j] + fed_pile_limit
                        fed_pile = fed_pile - fed_pile_limit
                        fed_flag = 1
                    elif fed_flag == 0 and fed_pile < fed_pile_limit:
                        state_to_state[n_states, j] = state_to_state[n_states, j] + fed_pile
                        fed_pile = 0
                        fed_flag = 1
        elif len(givers) >= 1 and len(geters) >= 1 and len(givers) < len(geters):
            for j in sorted_geters:
                fed_flag = 0
                for i in sorted_givers:
                    (geter_avail, geter_req, geter_ratio) = self.ratio_dict[j]
                    (giver_avail, giver_req, giver_ratio) = self.ratio_dict[i]
                    if geter_ratio < max_ratio:
                        #if j == 0:
                        #    print("geter ratio", self.ratio_dict[j])
                        geter2x = math.ceil(2 * geter_req - geter_avail)
                        giver10p = state_allocation_limit[i]
                        giver1x = math.floor(giver_avail - giver_req)
                        #print('giver:', self.states[i])
                        #print('geter2x:', geter2x)
                        #print('giver10p:', giver10p)
                        #print('giver1x:', giver1x)
                        giver_amt = min([geter2x, giver10p, giver1x])
                        if giver_amt < 0:
                            giver_amt = 0
                    else:
                        giver_amt = 0
                    state_to_state[i, j] = giver_amt
                    self.ratio_dict = self.update_ratio_dict_transaction(self.ratio_dict, i, giver_amt, j)
                    if fed_flag == 0 and fed_pile > fed_pile_limit:
                        state_to_state[n_states, j] = state_to_state[n_states, j] + fed_pile_limit
                        fed_pile = fed_pile - fed_pile_limit
                        fed_flag = 1
                    elif fed_flag == 0 and fed_pile < fed_pile_limit:
                        state_to_state[n_states, j] = state_to_state[n_states, j] + fed_pile
                        fed_pile = 0
                        fed_flag = 1
        
        #print(np.floor(state_to_state))
        return (np.floor(state_to_state), fed_pile)
    
    def sort_givers_geters(self, state_indexes, venti_available, venti_required, reverse):
        tuple_list = []
        for i in state_indexes:
            if venti_required[i] == 0:
                venti_req = 1
            else:
                venti_req = venti_required[i]
            next_ratio = venti_available[i] / venti_req
            next_tuple = (i, next_ratio)
            tuple_list.append(next_tuple)
        
        sorted_list = sorted(tuple_list, key = lambda x: x[1], reverse = reverse)
        index_list = []
        for (index, ratio) in sorted_list:
            index_list.append(index)
        return index_list
    
    def init_ratio_dict(self, venti_available, venti_required):
        ratio_dict = {}
        for i in range(len(self.states)):
            if venti_required[i] == 0:
                venti_req = 1
            else:
                venti_req = venti_required[i]
            ratio = venti_available[i] / venti_req
            ratio_dict[i] = (venti_available[i], venti_required[i], ratio)
            #if i == 0:
            #    print("District of Columbia: ", ratio_dict[i])
        return ratio_dict
    
    def update_ratio_dict_required(self, ratio_dict, venti_required):
        for i in range(len(self.states)):
            avail = ratio_dict[i][0]
            if venti_required[i] == 0:
                venti_req = 1
            else:
                venti_req = venti_required[i]
            ratio = avail / venti_req
            ratio_dict[i] = (avail, venti_required[i], ratio)
        return ratio_dict
    
    def update_ratio_dict_transaction(self, ratio_dict, giver, giver_amt, geter):
        # update giver
        (avail, req, ratio) = ratio_dict[giver]
        avail = avail - giver_amt
        if req == 0:
            venti_req = 1
        else:
            venti_req = req
        ratio = avail / venti_req
        ratio_dict[giver] = (avail, req, ratio)
        # update geter
        (avail, req, ratio) = ratio_dict[geter]
        avail = avail + giver_amt
        if req == 0:
            venti_req = 1
        else:
            venti_req = req
        ratio = avail / venti_req
        ratio_dict[geter] = (avail, req, ratio)
        return ratio_dict
        
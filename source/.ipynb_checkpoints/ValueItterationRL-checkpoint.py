# alternative implementation supporting multiple states
class ValueItterationRL():
    def __init__(self, states, proxy_reward=0):
        self.states = states
        self.n_states = len(self.states)
        self.w = 0.1
        self.discount_factor = 0.8
        self.threshold = 0
        self.max_allocate_in_one_step = 100
        self.n = 20
        self.max_vent_requirement = 6000
        self.ratio_step = 0.1
        
        self.reward_state = {}
        self.vals = {}
        
    def update_proxy_rewards(self, largest_observed, state_status_dict, vent_need_dict):
        for state in self.states:
            proxy_val = largest_observed[state]
            self.reward_state[state] = self.create_reward_proxy(proxy_val, self.n)
            self.vals[state] = self.optimal_value_iteration(self.max_vent_requirement, self.max_allocate_in_one_step, self.n, self.ratio_step, self.reward_state[state])
        
    def get_action(self, venti_available, venti_required, state_allocation_limit, federal_available, federal_transfer_limit):
        #print("Stage 2")
        #state_to_state = self.optimal_policy(venti_available, venti_required, self.n, self.ratio_step, self.max_allocate_in_one_step, self.n_states, vals1, vals2, vals3, vals4, state_allocation_limit)
        #state_to_state = self.optimal_policy_new(venti_available, venti_required, self.n, self.ratio_step, self.max_allocate_in_one_step, self.n_states, self.vals1, self.vals2, self.vals3, self.vals4, state_allocation_limit)
        (state_to_state, _) = self.optimal_policy_fed(venti_available, venti_required, self.n, self.ratio_step, self.max_allocate_in_one_step, self.n_states, state_allocation_limit, federal_available, federal_transfer_limit)
        return state_to_state
                
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
    
    def create_reward_proxy(self,population_density, n):
        #https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population_density#2015_density_(states,_territories_and_DC)
        reward_state = np.zeros(n)
        for i in range(0,20):
            reward_state[i] = 0.1*i*population_density - population_density
            if reward_state[i] > 0:
                reward_state[i] = 0
        return reward_state
    
    def optimal_value_iteration(self,max_vent_requirement, max_allocate_in_one_step, n, ratio_step, reward):
        i = 0
        vent_steps = []
        while i < max_vent_requirement/max_allocate_in_one_step:
            vent_steps.append(i*max_allocate_in_one_step)
            i = i + 1
        vent_steps = np.asarray(vent_steps)
        i = 0
        ratio_steps = []
        while i < n:
            ratio_steps.append(i*ratio_step)
            i = i + 1
        ratio_steps = np.asarray(ratio_steps)
        values = np.zeros((n))
        delta = float('inf')
        while delta > self.threshold:
            delta = 0
            for j in range(0, n):
                v = values[j]
                if j == 0:
                    reward_backward = reward[j]
                    val_backward = values[j]
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
                delta = max(delta, abs(v - values[j]))
        return(values)
    
    def optimal_policy_fed(self, venti_available, venti_required, max_states, increment, max_allocate_in_one_step, n_states, state_allocation_limit, fed_pile, fed_pile_limit):
        state_to_state = np.zeros((n_states+1,n_states+1)) # state_to_state transfer matrix
        state_state_dict = {}
        where_state_dict = {}
        where_state_dict_get = {}
        reward_array_get = []
        for i in range(len(self.states)):
            state = self.states[i]
            state_state_dict[state] = venti_available[i]/venti_required[i]
            where_state_dict[state] = self.find_where_it_fits(state_state_dict[state], max_states, increment)
            where_state_dict_get[state] = self.find_where_it_fits((venti_available[i]+max_allocate_in_one_step)/venti_required[i],max_states,increment)
            reward_tmp = self.vals[state][where_state_dict_get[state]]
            reward_array_get.append(reward_tmp)
            
        reward_array_get = np.asarray(reward_array_get)
        args_get = np.argsort(reward_array_get)

        vent_actual_available = []
        vent_actual_needed = []
        for i in range(0,len(venti_required)):
            vent_actual_available.append(venti_available[i] - venti_required[i])
            vent_actual_needed.append(venti_required[i] - venti_available[i])
        vent_actual_available = np.asarray(vent_actual_available)
        vent_actual_needed = np.asarray(vent_actual_needed)

        state_get = [-1,-1]
        state_give = [-1,-1]
        state_get = np.asarray(state_get)
        state_give = np.asarray(state_give)
        

        flag = 0
        run = 0
        while flag < 2 and run < 4:
            for arg in args_get:
                run = run + 1
                if vent_actual_needed[arg] > 0 and flag < 2:
                    state_get[flag] = arg
                    flag = flag + 1

        flag1 = 0
        run1 = 0
        while flag1 < 2 and run1 < 4:
            for arg in args_get:
                if arg not in state_get:
                    run1 = run1 + 1
                    if vent_actual_available[arg] > 0 and flag1 < 2:
                        state_give[flag1] = arg
                        flag1 = flag1 + 1

        for i in range(0, len(state_get)):
            fed_flag = 0
            for j in range(0, len(state_give)):
                if state_get[i] != -1 and state_give[j] != -1:
                    state_to_state[state_give[j], state_get[i]] = state_allocation_limit[state_give[j]]
                    if fed_flag == 0 and fed_pile > fed_pile_limit:
                        state_to_state[n_states, state_get[i]] = state_to_state[n_states, state_get[i]] + fed_pile_limit
                        fed_pile = fed_pile - fed_pile_limit
                        fed_flag = 1
                    elif fed_flag == 0 and fed_pile < fed_pile_limit:
                        state_to_state[n_states, state_get[i]] = state_to_state[n_states, state_get[i]] + fed_pile
                        fed_pile = 0
                        fed_flag = 1


        #print(state_get)
        #print(state_give)
        return (np.floor(state_to_state), fed_pile)
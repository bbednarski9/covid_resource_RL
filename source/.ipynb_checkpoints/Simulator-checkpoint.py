class Simulator():
    def __init__(self, data_start_date, data_end_date, sim_start_date, states, model_list, primary_dataloader, LSTM_dataloader, prediction_interval, ventillator_delay,\
                 path, ihme_path, loss_plots=False, prediction_analysis=False, LSTMplots=False, include_federal=False, proxy_reward=0, collect_modeling_data=False, model_select = None):
        if sim_start_date >= data_end_date:
            print("Simualtor error 1: start_date after available data")
        if sim_start_date < data_start_date:
            print("Simualtor error 2: start_date before available data")
        self.path = path
        self.ihme_path = ihme_path
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        self.sim_start_date = sim_start_date
        self.loss_plots = loss_plots
        self.prediction_interval = prediction_interval
        self.ventillator_delay = ventillator_delay
        self.prediction_analysis = prediction_analysis
        self.sim_length = (self.data_end_date - self.sim_start_date).days - self.prediction_interval
        self.states = states
        self.primary_dataloader = primary_dataloader
        self.LSTM_dataloader = LSTM_dataloader
        self.model_select = model_select
        self.model_names = {}
        if model_select is not None:
            self.model_list = [42]
        else:
            self.model_list = list(set(model_list))
        #self.quant_min = 0.0 # quantized state space to 20 states per state. 20n
        #self.quant_max = 2.0
        self.LSTM_model = None
        self.LSTM_epochs = 150
        self.LSTM_retrain_epochs = 100
        self.LSTMplots = LSTMplots
        self.LSTM_allow_cont_training = True
        self.LSTM_train_batch_size = 10
        
        self.log_rnn = True
        self.data_agg_window_length = 1000
        #JANGO
        self.RNN_log_filename = 'RNN_predictions_92day_14interval.csv'
        
        if self.log_rnn:
            self.log_rnn_dict = {}
            self.log_rnn_dict['dates'] = []
            self.log_rnn_dict['runtime'] = []
            for state in self.states:
                self.log_rnn_dict[state] = []
        
        self.include_federal = include_federal
        self.proxy_reward=proxy_reward
        if self.include_federal:
            self.federal_init = 1000
            self.federal_transfer_limit = 100
        else:
            self.federal_init = 0
            self.federal_transfer_limit = 0
        
        num_clusters = 4
        vent_per_division = 50
        #self.state_pop_dict = self.state_magnitude_mapping(num_clusters, vent_per_division)
        self.allocation_fraction = 0.1
        
        self.sim_dates_index = pd.date_range(self.sim_start_date, periods=self.sim_length-1, freq='D')
        self.loss_summary = pd.DataFrame(index = self.sim_dates_index, columns = None)
        self.loss_total = pd.DataFrame(index = self.sim_dates_index, columns = None)
        self.loss_by_state = pd.DataFrame()
        self.loss_by_state_per_capita = pd.DataFrame()
        self.model_string_list = []
        
        if self.prediction_analysis:
            self.prediction_tv_init = False
            self.prediction_loss_tv = pd.DataFrame(index = self.sim_dates_index, columns = None)
            self.prediction_loss_tv['tv_sum'] = np.nan
            self.prediction_loss_summary = pd.DataFrame(index = self.sim_dates_index, columns = None)
            self.prediction_loss_total = pd.DataFrame(index = self.sim_dates_index, columns = None)
        
        self.collect_modeling_data = collect_modeling_data
        if self.collect_modeling_data:
            self.track_states = {}
        
        self.initialize_models() # intializes self.models
        
        # Run simulation for each model
        for i, model in self.models.items():
            print("Running simulation for model: " , model)
            # return history of losses, transactions, whatever...
            # plot/summarize results... consider using a data logger to go to .csv
            self.LSTM_initial_train = False
            self.RLvalue_class_created = 0
            self.Qlearning_class_created = 0
            
            # for value itteration or q learning algo, track largest value observed by each state (init -1)
            if model[1] == 3 or model[1] == 4:
                self.largest_observed = {}
                for state in self.states:
                    self.largest_observed[state] = -1
            
            self.run_simulation(model)
        
        # compute and summarize loss totals
        #print(self.loss_summary)
        if self.loss_plots:
            self.plot_loss(self.loss_summary, title='Daily Ventilator Shortages:', total=False, error_or_shortage=1)
            self.plot_loss(self.loss_total, title='Accumulated Ventilator Shortages:', total=True, error_or_shortage=1)
        for model in list(self.loss_total):
            print("Model ", model, " total policy loss: ", self.loss_total[model].iloc[-1])
        if self.prediction_analysis == True:
            for col in list(self.prediction_loss_summary):
                for index, row in self.prediction_loss_summary.iterrows():
                    prev_day = index - pd.Timedelta(i, unit='D')
                    self.prediction_loss_total.loc[index, col] = self.prediction_loss_summary.loc[:index, col].abs().sum()
                #mse = mean_squared_error(self.prediction_loss_tv['tv_sum'].to_numpy(), self.prediction_loss_summary[col].to_numpy())
                n = len(self.prediction_loss_summary[col].to_numpy())
                print("RMSE array length: ")
                mse = np.sum(np.square(self.prediction_loss_summary[col].to_numpy()))/n
                rmse = math.sqrt(mse)
                print('Model: ', col, " RMSE: ", rmse)
            self.plot_loss(self.prediction_loss_summary, title='Inference Model Comparison: Daily Error', total=False, error_or_shortage=0)
            self.plot_loss(self.prediction_loss_total, title='Inference Model Comparison: Accumulated Error', total=True, error_or_shortage=0)
            for model in list(self.loss_total):
                print("Model: ", model, " total prediction loss: ", self.prediction_loss_total[model].iloc[-1])
                print("Model: ", model, " prediction loss mean: ", self.prediction_loss_summary[model].mean())
                print("Model: ", model, " prediction loss stdev: ", self.prediction_loss_summary[model].std())
                
    # main simulation loop for each model
    def run_simulation(self, model):
        time0 = time.time()
        inference_engine = model[0]
        policy = model[1]
        self.model_str = str(model[2])
            
        self.model_string_list.append(self.model_str)
        if self.collect_modeling_data:
            if not self.include_federal:
                self.track_states[self.model_str] = pd.DataFrame(index = self.sim_dates_index, columns=self.states)
            else:
                tmp_states = self.states.copy() + ["FEDERAL"]
                self.track_states[self.model_str] = pd.DataFrame(index = self.sim_dates_index, columns=tmp_states)
        self.loss_summary[self.model_str] = np.nan
        if self.prediction_analysis == True:
            self.prediction_loss_summary[self.model_str] = np.nan
            self.prediction_loss_total[self.model_str] = np.nan
        #xfer_q = self.initialize_transfer_queue()
        self.initialize_transfer_dict() # self.transfer_dict
        state_status_dict = self.initialize_states_trueval(self.path)
        if self.collect_modeling_data:
            self.optimal_shortage = 0
            self.total_ventillators = 0
            for state in self.states:
                self.total_ventillators = self.total_ventillators + state_status_dict[state].get_vh()
            print("total venti count: ", self.total_ventillators)
        
        date = self.sim_start_date
        
        # delete later?
        self.total_shortage = 0
        
        loss_by_state = [0] * len(self.states)
        
        for i in range(0,self.sim_length-1):
            # update state, infer need
            date = sim_start_date + pd.Timedelta(i, unit='D')
            print(date)
            if self.model_select is not None:
                #print(self.model_select)
                next_model = self.model_select['Best Performer'].loc[date]
                self.model_list = [next_model]
                #print("selecting model:", self.model_list[0], " on ", date)
                #print(self.models)
                self.initialize_models()
                #print(self.models)
                for i, model in self.models.items(): # only one here
                    inference_engine = model[0]
                    policy = model[1]
            infer_vent_needed = self.infer_need_matrix(date,inference_engine)
            immediate_vent_needed = self.infer_need_matrix(date,inference_engine=0)
            # after observing immediate need, update our observation dictionary for q learning
            if model[1] == 3 or model[1] == 4:
                for state in self.states:
                    if immediate_vent_needed[state] > self.largest_observed[state]:
                        self.largest_observed[state] = immediate_vent_needed[state]
            if self.allocation_fraction:
                venti_have_dict = {}
                for state in self.states:
                    venti_have_dict[state] = state_status_dict[state].get_vh()
                #print("checking keys")
                #print(venti_have_dict)
                self.state_possession_allocation_dict = self.state_possession_mapping(venti_have_dict)
            if self.prediction_analysis:
                self.log_prediction_loss(date,infer_vent_needed)
            #transfer_buffer_out = xfer_q.get()
            transfer_buffer_out = self.transfer_dict[date.strftime('%m/%d/%y')]
            state_status_dict = self.update_states(state_status_dict,transfer_buffer_out,send_receive=1,vent_needed=infer_vent_needed)
            # state_status_dict_quant = self.state_space_quant(state_status_dict)
            current_loss = self.calc_loss(state_status_dict, immediate_vent_needed)
            current_loss_by_state = self.calc_loss_by_state(state_status_dict, immediate_vent_needed)
            loss_by_state = self.sum_statewise_losses(loss_by_state, current_loss_by_state)
            # after calculating the current loss, adjust the Q learning offset parameter
            self.loss_summary.loc[date, self.model_str] = current_loss
            self.loss_total.loc[date, self.model_str] = self.loss_summary[self.model_str].sum()
            if self.collect_modeling_data:
                total_vr = 0
                for state in self.states:
                    #print(self.states)
                    vh = state_status_dict[state].get_vh()
                    vr = state_status_dict[state].get_vr()
                    total_vr = total_vr + vr
                    self.track_states[self.model_str].loc[date, state] = (vh, vr)
                if self.include_federal:
                    vh = state_status_dict["FEDERAL"].get_vh()
                    vr = state_status_dict["FEDERAL"].get_vr()
                    total_vr = total_vr + vr
                    self.track_states[self.model_str].loc[date, "FEDERAL"] = (vh, vr)
                optimal_vr_diff = total_vr - self.total_ventillators
                if optimal_vr_diff > 0:
                    self.optimal_shortage = self.optimal_shortage + optimal_vr_diff
                    
            transfer_buffer_in = self.compute_next_action(policy,state_status_dict,immediate_vent_needed)
            #xfer_q.put(transfer_buffer_in)
            self.update_transfer_dict(transfer_buffer_in, date)
            state_status_dict = self.update_states(state_status_dict,transfer_buffer_in,send_receive=0)
            
            time1 = time.time()
            diff = time1 - time0
            if self.log_rnn:
                self.log_rnn_dict['runtime'].append(diff)
            
        # adding population statistics per state to respective tables at end of each model sim
        self.loss_by_state[self.model_str] = pd.Series(loss_by_state)
        loss_by_state_per_capita = self.get_loss_per_state_per_capita(loss_by_state)
        self.loss_by_state_per_capita[self.model_str] = pd.Series(loss_by_state_per_capita)
    
        #print(model, " total loss: ", self.loss_total[self.model_str].iloc[-1])
        if self.prediction_analysis:
            self.prediction_tv_init = True
            
        if inference_engine == 2 and self.log_rnn == True:
            RNN_pred_df = pd.DataFrame()
            RNN_pred_df['dates'] = self.log_rnn_dict['dates']
            RNN_pred_df['runtime'] = self.log_rnn_dict['runtime']
            for state in self.states:
                RNN_pred_df[state] = self.log_rnn_dict[state]
            RNN_preds_path = self.path + self.RNN_log_filename
            RNN_pred_df.to_csv(RNN_preds_path, index=False)
            
    # returns list of total shortage per 100,000
    def get_loss_per_state_per_capita(self, loss_per_states):
        loss_by_state_per_capita = []
        pop_dict = self.state_populations_mapping()
        for i in range(len(self.states)):
            state = self.states[i]
            population = pop_dict[state]
            total_loss = loss_per_states[i]
            loss_per_100k = (total_loss / population) * 100000
            loss_by_state_per_capita.append(loss_per_100k)
        return loss_by_state_per_capita
            
    
    def calc_loss(self, state_status_dict, tv_vent_needed):
        # calculates total loss at each step
        total_loss = 0
        total_vh = 0
        total_vr = 0
        for state_name in self.states:
            state_obj = state_status_dict[state_name]
            vent_have = state_obj.get_vh()
            vent_need = tv_vent_needed[state_name]
            if vent_need > vent_have:
                shortage = vent_need - vent_have
                total_loss += shortage
            total_vh += vent_have
            total_vr += vent_need
        return total_loss
    
    def calc_loss_by_state(self, state_status_dict, tv_vent_needed):
        # calculates total loss at each step for each state
        statewise_loss = []
        for state_name in self.states:
            next_loss = 0
            state_obj = state_status_dict[state_name]
            vent_have = state_obj.get_vh()
            vent_need = tv_vent_needed[state_name]
            if vent_need > vent_have:
                next_loss = vent_need - vent_have
            statewise_loss.append(next_loss)
        return statewise_loss
    
    def sum_statewise_losses(self, total_loss, next_loss):
        for i in range(len(self.states)):
            total_loss[i] = total_loss[i] + next_loss[i]
        return total_loss
        
    def get_analysis_data(self):
        return self.track_states, self.loss_summary, self.model_string_list
    
    def best_model_lossmag(self, filter_period = 3):
        best_performer = pd.DataFrame(index = self.sim_dates_index, columns=['Best Performer'])
        model_change_count = -1
        chosen_model = None
        for i in range(0,self.sim_length-1):
            date = sim_start_date + pd.Timedelta(i, unit='D')
            # get loss sum over last 5 days for each model
            loss_sums = []
            filter_begin_date = date - pd.Timedelta(filter_period, unit='D')
            if filter_begin_date < self.sim_start_date:
                filter_begin_date = self.sim_start_date
            #print("Filter begin date: ", filter_begin_date)
            #print("Current sim date: ", date)
            for index, model_str in enumerate(self.model_string_list):
                loss_column = self.loss_summary[model_str].loc[filter_begin_date:date]
                #print(loss_column)
                loss_sums.append(loss_column.sum())
                #print(loss_sums)
            min_index = loss_sums.index(min(loss_sums))
            if isinstance(min_index, list):
                min_index = min_index[0]
            min_model = self.model_string_list[min_index]
            if min_model != chosen_model:
                model_change_count += 1
                chosen_model = min_model
            #print("min model:", min_model)
            best_performer.loc[date, 'Best Performer'] = min_model
        #print(best_performer)
        print("Number of model changes: ", model_change_count)
        return best_performer
    
    def best_model_lossgrad(self, filter_period = 3):
        best_performer = pd.DataFrame(index = self.sim_dates_index, columns=['Best Performer'])
        model_change_count = -1
        chosen_model = None
        for i in range(0,self.sim_length-1):
            date = sim_start_date + pd.Timedelta(i, unit='D')
            # get loss sum over last 5 days for each model
            loss_sums = []
            filter_begin_date = date - pd.Timedelta(filter_period, unit='D')
            if filter_begin_date < self.sim_start_date:
                filter_begin_date = self.sim_start_date
            #print("Filter begin date: ", filter_begin_date)
            #print("Current sim date: ", date)
            for index, model_str in enumerate(self.model_string_list):
                loss_start = self.loss_summary[model_str].loc[filter_begin_date]
                loss_end = self.loss_summary[model_str].loc[date]
                loss_diff = loss_end - loss_start
                #print(loss_column)
                loss_sums.append(loss_diff)
                #print(loss_sums)
            min_index = loss_sums.index(min(loss_sums))
            if isinstance(min_index, list):
                min_index = min_index[0]
            min_model = self.model_string_list[min_index]
            if min_model != chosen_model:
                model_change_count += 1
                chosen_model = min_model
            #print("min model:", min_model)
            best_performer.loc[date, 'Best Performer'] = min_model
        #print(best_performer)
        print("Number of model changes: ", model_change_count)
        return best_performer
    
    def plot_loss(self, df, title='Need a hint?', total=True, error_or_shortage=0):
        fig = plt.figure(figsize=(10,10))
        local_max = 0
        local_min = 0
        for model_col in list(df):
            val = int(model_col)
            if val <= 4:
                marker = 'o'
            elif 4 < val <= 9:
                marker = 'd'
            elif val < 15:
                marker = '^'
            else:
                marker = 'o'
            #if "max" in str(model_col):
            #    linestyle = '-.'
            #if 'min' in str(model_col):
            #    linestyle = '--'
            #if 'RNN' in str(model_col):
            #    linestyle = ':'
            #else:
            #    linestyle = '-'
            #str1 = str(self.sim_start_date)[:-9]
            #str2 = str(self.data_end_date)[:-9]
            str1 = df.index.to_series().iloc[0].strftime('%m/%d/%y')
            str2 = df.index.to_series().iloc[-1].strftime('%m/%d/%y')
            plt.plot(df.index.strftime('%m/%d/%y'), model_col, data=df, label=self.model_names[model_col], marker=marker, markersize=4)
            local_max = max(local_max, df[model_col].max())
            local_min = min(local_min, df[model_col].min())
        this_title = title + "\n" + str(len(self.states)) + " states from " + str1 + " to " + str2
        plt.title(this_title, fontsize=16)
        plt.xlabel("Date", fontsize=16)
        if total and error_or_shortage:
            plt.ylabel("Accumulated Shortage", fontsize=16)
        if not total and error_or_shortage:
            plt.ylabel("Daily Shortage", fontsize=16)
        if total and not error_or_shortage:
            plt.ylabel("Accumulated Prediction Error", fontsize=16)
            plt.grid(True, 'major', 'y', c='grey',linestyle='dashed')
        if not total and not error_or_shortage:
            plt.ylabel("Daily Prediction Error", fontsize=16)
            #print(model_col, ": \n", df[model_col])
        xtick_list = []
        for i,date in enumerate(df.index.strftime('%m/%d/%y')):
                if i % 14 == 0:
                    xtick_list.append(date)
                if date == str2:
                    xtick_list.append(date)
        plt.ylim((0, local_max+local_max*0.1))
        inc = 50 * round((local_max - local_min) * 0.1 / 50)
        #inc = 100 * round((local_max*0.1)/100)
        if inc == 0:
            inc = 10
        max_rounded = np.ceil(local_max / inc) * inc
        min_rounded = np.floor(local_min / inc) * inc
        #min_rounded = 100 * np.floor((local_min-local_min*0.1)/100)
        #max_rounded = 100 * np.ceil((local_max-local_max*0.1)/100)
        y_ticks_array = np.arange(min_rounded, max_rounded, inc)
        y_ticks_list = y_ticks_array.tolist()

        y_ticks_list_first = y_ticks_list[0]
        y_ticks_list_last = y_ticks_list[-1]
        if y_ticks_list_last < local_max:
            y_ticks_list.append(y_ticks_list_last+inc)
        if y_ticks_list_first > local_min:
            y_ticks_list.insert(0, y_ticks_list_first-inc)
        if not total and not error_or_shortage:
            ax = plt.axes()
            ax.axhline(y=0, c='grey', linestyle='dashed')
            plt.ylim(y_ticks_list[0], y_ticks_list[-1])
            y_ticks_list = y_ticks_list[1:-1]
            y_ticks_list.insert(0, local_min)
            y_ticks_list.append(local_max)
            #ax.axhline(y=local_min, c='grey', linestyle='dashed')
            #ax.axhline(y=local_max, c='grey', linestyle='dashed')
        if total and not error_or_shortage:
            ax = plt.axes()
            high = y_ticks_list[-1]
            low = y_ticks_list[0]
            buffer = (high-low)*0.05
            plt.ylim(low-buffer, high+buffer)
        plt.yticks(y_ticks_list, fontsize=14)          
        plt.xticks(xtick_list, fontsize=14, rotation=30)     
        plt.legend(fontsize=14)
            
    def display_state_histories(self, model_num, plot_cols = 1, same_range=True):
        n_states = len(self.states)
        if self.include_federal:
            n_states += 1
        plot_rows = math.ceil(n_states / plot_cols)
        #fig, axs = plt.subplots(nrows=plot_rows, ncols=plot_cols)
        fig = plt.figure(figsize=(9,7*n_states))
        #fig = plt.figure()
        model_data = self.track_states[str(model_num)]
        print(list(model_data))
        index = model_data.index
        if same_range:
            max_y = 0
            min_y = 0
            for i, state in enumerate(list(model_data)):
                vh, vr = [], []
                statewise_data = model_data[state]
                for k, (next_vh, next_vr) in statewise_data.items():
                    vh.append(next_vh)
                    vr.append(next_vr)
                next_max = max(vh + vr)
                next_min = min(vh + vr)
                if next_max > max_y:
                    max_y = next_max
                if next_min < min_y:
                    min_y = next_min
        for i, state in enumerate(list(model_data)):
            col = i % 4
            row = math.floor(i / plot_cols)
            vh, vr = [], []
            statewise_data = model_data[state]
            for k, (next_vh, next_vr) in statewise_data.items():
                vh.append(next_vh)
                vr.append(next_vr)
            state_df = pd.DataFrame(list(zip(vh, vr)),index=index, columns=['vent. supply', 'vent. demand'])
            #print(state_df)
            #print("rows:",plot_rows)
            #print("cols:",plot_cols)
            #print("index:",(i + 1))
            plt.subplot(plot_rows, plot_cols, (i + 1))
            max_y_local = 0.0
            for col_name in list(state_df):
                plt.plot(state_df.index.strftime('%m/%d/%y'), col_name, data=state_df, label=col_name, marker='o', markersize=3)
                max_y_local = max(max_y_local, state_df[col_name].max())
            #plt.plot(vh, vr)
            if same_range:
                plt.ylim((0, max_y+max_y*0.1))
                inc = 100 * round((max_y*0.2)/100)
                plt.yticks(np.arange(min_y, max_y+max_y*0.1, inc), fontsize=15)
            else:
                plt.ylim((0, max_y_local+max_y_local*0.1))
                inc = 100 * round((max_y_local*0.2)/100)
                plt.yticks(np.arange(0, max_y_local+max_y_local*0.1, inc), fontsize=15)
            xtick_list = []
            for i,date in enumerate(state_df.index.strftime('%m/%d/%y')):
                if i % 14 == 0:
                    xtick_list.append(date)
            ax1 = plt.gca()
            ax1.yaxis.grid(True)
            plt.xticks(xtick_list, fontsize=15, rotation=30)
            plt.subplots_adjust(hspace=0.5)
            plt.title(state, fontsize=20)
            plt.xlabel("Date", fontsize=18)
            plt.ylabel("Ventilator Count", fontsize=18)
            plt.legend(fontsize=16)


    #def plot_loss(self, df, title='Need a hint?'):
    #    plt.figure(figsize=(12,12))
    #    for model_col in list(df):
    #        #print(self.model_names)
    #        plt.plot(df.index, model_col, data=df, label=self.model_names[model_col], marker='o', markersize=3)
    #        plt.title(title)
    #        #print(model_col, ": \n", df[model_col])
    #    plt.legend()    
    
    # this function reads state_populations.csv and maps the population of a state to the magnitude
    # of ventillators that it can send and recieve from another state
    def state_magnitude_mapping(self, num_clusters, vent_per_division):
        pop_dict = {}
        pop_path = self.path + 'State_populations.csv'
        pop_df = pd.read_csv(pop_path)
        km = KMeans(n_clusters=num_clusters).fit(pop_df['Population'].to_numpy().reshape(-1, 1))
        cluster_centers = np.asarray(sorted(list(itertools.chain(*km.cluster_centers_.tolist()))))
        #pop_df['pop_label'] = km.labels_.tolist()
        for state in self.states:
            index = pop_df.loc[pop_df['State']==state].index[0]
            state_pop = pop_df['Population'].iloc[index]
            pop_dict[state] =  ((np.abs(cluster_centers - state_pop)).argmin() + 1) * vent_per_division
            
        pop_dict['FEDERAL'] = self.federal_transfer_limit
            
        return pop_dict
    
    def state_populations_mapping(self):
        pop_dict = {}
        pop_path = self.path + 'State_populations.csv'
        pop_df = pd.read_csv(pop_path)
        for state in self.states:
            index = pop_df.loc[pop_df['State']==state].index[0]
            state_pop = pop_df['Population'].iloc[index]
            pop_dict[state] = state_pop
        return pop_dict
        
    def state_possession_mapping(self, venti_have_dict):
        possession_dict = {}
        for state in self.states:
            possession_dict[state] = venti_have_dict[state] * self.allocation_fraction
        return possession_dict
        
    #def state_to_fips(self, data, path):
    #    fips_path = path + 'State_FIPS.csv'
    #    fips_df = pd.read_csv(fips_path)
    #    for index, row in data.iterrows():
    #        #state = row['location_name']
    #        #row['location_name'] = fips_df.loc[fips_df['State'] == state]['FIPS_Code']
    #        #print(row['location_name'])
    #        #print(fips_df.loc[fips_df['State']==row['location_name']]['FIPS Code'].item())
    #        state = row['location_name']
    #        data.loc[index,'location_name'] = fips_df.loc[fips_df['State']==state]['FIPS Code'].item()
    #    return data
    
    def update_states(self, state_status_dict, transfer_buffer, send_receive, vent_needed=None):
        if vent_needed is not None:
            for i, state in enumerate(self.states):
                state_status_dict[state].update_vr(vent_needed[state])
        tmp_state_list = self.states.copy()
        tmp_state_list.append('FEDERAL')
        #print(transfer_buffer)
        # sending
        if send_receive == 0:
            #print("SENDING - TMP STATE LIST: ", tmp_state_list)
            for y, state_y in enumerate(tmp_state_list):
                if transfer_buffer[y,y] != 0:
                    print("Simulator error 5: cannot send to self")
                total_sent = 0
                for x, state_x in enumerate(tmp_state_list):
                    total_sent += transfer_buffer[y, x]
                state_status_dict[state_y].send_ventillator(total_sent)
        # receiving
        elif send_receive == 1:
            #print("RECEIVING - TMP STATE LIST: ", tmp_state_list)
            for y, state_y in enumerate(tmp_state_list):
                if transfer_buffer[y,y] != 0:
                    print("Simulator error 6: cannot receive from self")
                total_rec = 0
                for x, state_x in enumerate(tmp_state_list):
                    total_rec += transfer_buffer[x, y]
                if state_y != 'FEDERAL':
                    state_status_dict[state_y].receive_ventillator(total_rec)
        else:
            print("Simulator error 4: incorrect send/receive specification")
        return state_status_dict
    
    def compute_next_action(self, policy, state_status_dict, immediate_vent_needed):
        vent_needed = []
        vent_available = []
        state_allocation_limit = []
        for next_state in self.states:
            vent_needed.append(state_status_dict[next_state].get_vr())
            vent_available.append(state_status_dict[next_state].get_vh())
            #print('State allocation limit:', self.state_pop_dict[next_state])
            #state_allocation_limit.append(self.state_pop_dict[next_state])
            state_allocation_limit.append(self.state_possession_allocation_dict[next_state])
        n_states = len(self.states)
        #if self.include_federal == 1:
        #    vent_needed.append(state_status_dict['FEDERAL'].get_vr())
        #    vent_available.append(state_status_dict['FEDERAL'].get_vh())
        #    state_allocation_limit.append(self.state_pop_dict['FEDERAL'])
        #    n_states += 1
        federal_available = state_status_dict['FEDERAL'].get_vh()
        if policy == -1:
            state_to_state = np.zeros((n_states+1, n_states+1))
        if policy == 0: # MAX_ALLOCATE_FIRST / MAX_ALLOCATE_FIRST_LIMIT
            (state_to_state, _) = max_allocate_first_fed(vent_needed, vent_available, n_states, state_allocation_limit, federal_available, self.federal_transfer_limit)
            #state_to_state = max_allocate_first_limit(vent_needed, vent_available, n_states, state_allocation_limit)
            #state_to_state = max_allocate_first(vent_needed, vent_available, n_states)
        if policy == 1: # MIN_ALLOCATE_FIRST / MIN_ALLOCATE_FIRST_LIMIT
            (state_to_state, _) = min_allocate_first_fed(vent_needed, vent_available, n_states, state_allocation_limit, federal_available, self.federal_transfer_limit)
            #state_to_state = min_allocate_first_limit(vent_needed, vent_available, n_states, state_allocation_limit)
            #state_to_state = min_allocate_first(vent_needed, vent_available, n_states)
        if policy == 2: # RANDOM_ALLOCATE / RANDOM_ALLOCATE_LIMIT
            (state_to_state, _) = random_allocate_fed(vent_needed, vent_available, n_states, state_allocation_limit, federal_available, self.federal_transfer_limit)
            #state_to_state = random_allocate_limit(vent_needed, vent_available, n_states, state_allocation_limit)
            #state_to_state = random_allocate(vent_needed, vent_available, n_states)
        if policy == 3: # Value iteration (RL)
            if self.RLvalue_class_created == 0:
                self.ValueItterator = ValueItterationRL(self.states, proxy_reward=self.proxy_reward)
            self.ValueItterator.update_proxy_rewards(self.largest_observed, state_status_dict, immediate_vent_needed)
            state_to_state = self.ValueItterator.get_action(vent_available, vent_needed, state_allocation_limit, federal_available, self.federal_transfer_limit)
            self.RLvalue_class_created = 1
        if policy == 4: # Q learning (Qlrn)
            if self.Qlearning_class_created == 0:
                self.Qlearning = QlearningRL(self.states, proxy_reward=self.proxy_reward)
            self.Qlearning.update_proxy_rewards(self.largest_observed, state_status_dict, immediate_vent_needed)
            state_to_state = self.Qlearning.get_action(vent_available, vent_needed, state_allocation_limit, federal_available, self.federal_transfer_limit)
            self.Qlearning_class_created = 1
            #print(state_to_state)
        return np.floor(state_to_state)
    
    def infer_need_matrix(self, date, inference_engine):
        #print("infer need matrix called")
        #print("inference for: ", date)
        (available_data_unscaled, available_data_scaled) = self.primary_dataloader.observed_data_split(date)
        #print(available_data_unscaled['California'])
        #print("Number of entries:", available_data_unscaled['California'][0].shape)
        #if self.prediction_analysis == True and self.prediction_tv_init == False:
        #    tv_sum = 0
        #    for __, (_, Y) in available_data_unscaled.items():
        #        tv_sum += Y.iloc[-1]
        #    self.prediction_loss_tv.loc[date, 'tv_sum'] = tv_sum
        statewise_need = {}
        # no inference engine, take immediate need (observed)
        if inference_engine == 0:
            for state, data in available_data_unscaled.items():
                statewise_need[state] = data[0]['ICUbed_mean'].iloc[-1]
                #statewise_need[state] = data[1].iloc[-1]
                #print(state + ": " + str(statewise_need[state]))
        # Holt linear model inference engine, need 'prediction_interval' days out
        elif inference_engine == 1:   
            statewise_need, statewise_tv = self.primary_dataloader.HOLTprediction(available_data_unscaled, self.states, self.prediction_interval)
            #self.primary_dataloader.HOLTsimulator(self.sim_start_date)
        # RNN non-linear inference engine, need 'prediction_interval' days out
        elif inference_engine == 2:
            # do initial training on all data available
            if self.LSTM_initial_train == False:
                self.LSTM_model = LSTM(input_size=self.primary_dataloader.feature_count, hidden_layer_size=300, batch_size=self.LSTM_train_batch_size, output_size=1, num_layers=1, dropout=0.0)
                if self.LSTM_dataloader.feature_count != self.primary_dataloader.feature_count:
                    print("Simulator error: LSTM training and simulator state data have different number of features")
                    print("LSTM training: ", LSTM_dataloader.feature_count)
                    print("Simulation dataloder: ", self.primary_dataloader.feature_count)
                (statewise_data_observed_unscaled, statewise_data_observed_scaled) = self.LSTM_dataloader.observed_data_split(self.sim_start_date)               
                self.dataset = PrepareDataset(statewise_data_observed_unscaled,statewise_data_observed_scaled,tw=14,only_last=False,prediction_interval=self.prediction_interval)
                dataloader = DataLoader(self.dataset, batch_size=self.LSTM_train_batch_size, shuffle=True, drop_last=True)
                self.LSTM_model.train(dataloader, epochs=self.LSTM_epochs, report_period=25, plot=self.LSTMplots, verbose=1)
                self.LSTM_initial_train = True
                print("LSTM Successfully Initialized and Trained!")
            # do additional training on last "retrain_lookback" days of data
            available_days = (date - self.data_start_date).days
            if available_days <= self.data_agg_window_length:
                (statewise_data_observed_unscaled, statewise_data_observed_scaled) = self.LSTM_dataloader.observed_data_split(date)
                #(statewise_data_observed_unscaled, statewise_data_observed_scaled) = self.primary_dataloader.observed_data_split(date)
            else:
                split_start_date = date - pd.Timedelta(self.data_agg_window_length, unit='D')
                #print("sim curent date: ", date)
                #print("sim split date: ", split_start_date)
                (statewise_data_observed_unscaled, statewise_data_observed_scaled) = self.LSTM_dataloader.observed_data_split(date, split_start_date = split_start_date)
                #print("unscaled data dims: ", np.shape(statewise_data_observed_unscaled['California'][0]))
                #print("scaled data dims: ", np.shape(statewise_data_observed_scaled['California'][0]))
                #(statewise_data_observed_unscaled, statewise_data_observed_scaled) = self.primary_dataloader.observed_data_split(date, split_start_date=split_start_date)
            self.dataset = PrepareDataset(statewise_data_observed_unscaled,statewise_data_observed_scaled,tw=14,only_last=False,prediction_interval=self.prediction_interval)
            dataloader = DataLoader(self.dataset, batch_size=self.LSTM_train_batch_size, shuffle=True, drop_last=True)
            self.LSTM_model.train(dataloader, epochs=self.LSTM_retrain_epochs, report_period=25, plot=self.LSTMplots, verbose=1)
            # infer statewise need
            LSTM_inference_only = LSTM(input_size=self.primary_dataloader.feature_count, hidden_layer_size=300, batch_size=1, output_size=1, num_layers=1, dropout=0.0)
            LSTM_inference_only.load_state_dict(self.LSTM_model.state_dict())
            statewise_need = LSTM_inference_only.inference(available_data_unscaled, available_data_scaled, self.dataset, date, len(self.states), report_accuracy=False)
            if self.log_rnn == True:
                self.log_rnn_dict['dates'].append(date)
                for state in self.states:
                    self.log_rnn_dict[state].append(statewise_need[state])
        # load RNN predictions from a previously logged RNN run
        elif inference_engine == 3:
            RNN_preds_path = self.path + self.RNN_log_filename
            RNN_pred_table = pd.read_csv(RNN_preds_path)
            RNN_pred_table['dates'] = RNN_pred_table['dates'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
            RNN_pred_table = RNN_pred_table.set_index('dates')
            for state in self.states:
                idx = date.date().strftime('%Y-%m-%d')
                statewise_need[state] = RNN_pred_table.loc[idx][state]
            
        else:
            print("Illegitimate inference engine selection...")
            return -1
        return statewise_need
    
    def log_prediction_loss(self,date,infered_need):
        (available_data_unscaled, _) = self.primary_dataloader.observed_data_split(date)
        if self.prediction_tv_init == False:
            tv_sum = 0
            for state, (_, Y) in available_data_unscaled.items():
                tv_sum += Y.iloc[-1]
            self.prediction_loss_tv.loc[date, 'tv_sum'] = tv_sum
        inference_sum = 0
        for _, state_need in infered_need.items():
            inference_sum += state_need
        #print("TV need on ", date, ": ", self.prediction_loss_tv.loc[date, 'tv_sum'])
        #print("Infered need on ", date, ": ", inference_sum)
        self.prediction_loss_summary.loc[date, self.model_str] = inference_sum - self.prediction_loss_tv.loc[date, 'tv_sum']
    
    # depreciated
    def initialize_states(self):
        state_status_dict = {}
        for state in self.states:
            if state == 'California':
                num_venti_have = 1000
            elif state == 'New York':
                num_venti_have = 1000
            elif state == 'Illinois':
                num_venti_have = 1000
            elif state == 'Florida':
                num_venti_have = 1000
            else:
                num_venti_have = 0
            state_status_dict[state] = State(state, self.sim_start_date, num_venti_have)
        state_status_dict['FEDERAL'] = State(state, self.sim_start_date, num_venti_have=self.federal_init)        
        return state_status_dict
    
    def initialize_states_trueval(self, path):
        state_ICU_dict = {}
        state_status_dict = {}
        ICU_path = path + 'State_ICU.csv'
        ICU_df = pd.read_csv(ICU_path)
        for index, row in ICU_df.iterrows():
            state = row['State']
            state_ICU_dict[state] = row['ICU beds']
        for state in self.states:
            state_status_dict[state] = State(state, self.sim_start_date, state_ICU_dict[state])
        state_status_dict['FEDERAL'] = State(state, self.sim_start_date, num_venti_have=self.federal_init)        
        return state_status_dict  
    
    # depreciated
    def initialize_transfer_queue(self):
        n_states = len(self.states)
        xfer_q = Queue(maxsize = self.ventillator_delay+1) 
        for i in range(0,self.ventillator_delay+1):
            xfer_q.put(np.zeros((n_states + 1, n_states + 1)))
        return xfer_q
    
    # new
    def initialize_transfer_dict(self):
        self.transfer_dict = {}
        self.twos = 0
        self.threes = 0
        self.fours = 0
        for i in range(0,self.sim_length-1):
            date = sim_start_date + pd.Timedelta(i, unit='D')
            date_str = date.strftime('%m/%d/%y')
            self.transfer_dict[date_str] = state_to_state = np.zeros((len(self.states)+1,len(self.states)+1))
        
    # new
    # dist floor and dis ciel in terms of number of days
    def update_transfer_dict(self, transfer_matrix, date, dist_stdev=0.5, dist_mean=3, dist_floor=2, dist_ceil=4):
        date_str = date.strftime('%m/%d/%y')
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                transfer_array = np.zeros(dist_ceil+1)
                # should be no rounding here. Transfer matrix floored before passed
                next_transfer_val = int(transfer_matrix[i,j])
                dist_array = np.random.normal(dist_mean, dist_stdev, next_transfer_val)
                dist_array = np.clip(np.rint(dist_array), dist_floor, dist_ceil).astype(int)
                for val in dist_array:
                    transfer_array[val] += 1
                    if val == 2.0:
                        self.twos += 1
                    if val == 3.0:
                        self.threes += 1
                    if val == 4.0:
                        self.fours += 1
                #print(transfer_array)       
                for day_offset in range(dist_floor, dist_ceil+1):
                    future_date = date + pd.Timedelta(day_offset, unit='D')
                    future_date_str = future_date.strftime('%m/%d/%y')
                    if future_date_str in self.transfer_dict.keys():
                        self.transfer_dict[future_date_str][i,j] = self.transfer_dict[future_date_str][i,j] + transfer_array[day_offset]
                        #xfer_array = self.transfer_dict[future_date_str]
                        #xfer_array[i,j] = xfer_val
                        #self.transfer_dict[future_date_str] = self.transfer_dict[future_date_str] + transfer_array[day_offset]
    
    def initialize_models(self):
        self.models = {}
        for _, model in enumerate(self.model_list):
            # (inference_engine, policy/model, model number)
            model = int(model)
            mod_string = str(model)
            if model == -1:
                self.models[model] = [0, -1, model]
                self.model_names[mod_string] = 'base, need now'
            if model == 0:
                self.models[model] = [0, 0, model]
                self.model_names[mod_string] = 'max, need now'
            if model == 1:
                self.models[model] = [0, 1, model]
                self.model_names[mod_string] = 'min, need now'
            if model == 2:
                self.models[model] = [0, 2, model]
                self.model_names[mod_string] = 'rand, need now'
            if model == 3:
                self.models[model] = [0, 3, model]
                self.model_names[mod_string] = 'val ittr, need now'
            if model == 4:
                self.models[model] = [0, 4, model]
                self.model_names[mod_string] = 'Qlrn, need now'    
            if model == 5:
                self.models[model] = [1, 0, model]
                self.model_names[mod_string] = 'max, HOLT'
            if model == 6:
                self.models[model] = [1, 1, model]
                self.model_names[mod_string] = 'min, HOLT'
            if model == 7:
                self.models[model] = [1, 2, model]
                self.model_names[mod_string] = 'rand, HOLT'
            if model == 8:
                self.models[model] = [1, 3, model]
                self.model_names[mod_string] = 'val ittr, HOLT'
            if model == 9:
                self.models[model] = [1, 4, model]
                self.model_names[mod_string] = 'Qlrn, HOLT'
            if model == 10:
                self.models[model] = [2, 0, model]
                self.model_names[mod_string] = 'max, RNN'
            if model == 11:
                self.models[model] = [2, 1, model]
                self.model_names[mod_string] = 'min, RNN'
            if model == 12:
                self.models[model] = [2, 2, model]
                self.model_names[mod_string] = 'rand, RNN'
            if model == 13:
                self.models[model] = [2, 3, model]
                self.model_names[mod_string] = 'val ittr, RNN'
            if model == 14:
                self.models[model] = [2, 4, model]
                self.model_names[mod_string] = 'Qlrn, RNN'
            if model == 15:
                self.models[model] = [3, 0, model]
                self.model_names[mod_string] = 'max, RNN (loaded)'
            if model == 16:
                self.models[model] = [3, 1, model]
                self.model_names[mod_string] = 'min, RNN (loaded)'
            if model == 17:
                self.models[model] = [3, 2, model]
                self.model_names[mod_string] = 'rand, RNN (loaded)'
            if model == 18:
                self.models[model] = [3, 3, model]
                self.model_names[mod_string] = 'val ittr, RNN (loaded)'
            if model == 19:
                self.models[model] = [3, 4, model]
                self.model_names[mod_string] = 'Qlrn, RNN (loaded)'
    
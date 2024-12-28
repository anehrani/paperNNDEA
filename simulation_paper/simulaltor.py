import numpy as np



class simulator:
    def __init__(self, data_size, 
                 data_real, inputs, 
                 outputs, data_v, data_u, 
                 x_noise_level=0.1, y_noise_level=0.1):
        self.data_size = data_size

        self.inputs = inputs
        self.outputs = outputs
        self.data_v = data_v
        self.data_u = data_u
        self.num_dmus = data_real.shape[1]-1
        self.num_gammas = data_real.shape[0]
        self.x_noise_level = x_noise_level
        self.y_noise_level = y_noise_level
        # sorting the real data
        self.data_real = data_real
        self.ranked_ground_truth = {}
        for i in range(self.num_gammas):
            dmu_rank_in_gamma = self.rank_dmus(data_real.iloc[i].values[1:])
            self.ranked_ground_truth.update({round(data_real.iloc[i][0],2): dmu_rank_in_gamma })

    def rank_dmus(self, datarow : list[float] ):
        """
        Rank the dmus based on the data
        """
        sorted_dmus = np.argsort(datarow)

        rank = {sorted_dmus[-1]: 1}
        for i in range(self.num_dmus - 2, -1, -1):
            if datarow[sorted_dmus[i]] == datarow[sorted_dmus[i+1]]:
                rank.update({ sorted_dmus[i]: rank[sorted_dmus[i+1]]})
            else:
                rank.update({ sorted_dmus[i]: rank[sorted_dmus[i+1]]+1})

        return rank


    def check_reanking(self, produced):

        ranked_generated_data = {}
        for i in range(self.num_gammas):
            dmu_rank_in_gamma_gen = self.rank_dmus(produced[i])
            ranked_generated_data.update({round(self.data_real.iloc[i][0],2): dmu_rank_in_gamma_gen })

        gama_comp = []
        for (gamma, ranked_s) in self.ranked_ground_truth.items():
            # assuming there are only 5 dmus
            dmu_rank_compare = []
            for (dmu, rank) in ranked_s.items():
                if rank == ranked_generated_data[gamma][dmu]:
                    dmu_rank_compare.append(1)
                else:
                    dmu_rank_compare.append(0)


            gama_comp.append(dmu_rank_compare)


        return gama_comp


    def generate_random_inputs(self):
        """
        Generate random inputs for the simulation
        """
        random_inputs = []
        for inrow in self.inputs.iterrows():
            random_x_dmu = []
            for xi in inrow[1]:
                in_x_rnd = np.random.uniform( xi - self.x_noise_level * xi,  xi + self.x_noise_level * xi, self.num_gammas)
                random_x_dmu.append(in_x_rnd)
            random_x_dmu = np.array(random_x_dmu).transpose(1,0) # change the position of gamma and inputs
            random_inputs.append(random_x_dmu)
        return random_inputs

    def generate_random_outputs(self):
        """
        Generate random inputs for the simulation
        """
        random_outputs = []
        for inrow in self.outputs.iterrows():
            random_y_dmu = []
            for yi in inrow[1]:
                in_y_rnd = np.random.uniform( yi - self.y_noise_level * yi,  yi + self.y_noise_level * yi, self.num_gammas)
                random_y_dmu.append(in_y_rnd)
            random_y_dmu = np.array(random_y_dmu).transpose(1,0) # change the position of gamma and outputs
            random_outputs.append(random_y_dmu)

        return random_outputs


    def run(self):
        sim = []

        for _ in range(self.data_size):
            rnd_x = self.generate_random_inputs()
            rnd_y = self.generate_random_outputs()

            sampled_data = []
            for iter_gamma in range(self.num_gammas):
                dmus = []
                for iter_dmu in range(1,self.num_dmus+1):
                    try:
                        uy = np.dot( self.data_u[iter_dmu][self.data_real.iloc[iter_gamma][0]] , rnd_y[iter_dmu-1][iter_gamma] )
                        vx = np.dot( self.data_v[iter_dmu][self.data_real.iloc[iter_gamma][0]], rnd_x[iter_dmu-1][iter_gamma] )
                    except:
                        print('here')
                    target = 1. if uy/vx >= 1. else uy/vx
                    dmus.append(target)
                sampled_data.append(dmus)


            #sampled_data = np.array(sampled_data)
            gama_ranking_compare = self.check_reanking( sampled_data )

            sim.append(gama_ranking_compare)


        return sim

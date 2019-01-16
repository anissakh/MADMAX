import csv
import numpy as np
from random import randint
from gurobipy import *
import time

class Knapsack_Model():

    def __init__(self, fichier_alternatives):
        self.P_Lexmax = list()             # dict solution monocritere optimale
        self.U = list()                    # list de dictionnaire d'utilite de chaque objet
        self.OPT = [0, set()]                    # tuple (valeur, elements retenus) decrivant la valeur optimale
        self.GurobiModel = None            # modele PL
        self.x_var = list()                # variables du modele
        self.list_CST= list()                # list des criteres apprises
        self.I = list()                    # point ideal
        self.N = list()                    # point nadir approche
        # self.Criteria_Weight = list()
        self.Weight = list()               # List de poids de chaque objet
        self.DM_W = None                   # vecteur de poids decrivant les preferences du decideur
        self.L_criteres = list()           # ID des criteres
        self.DM_prefered_alternative = None # Preference du decideur

        with open(fichier_alternatives) as csvfile:
            base_lignes_alternatives = csv.DictReader(csvfile, delimiter=',')
            L_criteres = list(set(base_lignes_alternatives.fieldnames) - {"id", "w"})
            L_criteres.sort()
            self.L_criteres = L_criteres
            for ligne_alternative in base_lignes_alternatives:
                # id_object = int(ligne_alternative["id"])
                del ligne_alternative["id"]
                self.Weight.append(int(ligne_alternative["w"]))
                del ligne_alternative["w"]
                l_utility = list()
                for i in range(len(ligne_alternative)):
                    l_utility.append(int(ligne_alternative["u%d"%(i+1)]))
                self.U.append(l_utility)

        self.p_obj = len(self.Weight)
        self.n_criteria = len(self.L_criteres)
        self.capacity = sum(self.Weight)/2           # capacite du sac-a-dos
        # print("WEIGHT BCPACK ",self.capacity)

    def initialize_I_N_X_star(self):
        self.I = [ None for i in range(self.n_criteria)]
        self.P_Lexmax = [ set() for i in range(self.n_criteria)]
        self.N = [ list() for i in range(self.n_criteria)]

    def set_N(self,critere, value):
        self.N[critere] = value

    def initialize_Model(self):
        self.GurobiModel = Model("MADMC")
        self.GurobiModel.setParam( 'OutputFlag', False)
        self.x_var = [self.GurobiModel.addVar(vtype=GRB.BINARY, lb=0, name="x_%d"%num) for num in range(self.p_obj )]
        self.y_var = [ quicksum(self.U[i][j] * self.x_var[i] for i in range(self.p_obj)) for j in range(self.n_criteria) ]
        self.z_var = self.GurobiModel.addVar(vtype=GRB.CONTINUOUS, lb=0, name="z")
        self.GurobiModel.update()
        # contrainte de sac-a-dos
        cst_knapsack = (quicksum([ self.Weight[i] * self.x_var[i]  for i in range(self.p_obj)])  <= self.capacity)
        self.list_CST.append(cst_knapsack)
        for cst  in self.list_CST :
            self.GurobiModel.addConstr( cst )

        self.GurobiModel.update()

    def compute_I_and_N_once(self):
        self.initialize_I_N_X_star()
        m = Model("monoObj")
        x_local_var = [m.addVar(vtype=GRB.BINARY, lb=0, name="x_%d" % num) for num in range(1, self.p_obj + 1)]
        m.addConstr(quicksum( [self.Weight[i] * x_local_var[i] for i in range(self.p_obj)]) <= self.capacity)  # contrainte de sac-a-dos
        m.update()

        for i in range(self.n_criteria):
            Obj = quicksum([self.U[j][i]* x_local_var[j] for j in range(self.p_obj)])
            m.setObjective(Obj, GRB.MAXIMIZE)
            m.setParam('OutputFlag', False)
            m.update()
            m.optimize()

            sol = set()
            for j in range(self.p_obj):
                if x_local_var[j].x == 1:
                    sol.add(j)
            self.I[i] = int(m.objVal)
            self.P_Lexmax[i] = sol

            for j in range(self.n_criteria):
                if j !=i :
                    self.N[j].append(sum([self.U[o][j]  for o in sol ]))

        self.N = np.array([ min(self.N[i]) for i in range(self.n_criteria)])
        self.I = np.array(self.I)

    def upload_criteria_weight(self, weight_file="weight_file_knapsack.csv"):
        with open(weight_file) as csvfile:
            ligne_poids = csv.DictReader(csvfile, delimiter=',')
            D_weight = ligne_poids.next()
            # self.Criteria_Weight = {f : float(D_weight[value]) for f,value in self.Criteria_Weight.items()}
            self.Criteria_Weight = np.array([float(D_weight[criteria]) for criteria in self.L_criteres])
        # print("Uploading weights to use from {}...\n\t{}\n\t{}".format(weight_file, self.L_criteres, self.Criteria_Weight))


    def upload_DM_weight_preference(self, UnknownDMAgregationFunctionFile='DM_weights_file_knapsack_2.csv'):
        # print("Uploading DM linear agregation function from {} ...".format(UnknownDMAgregationFunctionFile))
        with open(UnknownDMAgregationFunctionFile) as csvfile:
            ligne_poids = csv.DictReader(csvfile, delimiter=',')
            W = ligne_poids.next()
            self.DM_W = np.array([float(W[criteria]) for criteria in self.L_criteres])

        self.DM_prefered_alternative = None

        m = Model("DM_Pref")
        Obj = quicksum(self.DM_W[j] * self.y_var[j] for j in range(self.n_criteria))
        m.setObjective(Obj,GRB.MAXIMIZE)
        m.update()
        m.optimize()
        sol = set()
        for i in range(self.p_obj):
            if self.x_var[i].x == 1:
                sol.add(i)
        self.DM_prefered_alternative = np.array([ sum([self.U[i][j] for i in sol ]) for j in range(self.n_criteria)]),sol

        # print("\tDM unknown preference is {} :\n\t{}\n\t{}".format(self.D_IdToMod[self.DM_prefered_alternative], self.L_criteres,
        #                                                            self.M_Points[self.DM_prefered_alternative]))


    def criteria_to_improve(self):
        dif = (self.OPT[0] - self.DM_prefered_alternative) / (self.N - self.I)

        criteria_id  = 0
        max_value = dif[0]
        for i in range(1, self.n_criteria):
            if dif[i] > max_value:
                max_value = dif[i]
                criteria_id = i
        print("\t{} is too low for DM !\n\n".format(self.L_criteres[criteria_id]))

        return criteria_id


    def set_criteria_to_improve(self, criteria_id=2): #2 = euros
        value_of_this_criteria_in_current_opt = self.OPT[0][criteria_id]
        self.set_N(criteria_id,value_of_this_criteria_in_current_opt)
        cst = self.y_var[criteria_id]
        self.list_CST.append(cst >= value_of_this_criteria_in_current_opt)


    def add_Tchebycheff_CST(self, point_reference=None):

        if point_reference == None:
            point_reference = self.I

        for j in range(self.n_criteria):
            if (self.N[j] != self.I[j]):
                Lambda = 1 / (self.N[j] - self.I[j])  # self.Criteria_Weight[j] / (self.N[j] - self.I[j])
            else:
                Lambda = 1

            cst =  Lambda * self.y_var[j] - point_reference[j]
            self.GurobiModel.addConstr(self.z_var >= cst)


    def nearest_point_to_I(self, reference_point=None, epsilon=0.1):
        self.initialize_I_N_X_star()
        self.compute_I_and_N_once()
        self.initialize_Model()
        # self.upload_criteria_weight()
        # print("IDEAL ",self.I)
        # print("NADIRE ",self.N)
        # print("Cr_WEIGHT ",self.Criteria_Weight)

        if( np.array_equal(self.I, self.N)  ):
            # print("ONLY SOLUTION ",self.I)
            return

        Obj = self.z_var + epsilon * quicksum( [( self.y_var[j]  - self.I[j] ) / ( self.N[j] - self.I[j])  for j in range(self.n_criteria)  if self.N[j] != self.I[j] ])
                                     # quicksum(self.Criteria_Weight[j] * (self.y_var[j] - self.I[j]) / (self.N[j] - self.I[j])  for j in range(self.n_criteria) )
        self.GurobiModel.setObjective(Obj, GRB.MINIMIZE)
        self.GurobiModel.update()

        self.add_Tchebycheff_CST(reference_point)

        self.GurobiModel.optimize()

        for i in range(self.p_obj) :
            if self.x_var[i].x == 1 :
                self.OPT[1].add(i)

        # self.OPT[0] = self.GurobiModel.objVal
        self.OPT[0] = np.array([ sum([self.U[i][j] for i in self.OPT[1] ]) for j in range(self.n_criteria)])
        # print(self.OPT)

    def start_exploration(self):
        # self.upload_criteria_weight()
        self.upload_DM_weight_preference()

        while True:
            self.nearest_point_to_I()

            if self.OPT[1] == self.DM_prefered_alternative[1]:
                print("Exploration stopped : DM preference found!!!")
                return

            criteria_to_improve = self.criteria_to_improve()
            self.set_criteria_to_improve(criteria_to_improve)


    def nearest_alternative_to_a_reference_point(self, weight_file="weight_file.csv", performance_file="performance_cible_knapsack.csv", epsilon=0.1):
        # self.upload_criteria_weight(weight_file)
        with open(performance_file) as csvfile:
            ligne_performance = csv.DictReader(csvfile, delimiter=',')
            P = ligne_performance.next()
            reference_point = [int(P[criteria]) for criteria in self.L_criteres]

        n_p = self.nearest_point_to_I(reference_point=reference_point, epsilon=epsilon)
        return n_p


    @staticmethod
    def generate_knapsack_instance(n,p,filename = "knapsack_instance.csv"):
        weight_Obj = [ randint(1,25) for i in range(p)]
        # utiliy_Obj = [ randint(1, 15) for i in range(p)]

        with open(filename, 'w') as csvfile:
            fieldnames = ['id', 'w' ] + [ 'u%d'%i for i in range(1,n+1)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for i in range(p):
                row = dict()
                row["id"] = i
                row["w"] = str(weight_Obj[i])
                for j in range(1, n+1) :
                    row["u%d"%j] = str(randint(1,20))
                writer.writerow(row)


def write_in_file(filename,n,p,t2):

    with open(filename,"a") as f:
        f.write(str(n)+" "+str(p)+" "+str(t2)+"\n")


if __name__ == '__main__':

    for n in range(2, 200, 5):  # [2,3,4,5,7,9,12,14,17,20,24,28,35] :
        for p in range(3, 200, 5):  # [3,5,10,15,20,25,30,40,50,60,70,100] :
            Knapsack_Model.generate_knapsack_instance(n, p)
            Knapsack_Model.generate_knapsack_instance(n,p)
            knapsack = Knapsack_Model("knapsack_instance.csv")
            t1 = time.time()
            knapsack.nearest_point_to_I()
            t2 = time.time() - t1

            write_in_file("result.txt",n,p,t2)

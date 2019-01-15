import csv
import numpy as np
from random import randint

class Knapsack_Model():

    def __init__(self, fichier_alternatives, capacity):
        self.P_Lexmax = list()             # dict solution monocritere optimale
        self.U = list()                    # list de dictionnaire d'utilite de chaque objet
        self.OPT = None                    # tuple (valeur, elements retenus) decrivant la valeur optimale
        self.GurobiModel = None            # modele PL
        self.x_var = list()                # variables du modele
        self.z_var = list()                # to linearize the model
        self.I = list()                    # point ideal
        self.N = list()                    # point nadir approche
        # self.Criteria_Weight = list()
        self.capacity = capacity           # capacite du sac-a-dos
        self.Weight = list()               # List de poids de chaque objet
        self.DM_W = None                   # vecteur de poids decrivant les preferences du decideur
        self.L_criteres = list()           # ID des criteres
        # self.E = None

        with open(fichier_alternatives) as csvfile:
            base_lignes_alternatives = csv.DictReader(csvfile, delimiter=',')
            L_criteres = list(set(base_lignes_alternatives.fieldnames) - {"id", "w"})
            L_criteres.sort()
            self.L_criteres = L_criteres
            for ligne_alternative in base_lignes_alternatives:
                # id_object = int(ligne_alternative["id"])
                del ligne_alternative["id"]
                self.Weight.append(ligne_alternative["w"])
                del ligne_alternative["w"]
                l_utility = list()
                for i in range(len(ligne_alternative)):
                    l_utility.append(int(ligne_alternative["u%d"%(i+1)]))
                self.U.append(l_utility)

        self.p_obj = len(self.Weight)
        self.n_criteria = len(self.L_criteres)

    def initialize_I_N_X_star(self):
        self.I = [ None for i in range(self.n_criteria)]
        self.P_Lexmax = [ set() for i in range(self.n_criteria)]
        self.N = [ list() for i in range(self.n_criteria)]

        # self.I = {fieldname: None for fieldname in self.L_criteres }
        # self.P_Lexmax = {fieldname: None for fieldname in self.L_criteres}
        # self.N = {fieldname: list() for fieldname in self.L_criteres}



    def initialization(self, UnknownDMAgregationFunctionFile='DM_weights_file_knapsack.csv'):
        with open(UnknownDMAgregationFunctionFile) as csvfile:
            ligne_poids = csv.DictReader(csvfile, delimiter=',')
            W = ligne_poids.next()
            self.DM_W = [int(W[criteria]) for criteria in self.L_criteres]
            #Verification si strictement positifs et somme a 1

        self.GurobiModel = Model("MADMC")
        self.GurobiModel.setParam( 'OutputFlag', False)
        self.x_var = [self.GurobiModel.addVar(vtype=GRB.BINARY, lb=0, name="x_%d"%num) for num in range(1,self.n_criteria + 1)]
        self.z_var = [self.GurobiModel.addVar(vtype=GRB.CONTINUOUS, lb=0, name="z_%d"%num) for num in range(1, self.n_criteria + 1)]
        self.GurobiModel.update()

        self.GurobiModel.addConstr( quicksum([ self.Weight[i] * self.x_var[i]  for i in range(self.p_obj)])  <= self.capacity)   #contrainte de sac-a-dos
        # for i in range(1,self.n_criteria + 1):
        #     c1 =
        #     self.GurobiModel.addConstr( c1 <= 0 )
        self.GurobiModel.update()




    def compute_I_and_N(self):
        self.initialize_I_N_X_star()
        m = Model("monoObj")
        x_local_var = [m.addVar(vtype=GRB.BINARY, lb=0, name="x_%d" % num) for num in range(1, self.p_obj + 1)]
        m.addConstr(quicksum( [self.Weight[i] * x_local_var[i] for i in range(self.p_obj)]) <= self.capacity)  # contrainte de sac-a-dos
        m.update()

        for i in range(self.n_criteria):
            Obj = quicksum([self.U[j][i] for j in range(self.p_obj)])
            m.setObjective(Obj, GRB.MAXIMIZE)
            m.update()
            m.optimize()

            sol = set()
            for j in range(self.p_obj):
                if x_local_var[j].x == 1:
                    sol.append(j)
            self.I[i] = m.objVal
            self.P_Lexmax[i] = sol

            for j in range(self.n_criteria):
                if j !=i :
                    self.N[j].append(sum([self.U[o][j]  for o in sol ]))


        self.N = [ max(self.N[i]) for i in range(self.n_criteria)]
        for i in range(self.n_criteria) :
            self.N[i] = min(self.N[i])


        self.dif_n_i = self.N - self.I



    def upload_criteria_weight(self, weight_file="weight_file.csv"):
        with open(weight_file) as csvfile:
            ligne_poids = csv.DictReader(csvfile, delimiter=',')
            D_weight = ligne_poids.next()
            # self.Criteria_Weight = {f : float(D_weight[value]) for f,value in self.Criteria_Weight.items()}
            self.Criteria_Weight = np.array([float(D_weight[criteria]) for criteria in self.L_criteres])
        print("Uploading weights to use from {}...\n\t{}\n\t{}".format(weight_file, self.L_criteres, self.W))






    @staticmethod
    def generate_knapsack_instance(n,p,filename = "knapsack_instance.csv"):
        weight_Obj = [ randint(1,60) for i in range(p)]
        utiliy_Obj = [ randint(1, 15) for i in range(p)]

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

        return int(sum(weight_Obj)/p)

if __name__ == '__main__':
    knapsack = Knapsack_Model("knapsack_instance.csv",23)
    # Knapsack_Model.generate_knapsack_instance(3,10)
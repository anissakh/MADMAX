import matplotlib.pyplot as plt
import numpy as np



def read_testing(filename="result.txt"):
    N = list()
    P = list()
    T = list()
    with open(filename,"r") as f:
        for line in f.readlines():
            line = line.split(" ")
            N.append(int(line[0]))
            P.append(int(line[1]))
            T.append(float(line[2]))

    return N,P,T


# N,P,T = read_testing("result_n.txt")
# fig = plt.figure(figsize=(10, 5))
# plt.grid()
# plt.title("Temps de resolution pour des instances de nombre de criteres variable (p=300)")
# plt.xlabel("n criteres")
# plt.ylabel("temps (seconde)")
# plt.plot(N,T,label="p=300")
# # plt.show()
# plt.savefig("plot_n_var.png",dpi=fig.dpi)

N,P,T = read_testing("result_p.txt")
fig = plt.figure(figsize=(10, 5))
plt.grid()
plt.title("Temps de resolution pour des instances de tailles variable (n=200)")
plt.xlabel("p objets")
plt.ylabel("temps (seconde)")
plt.plot(P,T,color="red",label="p=300")
# plt.show()
plt.savefig("plot_p_var.png",dpi=fig.dpi)

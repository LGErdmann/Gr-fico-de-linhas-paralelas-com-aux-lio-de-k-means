import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
#GA
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.pntx import TwoPointCrossover
#Problem
from pymoo.problems.many.wfg import WFG5
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

populacao = input("Tamanho da população: ")
geracao = input("Número de gerações: ")
numero = input("Número de agrupamentos: ")
def problem(populacao, geracao):
	wfg = WFG5(n_var=20, n_obj=4)
	problem = wfg

	ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=4)
	algorithm = NSGA3(pop_size=int(populacao),
	                  ref_dirs=ref_dirs,
	                  crossover=TwoPointCrossover())

	res = minimize(problem, algorithm, seed=1, termination=('n_gen', int(geracao)))
	Pop = res.F
	return Pop

x = problem(populacao, geracao)

x = problem(populacao, geracao)
df = pd.DataFrame(x , columns=['F1', 'F2', 'F3', 'F4'])

#Simulando um CSV já pronto
df.to_csv('data.csv', index=False)

# Carregar dados de exemplo
data = pd.read_csv('data.csv')

# Executar o algoritmo K-means
kmeans = KMeans(n_clusters=int(numero))
data['cluster'] = kmeans.fit_predict(x)

# Criar o gráfico de barras paralelas
fig = px.parallel_coordinates(data, color="cluster", dimensions=['F1', 'F2', 'F3', 'F4'], 
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=2)

# Exibir o gráfico
fig.show()
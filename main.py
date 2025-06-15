import numpy as np  # Biblioteca para cálculos numéricos e manipulação de arrays
import matplotlib.pyplot as plt  # Biblioteca para criação de gráficos
import pandas as pd  # Biblioteca para manipulação de tabelas de dados
import imageio  # Biblioteca para criação de GIFs animados
import os  # Biblioteca para manipulação de arquivos e diretórios
import sys  # Biblioteca para interação com argumentos de linha de comando
import json  # Biblioteca para manipulação de arquivos JSON

#Grupo:
# Álvaro José Souza Gomes - 202465095A
# Arthur Augusto de Araujo Brito - 202465006A

# Leitura do arquivo JSON contendo os parâmetros do modelo
json_file = sys.argv[1]  # Nome do arquivo JSON passado como argumento
with open(json_file, "r") as f:
            params = json.load(f) # Carregar os parâmetros em um dicionário

# Constantes do modelo Hodgkin-Huxley (retiradas do arquivo JSON)
C_m = params["cm"] # Capacitância da membrana
a = params["a"] # Raio do axônio
R = params["rl"] # Resistência axial por unidade de comprimento
g_Na = params["gna"] # Condutância de canais de sódio
g_K = params["gk"] # Condutância de canais de potássio
g_L = params["gl"] # Condutância de canais de vazamento
E_Na = params["ena"] # Potencial de reversão do sódio
E_K = params["ek"] # Potencial de reversão do potássio
E_L = params["el"] # Potencial de reversão do vazamento
temp = params["T_max"] # Tempo total de simulação
L = params["L_max"] # Comprimento total do axônio
dt = params["dt"] # Passo temporal
dx = params["dx"] # Passo espacial
V_m0 = params["vm0"] # Potencial inicial da membrana
m0 = params["m0"] # Estado inicial do gate m (sódio)
h0 = params["h0"] # Estado inicial do gate h (sódio)
n0 = params["n0"] # Estado inicial do gate n (potássio)
J = np.array(params["J"]) # Corrente externa
Mie = np.array(params["Mie"]) # Máscara para mielina
lambida = a / (R * 2*C_m) # Constante da equação do cabo

# Parte 1: Modelo sem bainha de mielina
# Configuração dos intervalos de tempo e espaço
t = np.arange(0, temp, dt) # Intervalos de tempo de 0 até temp, com passo dt
x = np.arange(0, L, dx) # Intervalos espaciais de 0 até L, com passo dx
n_x = int(L / dx) # Número de posições no espaço
n_t = int(temp / dt) # Número de passos no tempo

# Funções para as Equações dos canais de Ka e Na
def I_ap(temp, x):
    # Estímulo aplicado na primeira posição do espaço
    return 20.0 if (x == dx) else 0.0

def alpha_n(V):
    # Taxa de transição de abertura da porta n.
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V):
    # Taxa de transição de fechamento da porta n.
    return 0.125 * np.exp(-(V + 65) / 80)

def alpha_m(V):
    # Taxa de transição de abertura da porta m.
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V):
    # Taxa de transição de fechamento da porta m.
    return 4.0 * np.exp(-(V + 65) / 18)

def alpha_h(V):
    # Taxa de transição de abertura da porta h.
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V):
    # Taxa de transição de fechamento da porta h.
    return 1 / (1 + np.exp(-(V + 35) / 10))

# Vetores que recebem os respectivos valores
Vm = np.zeros(n_x) + m0 # Inicialização do gate m
Vh = np.zeros(n_x) + h0 # Inicialização do gate h
Vn = np.zeros(n_x) + n0 # Inicialização do gate n
V = np.zeros((n_t, n_x)) # Potencial de membrana ao longo do tempo e espaço
V[0, :] = V_m0 # Definição do potencial inicial

# Simulação do modelo sem bainha de mielina
for i in range(n_t - 1):
    V[i,0] = V[i,1] # Condição de contorno à esquerda
    V[i,-1] = V[i,-2] # Condição de contorno à direita
    Vn[0], Vn[-1] = Vn[1], Vn[-2] # Atualização do gate n nas bordas
    Vm[0], Vm[-1] = Vm[1], Vm[-2] # Atualização do gate m nas bordas
    Vh[0], Vh[-1] = Vh[1], Vh[-2] # Atualização do gate h nas bordas

    for j in range(1, n_x - 1):
      # Atualização dos gates dos canais
      Vn[j] += dt*(alpha_n(V[i,j])*(1 - Vn[j]) - beta_n(V[i,j])*Vn[j])
      Vm[j] += dt*(alpha_m(V[i,j])*(1 - Vm[j]) - beta_m(V[i,j])*Vm[j])
      Vh[j] += dt*(alpha_h(V[i,j])*(1 - Vh[j]) - beta_h(V[i,j])*Vh[j])
      
      # Cálculo das correntes iônicas
      I_Na = g_Na * Vm[j]**3 * Vh[j] * (V[i, j] - E_Na)
      I_K = g_K * Vn[j]**4 * (V[i, j] - E_K)
      I_l = g_L * (V[i, j] - E_L)
      I_est = I_ap(dt * i, dx * j)
      
      # Cálculo da segunda derivada espacial
      dv2_d2x = (V[i,j-1] - 2*V[i,j] + V[i,j+1]) / dx**2
      
      # Atualização do potencial de membrana
      V[i+1,j] = V[i, j] + dt * (lambida * dv2_d2x + (I_est - I_Na - I_K - I_l) / C_m)

# Criar a tabela com Pandas após a simulação
tabela = pd.DataFrame(V, columns=[f"x={xi:.2f}mm" for xi in x]) # Criação de tabela com colunas representando posições no espaço
tabela.index = tabela.index * dt  # Ajustar índice para representar o tempo em ms
tabela.index.name = "Tempo (ms)" # Nomear o índice para indicar que representa o tempo

# Exibir a tabela no console
print(tabela)

# Salvar a tabela em um arquivo CSV
tabela.to_csv("tabela_voltagem_distancia_tempo_sem_bainha.csv", index=True) # Exportar a tabela para um arquivo CSV

# Gerar os gráficos para criar o GIF
temp_files = [] # Lista para armazenar nomes dos arquivos temporários
for i in range(0, n_t, 100): # Iterar em passos de 100 para reduzir o número de quadros
    plt.figure(figsize=(8, 4)) # Configurar o tamanho da figura
    plt.plot(V[i, 1:-1], label=f'Tempo = {i*dt:.1f} (ms)') # Plotar o potencial de membrana para o instante de tempo atual

    # Configurações do gráfico
    plt.xlabel('Espaço (mm)') # Rótulo do eixo x
    plt.ylabel('Voltagem (mV)') # Rótulo do eixo y
    plt.ylim(-100, 100) # Limites do eixo y
    plt.title('Propagação da Voltagem sem Bainha de Mielina') # Título do gráfico
    plt.legend() # Adicionar legenda

    # Salvar o gráfico como imagem temporária
    temp_file = f"frame_{i}.png" # Nome do arquivo temporário
    plt.savefig(temp_file) # Salvar o gráfico
    temp_files.append(temp_file) # Adicionar o nome do arquivo à lista
    plt.close() # Fechar o gráfico para liberar memória

# Criar o GIF utilizando as imagens geradas
output_gif_path = "propagacao_potencial_sem_mielina.gif" # Nome do arquivo GIF de saída
with imageio.get_writer(output_gif_path, mode="I", duration=0.01) as writer: # Configurar o writer do GIF
    for temp_file in temp_files: # Iterar sobre os arquivos temporários
        image = imageio.v2.imread(temp_file) # Ler a imagem
        writer.append_data(image) # Adicionar a imagem ao GIF

# Excluir os arquivos temporários após a criação do GIF
for temp_file in temp_files:
        os.remove(temp_file) # Remover o arquivo temporário

# Imprimir a mensagem de conclusão e o caminho do GIF
print(f"GIF salvo em: {output_gif_path}")

# Parte 2: Modelo com bainha de mielina
lambida_base = a / (R * 2) # Recalcular a constante lambida sem levar em conta a capacitância
lambida = np.where(Mie == 1, lambida_base * 10, lambida_base)  # Aumentar lambida nas regiões mielinizadas

# Ajustar a capacitância e as condutâncias para regiões mielinizadas
C_m_array = np.where(Mie == 1, C_m / 10, C_m) # Reduzir a capacitância nas regiões mielinizadas
g_Na_array = np.where(Mie == 1, g_Na / 10, g_Na) # Reduzir a condutância de sódio
g_K_array = np.where(Mie == 1, g_K / 10, g_K) # Reduzir a condutância de potássio
g_L_array = np.where(Mie == 1, g_L / 10, g_L) # Reduzir a condutância de vazamento

# Simulação do modelo com bainha de mielina
for i in range(n_t - 1):
    V[i,0] = V[i,1] # Condição de contorno à esquerda
    V[i,-1] = V[i,-2] # Condição de contorno à direita
    Vn[0], Vn[-1] = Vn[1], Vn[-2] # Atualizar gate n nas bordas
    Vm[0], Vm[-1] = Vm[1], Vm[-2] # Atualizar gate m nas bordas
    Vh[0], Vh[-1] = Vh[1], Vh[-2] # Atualizar gate h nas bordas

    for j in range(1, n_x - 1):
      # Atualizar os gates com as taxas de transição
      Vn[j] += dt*(alpha_n(V[i,j])*(1 - Vn[j]) - beta_n(V[i,j])*Vn[j])
      Vm[j] += dt*(alpha_m(V[i,j])*(1 - Vm[j]) - beta_m(V[i,j])*Vm[j])
      Vh[j] += dt*(alpha_h(V[i,j])*(1 - Vh[j]) - beta_h(V[i,j])*Vh[j])
      
      # Calcular as correntes nas regiões mielinizadas e não mielinizadas
      I_Na = g_Na_array[j] * Vm[j]**3 * Vh[j] * (V[i, j] - E_Na)
      I_K = g_K_array[j] * Vn[j]**4 * (V[i, j] - E_K)
      I_l = g_L_array[j] * (V[i, j] - E_L)
      I_est = I_ap(dt * i, dx * j)
      
      # Cálculo da segunda derivada espacial
      dv2_d2x = (V[i,j-1] - 2*V[i,j] + V[i,j+1]) / dx**2
      
      # Atualização do potencial de membrana com ajustes para mielina
      V[i+1,j] = V[i, j] + dt * ((lambida[j] * dv2_d2x + (I_est - I_Na - I_K - I_l)) / C_m_array[j])

# Criar uma tabela com os dados da simulação com mielina
tabela = pd.DataFrame(V, columns=[f"x={xi:.2f}mm" for xi in x]) # Criar tabela com as colunas correspondendo ao espaço
tabela.index = tabela.index * dt  # Ajustar índice para representar o tempo em ms
tabela.index.name = "Tempo (ms)" # Nomear o índice como "Tempo (ms)"

# Exibir a tabela no console
print(tabela)

# Salvar a tabela em um arquivo CSV
tabela.to_csv("tabela_voltagem_distancia_tempo_com_mielina.csv", index=True) # Exportar a tabela como CSV

# Gerar os gráficos para criar o GIF com mielina
temp_files = [] # Lista para armazenar nomes dos arquivos temporários
for i in range(0, n_t, 100): # Iterar em passos de 100 para reduzir o número de quadros
    plt.figure(figsize=(8, 4)) # Configurar o tamanho da figura
    plt.plot(V[i, 1:-1], label=f'Tempo = {i*dt:.1f} (ms)') # Plotar o potencial de membrana para o instante de tempo atual

    # Configurações do gráfico
    plt.xlabel('Espaço (mm)') # Rótulo do eixo x
    plt.ylabel('Voltagem (mV)') # Rótulo do eixo y
    plt.ylim(-100,100) # Limites do eixo y
    plt.title('Propagação da Voltagem com Bainha de Mielina') # Título do gráfico
    plt.legend() # Adicionar legenda

    # Salvar o gráfico como imagem temporária
    temp_file = f"frame_{i}.png" # Nome do arquivo temporário
    plt.savefig(temp_file) # Salvar o gráfico
    temp_files.append(temp_file) # Adicionar o nome do arquivo à lista
    plt.close() # Fechar o gráfico para liberar memória

# Criar o GIF utilizando as imagens geradas
output_gif_path = "propagacao_potencial_com_mielina.gif" # Nome do arquivo GIF de saída
with imageio.get_writer(output_gif_path, mode="I", duration=0.01) as writer: # Configurar o writer do GIF
    for temp_file in temp_files: # Iterar sobre os arquivos temporários
        image = imageio.v2.imread(temp_file) # Ler a imagem
        writer.append_data(image) # Adicionar a imagem ao GIF

# Excluir os arquivos temporários após a criação do GIF
for temp_file in temp_files:
        os.remove(temp_file) # Remover o arquivo temporário

# Imprimir a mensagem de conclusão e o caminho do GIF
print(f"GIF salvo em: {output_gif_path}")

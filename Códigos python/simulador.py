from scipy.integrate import solve_ivp
from scipy.optimize import fsolve,root,least_squares,minimize
from numpy import array,arcsin,sqrt,sin,cos,tan,arctan2,clip,ceil,rad2deg,mod
import matplotlib.pyplot as plt
from math import pi
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

 



def MatrizTransformacoesVentoCorpo(alfa,beta):
    return array([[cos(alfa)*cos(beta),-cos(alfa)*sin(beta),-sin(alfa)],
                             [sin(beta),cos(beta),0],
                             [sin(alfa)*cos(beta),-sin(alfa)*sin(beta),cos(alfa)]])
def MatrizTransformacoesVentoCorpoFORCA(alfa,beta):
    return array([[cos(alfa)/cos(beta),-cos(alfa)*tan(beta),-sin(alfa)],
                             [0,1,0],
                             [sin(alfa)/cos(beta),-sin(alfa)*tan(beta),cos(alfa)]])
def stall(t,y):
    return abs(sqrt(y[0]**2+y[1]**2+y[2]**2) - 50)
def stall_angle(t,y):
    return abs(arctan2(y[2], max(y[0], 1e-6)) - 12*pi/180)

def hit_ground(t,y):
    return y[11]
hit_ground.terminal = True
stall.terminal = True
stall_angle.terminal = True

def comparar_simulacoes(avioes, lista_labels, t_fim=100.0, savefile=None,perturbacao=None):
    """
    Roda simulações e plota cada variável em FIGURAS SEPARADAS.
    """


    # --- Estilo ---
    plt.rcdefaults()
    plt.style.use('seaborn-v0_8-ticks')

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
    })
    import seaborn as sns
    sns.set_context("paper", font_scale=1)

    # --- Títulos e labels ---
    titles = {
        'u': r'Velocidade longitudinal $u$',
        'v': r'Velocidade lateral $v$',
        'w': r'Velocidade vertical $w$',
        'p': r'Taxa de rolamento $p$',
        'q': r'Taxa de arfagem $q$',
        'r': r'Taxa de guinada $r$',
        'phi': r'Ângulo de rolamento $\phi$',
        'theta': r'Ângulo de arfagem $\theta$',
        'psi': r'Ângulo de guinada $\psi$',
        'x_E': r'Posição inercial $x_E$',
        'y_E': r'Posição inercial $y_E$',
        'h_E': r'Altitude $h_E$',
        'alpha': r'Ângulo de ataque $\alpha$',
        'beta': r'Ângulo de derrapagem $\beta$',
        'V_inf': r'Velocidade total $V_\infty$',
    }

    latex_labels = {
        'u': r'$u \; [\mathrm{m/s}]$', 'v': r'$v \; [\mathrm{m/s}]$', 'w': r'$w \; [\mathrm{m/s}]$',
        'p': r'$p \; [^\circ/\mathrm{s}]$', 'q': r'$q \; [^\circ/\mathrm{s}]$', 'r': r'$r \; [^\circ/\mathrm{s}]$',
        'phi': r'$\phi \; [\mathrm{rad}]$', 'theta': r'$\theta \; [\mathrm{rad}]$', 'psi': r'$\psi \; [\mathrm{rad}]$',
        'x_E': r'$x_E \; [\mathrm{m}]$', 'y_E': r'$y_E \; [\mathrm{m}]$', 'h_E': r'$h_E \; [\mathrm{m}]$',
        'alpha': r'$\alpha \; [^\circ]$', 'beta': r'$\beta \; [^\circ]$',
        'V_inf': r'$V_\infty \; [\mathrm{m/s}]$',
    }

    order = [
        'u','v','w','p','q','r','phi','theta','psi',
        'x_E','y_E','h_E','alpha','beta','V_inf'
    ]

    # --- Simulações ---
    resultados = []
    for i, aviao in enumerate(avioes):
        sim = SimuladorDeVoo(aviao, perturbacao)
        t, y = sim.run_simulation(t_fim=t_fim)

        y = array(y)
        if y.shape[0] == 12 and y.shape[1] > 12:
            y = y.T

        label = lista_labels[i] if lista_labels else getattr(aviao, 'name', f"Cenário {i+1}")
        resultados.append((t, y, label))

    # --- Função auxiliar ---
    def get_var_formatted(var, y):
        u, v, w = y[:, 0], y[:, 1], y[:, 2]

        if var == 'h_E':
            return -y[:, 11]

        elif var == 'alpha':
            return rad2deg(arctan2(w, u))

        elif var == 'beta':
            V = sqrt(u**2 + v**2 + w**2)
            beta = arcsin(v / V)
            return rad2deg(beta)

        elif var == 'V_inf':
            return sqrt(u**2 + v**2 + w**2)

        elif var in ['p','q','r']:
            idx = order.index(var)
            return rad2deg(y[:, idx])

        elif var in ['phi','theta','psi']:
            idx = order.index(var)
            val = y[:, idx]

            if var == 'phi':
                val = val
            elif var == 'theta':
                val = val
            elif var == 'psi':
                val = val

            return val

        else:
            return y[:, order.index(var)]

   
    colors = ['k', "#FF4D00", "#15FF00", "#00D9FF", "#FF00C8", 'y']
    for var in order:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

        for i, (t, y, label) in enumerate(resultados):
            y_plot = get_var_formatted(var, y)

            if i == 0:
                ax.plot(t, y_plot, lw=1, color='b', label=label)
            else:
                ax.plot(t, y_plot, lw=1, linestyle='--', color=colors[i], label=label)

        ax.set_title(titles[var], fontsize=30, fontweight='bold')
        ax.set_ylabel(latex_labels[var], fontsize=27)
        ax.set_xlabel(r"Tempo [s]", fontsize=27)

        ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=25)
        ax.legend(fontsize=12)

        match perturbacao[0]:
            case '1': nome_superficie = 'aileron'
            case '2': nome_superficie = 'profundor'
            case '3': nome_superficie = 'leme'
            case _: nome_superficie = 'desconhecida'

        match perturbacao[1]:
            case 1: tipo_perturbacao = 'pulso'
            case 2: tipo_perturbacao = 'doublet'
            case _: tipo_perturbacao = 'nenhum'

        if savefile:
            nome = f"{savefile}_{var}_{nome_superficie}_{tipo_perturbacao}.pdf"
            fig.savefig(nome, dpi=300, bbox_inches='tight')
            print("Salvo:", nome)

        #plt.show()

def plotar_variaveis(t, y, variaveis,perturbacao=None):
    """
    Plota individualmente as variáveis de estado escolhidas.
    
    
    Parâmetros:
    -----------
    t : array
        Vetor de tempo da simulação.
    y : array
        Matriz de estados (shape: estados x tempo).
    variaveis : list
        Lista de strings com os nomes das variáveis desejadas.
        Exemplo: ['u', 'theta', 'h_E']
    """
    plt.rcdefaults()
 
    plt.style.use('seaborn-v0_8-ticks') 
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,     
    })
    import seaborn as sns

    sns.set_context("paper", font_scale=1)

    titles = {
        'u'     : r'Velocidade longitudinal $u$',
        'v'     : r'Velocidade lateral $v$',
        'w'     : r'Velocidade vertical $w$',
        'p'     : r'Taxa de rolamento $p$',
        'q'     : r'Taxa de arfagem $q$',
        'r'     : r'Taxa de guinada $r$',
        'phi'   : r'Ângulo de rolamento $\phi$',
        'theta' : r'Ângulo de arfagem $\theta$',
        'psi'   : r'Ângulo de guinada $\psi$',
        'x_E'   : r'Posição inercial $x_E$',
        'y_E'   : r'Posição inercial $y_E$',
        'h_E'   : r'Altitude $h_E$',
        'alpha' : r'Ângulo de ataque $\alpha$',
        'beta'  : r'Ângulo de derrapagem $\beta$',
        'V_inf' : r'Velocidade total $V_\infty$',
    }
    match perturbacao[0]:
        case '1':
            nome_superficie = 'aileron'
        case '2':
            nome_superficie = 'profundor'
        case '3':
            nome_superficie = 'leme'
        case _:
            nome_superficie = 'desconhecida'

    # segunda parte: tipo de perturbação
    match perturbacao[1]:
        case 1:
            tipo_perturbacao = 'pulso'
        case 2:
            tipo_perturbacao = 'doublet'
        case _:
            tipo_perturbacao = 'nenhum'


    latex_labels = {
        'u': r'$u \; [\mathrm{m/s}]$', 'v': r'$v \; [\mathrm{m/s}]$', 'w': r'$w \; [\mathrm{m/s}]$',
        'p': r'$p \; [^\circ/\mathrm{s}]$', 'q': r'$q \; [^\circ/\mathrm{s}]$', 'r': r'$r \; [^\circ/\mathrm{s}]$',
        'phi': r'$\phi \; [\mathrm{rad}]$', 'theta': r'$\theta \; [\mathrm{rad}]$', 'psi': r'$\psi \; [\mathrm{rad}]$',
        'x_E': r'$x_E \; [\mathrm{m}]$', 'y_E': r'$y_E \; [\mathrm{m}]$', 'h_E': r'$h_E \; [\mathrm{m}]$',
        'alpha': r'$\alpha \; [^\circ]$', 'beta': r'$\beta \; [^\circ]$',
        'V_inf': r'$V_\infty \; [\mathrm{m/s}]$',
    }

    order = ['u','v','w','p','q','r','phi','theta','psi',
             'x_E','y_E','h_E','alpha','beta','V_inf']

    def get_var_formatted(var, y):
        u, v, w = y[0,:], y[1,:], y[2,:]
        if var == 'h_E':
            return -y[11,:]
        elif var == 'alpha':
            return rad2deg(arctan2(w, u))
        elif var == 'beta':
            V = sqrt(u**2 + v**2 + w**2)
            return rad2deg(arcsin(v/V))
        elif var == 'V_inf':
            return sqrt(u**2 + v**2 + w**2)
        elif var in ['p','q','r']:
            idx = order.index(var)
            return rad2deg(y[idx,:])
        else:
            return y[order.index(var),:]


    for var in variaveis:
        plt.figure(figsize=(7,5))
        y_plot = get_var_formatted(var, y)
        plt.plot(t, y_plot, lw=2, color='b')
        plt.title(titles[var], fontsize=30, fontweight='bold')
        plt.xlabel("Tempo [s]", fontsize=27)
        plt.ylabel(latex_labels[var], fontsize=27)
        plt.grid(True, alpha=0.5, linestyle='--')
        eix = plt.gca()
    
        eix.spines['top'].set_visible(False)
        eix.spines['right'].set_visible(False)
        eix.tick_params(labelsize=25)
        plt.savefig(var+'_'+nome_superficie+'_'+tipo_perturbacao+'.pdf', dpi=300, bbox_inches='tight')
        #plt.show()


def plotar_deflexoes(t, sim):
    """
    Plota as deflexões das superfícies de controle ao longo do tempo.
    
    Parâmetros:
    -----------
    t : array
        Vetor de tempo da simulação.
    sim : objeto SimuladorDeVoo
        Simulador já configurado com perturbações.
    """
    d01_list, d02_list, d03_list, d04_list = [], [], [], []
    plt.rcdefaults()

    plt.style.use('seaborn-v0_8-ticks') 
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,     # Tamanho base forçado
    })
    import seaborn as sns
    
    sns.set_context("paper", font_scale=1)

    # percorre o tempo e calcula as deflexões
    for ti in t:
        d01_p, d02_p, d03_p, d04_p = sim.perturb_state(ti)
        d01_list.append(sim.aviao.d01_ref + d01_p)
        d02_list.append(sim.aviao.d02_ref + d02_p)
        d03_list.append(sim.aviao.d03_ref + d03_p)
        d04_list.append(sim.aviao.d04_ref + d04_p)



    # plota
    
        # plota apenas as superfícies perturbadas
    if any(val != sim.aviao.d01_ref for val in d01_list):
        plt.figure(figsize=(10,6))
        plt.plot(t, [val*180/pi for val in d01_list], lw=2, color='b')
        plt.ylabel(r"$\delta_a$ [$^\circ$]", fontsize=12)
        plt.title(r"Deflexão do aileron $\delta_a$", fontsize=14, fontweight="bold")

    if any(val != sim.aviao.d02_ref for val in d02_list):
        plt.figure(figsize=(10,6))
        plt.plot(t, [val*180/pi for val in d02_list], lw=2, color='b')
        plt.ylabel(r"$\delta_e$ [$^\circ$]", fontsize=27)
        plt.title(r"Deflexão do profundor $\delta_e$", fontsize=30, fontweight="bold")

    if any(val != sim.aviao.d03_ref for val in d03_list):
        plt.figure(figsize=(7,5))
        plt.plot(t, [val*180/pi for val in d03_list], lw=2, color='b')
        plt.ylabel(r"$\delta_r$ [$^\circ$]", fontsize=27)
        plt.title(r"Deflexão do leme $\delta_r$", fontsize=27, fontweight="bold")

    
    plt.xlabel("Tempo [s]", fontsize=27)
    plt.grid(True, alpha=0.5, linestyle='--')
    eix = plt.gca()
    # Remove bordas superior e direita
    eix.spines['top'].set_visible(False)
    eix.spines['right'].set_visible(False)
    eix.tick_params(labelsize=12)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("deflexoes_superficies_controle.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plotar_trajetoria_3D(t, y, titulo="Trajetória 3D da Aeronave", perturbacao=None):
    """
    Plota a trajetória 3D da aeronave mostrando o ponto inicial e final.
    
    Parâmetros:
    -----------
    t : array
        Vetor de tempo da simulação.
    y : array
        Matriz de estados (cada linha corresponde a um instante).
        Espera-se que as colunas 9, 10 e 11 sejam x_E, y_E, z_E.
    titulo : str
        Título do gráfico.
    """
    plt.rcdefaults()

    plt.style.use('seaborn-v0_8-ticks') 
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,     # Tamanho base forçado
    })
    import seaborn as sns
 
    sns.set_context("paper", font_scale=1)


    x = y[9, :]   # posição x_E
    y_pos = y[10, :]  # posição y_E
    z = -y[11, :]  # altitude 

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Trajetória
    ax.plot(x, y_pos, z, color='b', lw=2, label="Trajetória")

    # Ponto inicial
    ax.scatter(x[0], y_pos[0], z[0], color='g', s=40, marker='s', label="Início")

    # Ponto final
    ax.scatter(x[-1], y_pos[-1], z[-1], color='r', s=40, marker='^', label="Fim")
    ax.view_init(30, -45)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
    # Configurações visuais
    ax.set_title(titulo, fontsize=16, fontweight='bold')
    
    ax.set_xlabel("Norte $x_E$ [m]", fontsize=10)
    ax.set_ylabel("Leste $y_E$[m]",fontsize=10)
    ax.set_zlabel("Altitude $h_E$ [m]", fontsize=10)
    ax.grid(True)
    match perturbacao[0]:
        case '1':
            nome_superficie = 'aileron'
        case '2':
            nome_superficie = 'profundor'
        case '3':
            nome_superficie = 'leme'
        case _:
            nome_superficie = 'desconhecida'

    # segunda parte: tipo de perturbação
    match perturbacao[1]:
        case 1:
            tipo_perturbacao = 'pulso'
        case 2:
            tipo_perturbacao = 'doublet'
        case _:
            tipo_perturbacao = 'nenhum'
    plt.tight_layout()
    plt.savefig("trajetoria_3D_"+nome_superficie+"_"+tipo_perturbacao+".pdf", dpi=300, bbox_inches='tight')

    #plt.show()
class Bandeirante:
    def __init__(self, mass=4600, Ixx=31242.0832, Iyy=18261.2505, Izz=47039.5484, V=118.3, mudar_derivadas = None,
                 alfa_ref = -1.18207 * pi/180,d02_ref =2.78855*pi/180): 
        
        beta_ref = 0
        self.alpha_ref = alfa_ref
        self.beta_ref = beta_ref
        matriz_T_B_W = MatrizTransformacoesVentoCorpo(alfa_ref,beta_ref)

        # Geometria
        self.S = 29.000 
        self.CMA = 1.9400 
        self.b = 15.330  
        self.AR = self.b**2/self.S
        # Massa/Inércia
        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.V = V

        # Derivadas Aerodinâmicas
        
        derivadas = {
                'CLa': 5.766647,
                'CLb': 0,
                'CLp': 0,
                'CLq': 15.734747 ,
                'CLr': 0,
                'CLd01': 0,
                'CLd02': 0.021137*180/pi,
                'CLd03': 0,

                'CYa': 0,
                'CYb': -0.322016,
                'CYp': -0.261167,
                'CYq': 0,
                'CYr': 0.295798,
                'CYd01': -0.001164 * 180/pi,
                'CYd02': 0,
                'CYd03': -0.003713*180/pi,

                'CDa': 0.114003,
                'CDb': 0.000089,
                'CDp': 0.000019,
                'CDq': 0.251085 ,
                'CDr': -0.000088,
                'CDd01': 0.000062*180/pi,
                'CDd02': 0.000767*180/pi,
                'CDd03': 0.000579*180/pi,

                'Cla': 0,
                'Clb': -0.136432,
                'Clp': -0.537696 ,
                'Clq': 0,
                'Clr': 0.112338,
                'Cld01': -0.004167*180/pi, 
                'Cld02': 0,
                'Cld03': -0.000631*180/pi,

                'Cma': -3.103564,
                'Cmb': 0.000137,
                'Cmp': 0.000031,
                'Cmq': -36.605337,
                'Cmr': -0.000136,
                'Cmd01': 0.000014*180/pi,
                'Cmd02': -0.070779*180/pi,
                'Cmd03': 0.000133*180/pi,

                'Cna': 0,
                'Cnb': 0.111166,
                'Cnp': 0.009790 ,
                'Cnq': 0,
                'Cnr': -0.118304,
                'Cnd01':0.000083*180/pi,
                'Cnd02': -0.000009*180/pi,
                'Cnd03': 0.001651*180/pi,

                'CL_ref' : 0.24497,
                'CD_ref' : 0.00961 + 0.37/29 + 1/(pi*0.9934*self.AR)*0.24497**2,#0.01197 + 0.37/29,
                'CY_ref' : 0,
                'Cl_ref' : 0,
                'Cm_ref' : 0,
                'Cn_ref' : 0

            }
        if mudar_derivadas:
            derivadas.update(mudar_derivadas)
        for derivada, valor in derivadas.items():
            setattr(self, derivada, valor)


        # Estado de referência
        v_infty = array([self.V,0,0])
        v_corpo = matriz_T_B_W @ v_infty
        (u,v,w) = v_corpo
        ref_state = {
            'u0': u, 'v0': v, 'w0': w, 'p0': 0, 'q0': 0, 'r0': 0,
            'phi0': 0, 'theta0': alfa_ref, 'psi0': 0,
            'x0': 0, 'y0': 0, 'z0': -3048
        }
        lista = [ref_state[k] for k in ref_state]
        self.ref_state = array(lista)

        # Controles
        self.d01_ref, self.d02_ref, self.d03_ref, self.d04_ref = 0, d02_ref, 0, 1

        # Outros
        self.V_ref = self.V
        self.rho_ref = 0.905
        self.T_ref = self.CD_ref * self.V_ref**2*0.5*self.rho_ref * self.S
        self.p_ref = self.q_ref = self.r_ref = 0


def configurar_perturbacao():
        """Coleta os dados do usuário e armazena em self para usar durante a simulação."""
        print("--- Configuração da Perturbação ---")
        perturbar = input('Deseja adicionar uma perturbação? (s/n): ').lower() == 's'
        if not perturbar:
            pert_superficie = None
            pert_tipo = None
            pert_magnitude = 0.0
            pert_tempo = 0.0
            pert_atraso = 0.0
            return pert_superficie, pert_tipo, pert_magnitude, pert_tempo, pert_atraso
        pert_superficie = input('Selecione a superfície de controle:\n1 - Aileron (d01)\n2 - Profundor (d02)\n3 - Leme (d03)\n4 - Potência (d04) [Não implementado]\n')
        pert_tipo = int(input('Selecione o tipo:\n1 - Pulso\n2 - Doublet\n')) # Converta para int
        pert_magnitude = float(input('Magnitude (graus): '))*pi/180 
        pert_tempo = float(input('Duração do pulso (s): '))
        atraso_input = input('Atraso (s) [Enter para 0s]: ')
        pert_atraso = float(atraso_input) if atraso_input else 0.0
        return pert_superficie, pert_tipo, pert_magnitude, pert_tempo, pert_atraso

class SimuladorDeVoo:
    def __init__(self, aviao,perturbacao):
        self.aviao = aviao
        self.g = 9.78
        self.rho = 0.905

        perturbacao_global = perturbacao
        self.pert_superficie = perturbacao_global[0]
        self.pert_tipo = perturbacao_global[1]
        self.pert_magnitude = perturbacao_global[2]
        self.pert_tempo = perturbacao_global[3]
        self.pert_atraso = perturbacao_global[4]
    
    def perturb_state(self, t):
        # Inicializa perturbações como 0
        d01_p, d02_p, d03_p, d04_p = 0.0, 0.0, 0.0, 0.0
        
        # Verifica se estamos dentro da janela de tempo do atraso
        # Se t for menor que o atraso, não faz nada
        if t < self.pert_atraso:
            return 0.0, 0.0, 0.0, 0.0

        # Lógica do Pulso
        if self.pert_tipo == 1:
            if self.pert_atraso < t <= (self.pert_atraso + self.pert_tempo):
                val = self.pert_magnitude
                if self.pert_superficie == '1': d01_p = val
                elif self.pert_superficie == '2': d02_p = val
                elif self.pert_superficie == '3': d03_p = val
                elif self.pert_superficie == '4': d04_p = val
        # Lógica do Doublet
        elif self.pert_tipo == 2:
            t_mid = self.pert_atraso + self.pert_tempo * 0.5
            t_end = self.pert_atraso + self.pert_tempo

            if self.pert_atraso < t <= t_mid:
                val = self.pert_magnitude
                if self.pert_superficie == '1': d01_p = val
                elif self.pert_superficie == '2': d02_p = val
                elif self.pert_superficie == '3': d03_p = val
                elif self.pert_superficie == '4': d04_p = val
            elif t_mid < t <= t_end:
                val = -self.pert_magnitude
                if self.pert_superficie == '1': d01_p = val
                elif self.pert_superficie == '2': d02_p = val
                elif self.pert_superficie == '3': d03_p = val
                elif self.pert_superficie == '4': d04_p = val
        return d01_p, d02_p, d03_p, d04_p

        
    def EquacoesDeMovimento(self,t,y):

        u, v, w, p, q, r, phi, theta, psi, x_E, y_E, z_E = y


        d01_p, d02_p, d03_p, d04_p = self.perturb_state(t)
        d01_p, d02_p, d03_p, d04_p = self.perturb_state(t)
        d01 = self.aviao.d01_ref + d01_p
        d02 = self.aviao.d02_ref + d02_p
        d03 = self.aviao.d03_ref + d03_p
        d04 = self.aviao.d04_ref + d04_p


        alpha = arctan2(w, max(u, 1e-6))
            

        V = sqrt(u**2 + v**2 + w**2)

        beta = arcsin(v / V) if V != 0 else 0.0


        q_dyn = 0.5 * self.rho * V**2
        aviao = self.aviao
        # Matrizes de transformacao
        
        cos_alpha = cos(alpha)
        sin_alpha = sin(alpha)
        cos_beta = cos(beta)
        tan_beta = tan(beta)

        matriz_T_B_W_CERTO=  array([[cos_alpha/cos_beta,-cos_alpha*tan_beta,-sin_alpha],
                             [0,1,0],
                             [sin_alpha/cos_beta,-sin_alpha*tan_beta,cos_alpha]])

      

        
        # Calcula os adimentsionais: *self.b/(2*self.V) 
        
        p_ad = p*self.aviao.b/(2*V) 
        q_ad = q*self.aviao.CMA/(2*V) 
        r_ad = r*self.aviao.b/(2*V) 
        p_ref_ad = aviao.p_ref*self.aviao.b/(2*self.aviao.V_ref) 
        q_ref_ad = aviao.q_ref*self.aviao.CMA/(2*self.aviao.V_ref) 
        r_ref_ad = aviao.r_ref*self.aviao.b/(2*self.aviao.V_ref)


        # Calcula os coeficientes de forças no sistema do vento
        CL = (aviao.CL_ref + aviao.CLa * (alpha - aviao.alpha_ref)
              + aviao.CLq * (q_ad - q_ref_ad)
              + aviao.CLd02 * (d02 - aviao.d02_ref))
        CY = (aviao.CY_ref + aviao.CYa * (alpha - aviao.alpha_ref)
              + aviao.CYb * (beta - aviao.beta_ref)
              + aviao.CYp * (p_ad - p_ref_ad)
              + aviao.CYr * (r_ad - r_ref_ad)
              + aviao.CYd01 * (d01 - aviao.d01_ref)
              + aviao.CYd03 * (d03 - aviao.d03_ref))
        CD = (0.00961 + 0.37/29+ 1/(pi* 0.9934*self.aviao.AR)*CL**2) #+ aviao.CDa * (alpha - aviao.alpha_ref) + aviao.CDd02 * (d02 - aviao.d02_ref))
              

        Cl = (aviao.Cl_ref + aviao.Clb * (beta - aviao.beta_ref)
              + aviao.Clp * (p_ad - p_ref_ad) + aviao.Clr * (r_ad - r_ref_ad)
              + aviao.Cld01 * (d01 - aviao.d01_ref)
              + aviao.Cld03 * (d03 - aviao.d03_ref))
        Cm = (aviao.Cm_ref + aviao.Cma * (alpha - aviao.alpha_ref)
              + aviao.Cmq * (q_ad - q_ref_ad)
              + aviao.Cmd02 * (d02 - aviao.d02_ref))
        Cn = (aviao.Cn_ref + aviao.Cnb * (beta - aviao.beta_ref)
              + aviao.Cnp * (p_ad - p_ref_ad) + aviao.Cnr * (r_ad - r_ref_ad)
              + aviao.Cnd01 * (d01 - aviao.d01_ref)
              + aviao.Cnd03 * (d03 - aviao.d03_ref))


        forcas_corpo = matriz_T_B_W_CERTO @ array([-CD,
                                             CY,
                                             -CL])
        
        momentos_corpo = array([Cl, Cm,Cn])
        X = q_dyn * self.aviao.S * forcas_corpo[0]
        
        #print(f'Velocidade = {self.V} m/s at time t={t} s')
        Y = q_dyn * self.aviao.S * forcas_corpo[1]
        Z = q_dyn * self.aviao.S * forcas_corpo[2]
        L = q_dyn * self.aviao.S * self.aviao.b * momentos_corpo[0] 
        M = q_dyn * self.aviao.S * self.aviao.CMA * momentos_corpo[1]
        N = q_dyn * self.aviao.S * self.aviao.b * momentos_corpo[2]

            
       
        
        T = self.aviao.T_ref 
       
  
        # Equações de movimento

        u_dot = 1 / self.aviao.mass * (X + T) - self.g * sin(theta) + r * v - q * w 
        v_dot = 1 / self.aviao.mass * (Y) + self.g * sin(phi) * cos(theta) + p * w - r * u
        w_dot = 1 / self.aviao.mass * (Z) + self.g * cos(phi) * cos(theta) + q * u - p * v 
        if t == 0:
            print(u_dot,v_dot,w_dot)
        p_dot = 1 / self.aviao.Ixx * (L - (self.aviao.Izz - self.aviao.Iyy) * q * r)
        q_dot = 1 / self.aviao.Iyy * (M - (self.aviao.Ixx - self.aviao.Izz) * p * r)
        r_dot = 1 / self.aviao.Izz * (N - (self.aviao.Iyy - self.aviao.Ixx) * p * q)

        phi_dot = p + (q * sin(phi) + r * cos(phi)) * tan(theta)
        theta_dot = q * cos(phi) - r * sin(phi)
        psi_dot = (q * sin(phi) + r * cos(phi)) / cos(theta)

        if theta == pi/2-1e6 or theta == -pi/2+1e6:
            print(f'CUIDADO: GIMBAL LOCK')

        sin_theta = sin(theta)
        cos_theta = cos(theta)
        sin_psi = sin(psi)
        cos_psi = cos(psi)
        sin_phi = sin(phi)
        cos_phi = cos(phi)
        


   
        R = array([[cos_theta * cos_psi,
                sin_phi * sin_theta * cos_psi - cos_phi * sin_psi,
                cos_phi * sin_theta * cos_psi + sin_phi * sin_psi],
               [cos_theta * sin_psi,
                sin_phi * sin_theta * sin_psi + cos_phi * cos_psi,
                cos_phi * sin_theta * sin_psi - sin_phi * cos_psi],
               [-sin_theta,
                sin_phi * cos_theta,
                cos_phi * cos_theta]])

        x_dot_E, y_dot_E, z_dot = R @ array([u, v, w])


        equacoes = array([u_dot, v_dot, w_dot, p_dot, q_dot, r_dot,
                  phi_dot, theta_dot, psi_dot, x_dot_E, y_dot_E, z_dot])

        return equacoes


    def run_simulation(self, t_fim=100.0):
        # Runge Kutta de 4 ordem


        sol = solve_ivp(fun=self.EquacoesDeMovimento, t_span=(0.0, float(t_fim)), y0=self.aviao.ref_state, method='RK45',
                        rtol=1e-6,      
                        atol=1e-9,events=[stall,stall_angle,hit_ground],
                        max_step=t_fim/1000)
        if sol.status ==1:
            if sol.t_events[0] >0:
                print(f'Simulação interrompida: Estol por velocidade')
            elif sol.t_events[1] >0:
                print(f'Simulação interrompida: Estol por ângulo de ataque')
            elif sol.t_events[2] >0:
                print(f'Simulação interrompida: O avião atingiu o chão')
        else:
            print(f'Simulação concluída com sucesso!')
        return sol.t, sol.y



derivadas_4054 = {# Forças
                            'CLa': 5.767235,
                            'CLb': 0.000000,
                            'CYa': 0.000000,
                            'CYb': -0.322645,
                            'CDa': 0.100380 ,
                            'CDb': 0.000090,

                            # Momentos
                            'Cla': 0.000000,
                            'Clb': -0.137166,
                            'Cma': -3.091151,
                            'Cmb': 0.000139,
                            'Cna': -0.000000,
                            'Cnb':  0.111500,

                            # Derivadas em relação às taxas
                            'CLp': 0.000000,
                            'CLq': 15.737385,
                            'CLr': 0.000000,
                            'CYp': -0.263705,
                            'CYq': 0.000000,
                            'CYr': 0.293222,
                            'CDp': 0.000020,
                            'CDq': 0.216313,
                            'CDr': -0.000088,
                            'Clp': -0.538390,
                            'Clq': 0.000000,
                            'Clr': 0.106341,
                            'Cmp': 0.000032,
                            'Cmq': -36.538411,
                            'Cmr': -0.000138,
                            'Cnp': 0.012280 ,
                            'Cnq': -0.000000,
                            'Cnr': -0.117818,

                            # Derivadas em relação às superfícies de controle
                            'CLd01': -0.000000*180/pi,
                            'CLd02': 0.021133*180/pi,
                            'CLd03': -0.000000*180/pi,
                            'CYd01': -0.001166*180/pi,
                            'CYd02': 0.000000* 180/pi,
                            'CYd03': -0.003731*180/pi,
                            'CDd01': 0.000049*180/pi,
                            'CDd02': 0.000635*180/pi,
                            'CDd03': 0.000455*180/pi,
                            'Cld01': -0.004168*180/pi,
                            'Cld02': 0.000000*180/pi,
                            'Cld03': -0.000644*180/pi,
                            'Cmd01': 0.000011*180/pi,
                            'Cmd02': -0.070718*180/pi,
                            'Cmd03': 0.000104*180/pi,
                            'Cnd01': 0.000067*180/pi,
                            'Cnd02': -0.000007*180/pi,
                            'Cnd03': 0.001658*180/pi,

                            # Coeficientes de referência (do caso trimado)
                            'CL_ref': 0.21589,
                            'CD_ref': 0.01141+0.37/29,
                            'CY_ref': 0.00000,
                            'Cl_ref': -0.00000,
                            'Cm_ref': 0.00000,
                            'Cn_ref': -0.00000
    }
derivadas_5000 = {
                                'CLa': 5.765984,
                                'CLb': 0.000000,
                                'CYa': 0.000000,
                                'CYb': -0.321549,
                                'CDa': 0.123989,
                                'CDb': 0.000089,

                               
                                'Cla': 0.000000,
                                'Clb': -0.135896,
                                'Cma': -3.112659 ,
                                'Cmb': 0.000136,
                                'Cna': -0.000000,
                                'Cnb': 0.110916,

                               
                                'CLp': 0.000000,
                                'CLq': 15.732585,
                                'CLr': 0.000000,
                                'CYp': -0.259287,
                                'CYq': 0.000000,
                                'CYr': 0.297667,
                                'CDp': 0.000019,
                                'CDq': 0.276544,
                                'CDr': -0.000088,
                                'Clp': -0.537168,
                                'Clq': 0.000000,
                                'Clr': 0.116727,
                                'Cmp': 0.000030,
                                'Cmq': -36.653678,
                                'Cmr': -0.000135,
                                'Cnp': 0.007961,
                                'Cnq': -0.000000,
                                'Cnr': -0.118666,

                              
                                'CLd01': -0.000000*180/pi,
                                'CLd02': 0.021138*180/pi,
                                'CLd03': -0.000000*180/pi,
                                'CYd01': -0.001163*180/pi,
                                'CYd02': 0.000000*180/pi,
                                'CYd03': -0.003700*180/pi,
                                'CDd01': 0.000073*180/pi,
                                'CDd02': 0.000871*180/pi,
                                'CDd03': 0.000678*180/pi,
                                'Cld01': -0.004167*180/pi,
                                'Cld02': 0.000000*180/pi,
                                'Cld03': -0.000621*180/pi,
                                'Cmd01': 0.000016*180/pi,
                                'Cmd02': -0.070819*180/pi,
                                'Cmd03': 0.000156*180/pi,
                                'Cnd01': 0.000094*180/pi,
                                'Cnd02': -0.000010*180/pi,
                                'Cnd03': 0.001646*180/pi,

                                
                                'CL_ref': 0.26627,
                                'CD_ref': 0.01242+0.37/29,
                                'CY_ref': 0.00000,
                                'Cl_ref': 0.00000,
                                'Cm_ref': 0.00000,
                                'Cn_ref': 0.00000}
    
derivadas_4600_15_percent = {
        # --- Forças ---
        'CLa': 5.766271,
        'CLb': 0.000000,
        'CYa': 0.000000,
        'CYb': -0.321472,
        'CDa': 0.112825,
        'CDb': 0.000089,

        # --- Momentos ---
        'Cla': 0.000000,
        'Clb': -0.136263,
        'Cma': -3.669748,
        'Cmb': 0.000137,
        'Cna': -0.000000,
        'Cnb': 0.114881,

        # --- Derivadas nas taxas ---
        'CLp': 0.000000,
        'CLq': 16.883834 ,
        'CLr': 0.000000,
        'CYp': -0.260665,
        'CYq': 0.000000,
        'CYr': 0.304052,
        'CDp': 0.000019,
        'CDq': 0.254746,
        'CDr': -0.000090,
        'Clp': -0.537639,
        'Clq': 0.000000,
        'Clr': 0.116765,
        'Cmp': 0.000031,
        'Cmq': -38.934726,
        'Cmr': -0.000139,
        'Cnp': 0.012542,
        'Cnq': -0.000000,
        'Cnr': -0.124779,

        # --- Superfícies de controle (multiplicadas por 180/pi) ---
        'CLd01': -0.000000 * 180/pi,
        'CLd02': 0.021136 * 180/pi,
        'CLd03': -0.000000 * 180/pi,
        'CYd01': -0.001164 * 180/pi,
        'CYd02': 0.000000 * 180/pi,
        'CYd03': -0.003702 * 180/pi,
        'CDd01': 0.000067 * 180/pi,
        'CDd02': 0.000766 * 180/pi,
        'CDd03': 0.000628 * 180/pi,
        'Cld01': -0.004168 * 180/pi,
        'Cld02': 0.000000 * 180/pi,
        'Cld03': -0.000628 * 180/pi,
        'Cmd01': 0.000015 * 180/pi,
        'Cmd02': -0.072889 * 180/pi,
        'Cmd03': 0.000144 * 180/pi,
        'Cnd01': 0.000101 * 180/pi,
        'Cnd02': -0.000010 * 180/pi,
        'Cnd03': 0.001692 * 180/pi,

        # --- Referência do caso trimado ---
        'CL_ref': 0.24497,
        'CD_ref': 0.01200+0.37/29,
        'CY_ref': 0.00000,
        'Cl_ref': -0.00000,
        'Cm_ref': 0.00000,
        'Cn_ref': -0.00000
    }
derivadas_4600_34_percent = {
        # --- Forças ---
        'CLa': 5.766987,
        'CLb': 0.000000,
        'CYa': 0.000000,
        'CYb': -0.322504,
        'CDa': 0.115059,
        'CDb': 0.000090,

        # --- Momentos ---
        'Cla': 0.000000,
        'Clb': -0.136574,
        'Cma': -2.597041 ,
        'Cmb': 0.000138,
        'Cna': -0.000000,
        'Cnb': 0.107831,

        # --- Derivadas em relação às taxas ---
        'CLp': 0.000000,
        'CLq': 14.706715,
        'CLr': 0.000000,
        'CYp': -0.261595 ,
        'CYq': 0.000000,
        'CYr': 0.288389,
        'CDp': 0.000019,
        'CDq': 0.247393,
        'CDr': -0.000086,
        'Clp': -0.537729 ,
        'Clq': 0.000000,
        'Clr': 0.108360,
        'Cmp': 0.000031,
        'Cmq': -34.712342,
        'Cmr': -0.000133,
        'Cnp': 0.007310,
        'Cnq': -0.000000,
        'Cnr': -0.112674,

        # --- Superfícies de controle (corrigidas com 180/pi) ---
        'CLd01': -0.000000 * 180/pi,
        'CLd02': 0.021137 * 180/pi,
        'CLd03': -0.000000 * 180/pi,
        'CYd01': -0.001164 * 180/pi,
        'CYd02': 0.000000 * 180/pi,
        'CYd03': -0.003723 * 180/pi,
        'CDd01': 0.000058 * 180/pi,
        'CDd02': 0.000772 * 180/pi,
        'CDd03': 0.000540 * 180/pi,
        'Cld01': -0.004167 * 180/pi,
        'Cld02': 0.000000 * 180/pi,
        'Cld03': -0.000634 * 180/pi,
        'Cmd01': 0.000013 * 180/pi,
        'Cmd02': -0.068887 * 180/pi,
        'Cmd03': 0.000125 * 180/pi,
        'Cnd01': 0.000066 * 180/pi,
        'Cnd02': -0.000008 * 180/pi,
        'Cnd03': 0.001615 * 180/pi,

        # --- Coeficientes de referência (caso trimado) ---
        'CL_ref': 0.24497,
        'CD_ref': 0.01195+0.37/29,
        'CY_ref': 0.00000,
        'Cl_ref': 0.00000,
        'Cm_ref': 0.00000,
        'Cn_ref': 0.00000
    }






if __name__ == "__main__":
    perturbar=configurar_perturbacao()
    # Avião padrão
    missao2 = Bandeirante()

    # Analise dinamica:
    # traj3d = SimuladorDeVoo(missao2, perturbacao = perturbar)
    # t, y = traj3d.run_simulation(t_fim=400)
    # plotar_trajetoria_3D(t, y, titulo='Trajetória 3D da Aeronave', perturbacao=perturbar)
    # plotar_variaveis(t, y, ['x_E','y_E','h_E'],
    #                   perturbacao=perturbar)
    #plotar_deflexoes(t,traj3d)

    # Variacao de cg
    missao2_cg_dianteiro = Bandeirante(mudar_derivadas=derivadas_4600_15_percent,alfa_ref=-1.09752 *pi/180,d02_ref=2.38593*pi/180)
    missao2_cg_traseiro = Bandeirante(mudar_derivadas=derivadas_4600_34_percent,alfa_ref=-1.25777*pi/180,d02_ref=3.14901*pi/180)

    # Variacao de massa
    missao1 = Bandeirante(mudar_derivadas=derivadas_4054,mass= 4054, Ixx =27533.7837,Iyy=16093.7195,Izz=41456.1586,alfa_ref=-1.52608*pi/180,d02_ref=3.05110*pi/180)
    missao3 = Bandeirante(mudar_derivadas=derivadas_5000,mass= 5000, Ixx =33958.7861,Iyy=19849.1854,Izz=51129.9440,alfa_ref=-0.92989*pi/180,d02_ref=2.59560*pi/180)


    #Rodar comparacoes de massa e cg
    comparar_simulacoes([missao2,missao2_cg_traseiro,missao3,missao1,missao2_cg_dianteiro],
                       ['Missão 2 - CG Padrão','Missão 2 - CG Traseiro','Missão 3 - Massa 5000 kg','Missão 1 - Massa 4054 kg','Missão 2 - CG Dianteiro'],
                       t_fim=125.0, savefile='comparacao_',perturbacao = perturbar)
    # comparar_simulacoes([missao2, missao1,missao3], ['Missão 2 - Massa 4600 kg','Missão 1 - Massa 4054 kg','Missão 3 - Massa 5000 kg'],
    #                     t_fim=200.0, savefile='comparacao_massa')
    
    




import numpy as np
import matplotlib.pyplot as plt

def fit(X,Y,Order):
    assert(np.shape(X) == np.shape(Y))
    assert(np.shape(X)[0] > Order)
    New_Mat = np.zeros([Order + 1, np.shape(X)[0]])
    for x in reversed(range(Order + 1)):
        for y in range(np.shape(X)[0]):
            New_Mat[x,y] = np.power(X[y],x)
    Coeffs = np.linalg.lstsq(np.transpose(New_Mat),Y, rcond=None)[0]   
    return(Coeffs)

def calc_values(Coeffs, Order, Min, Max):
    X = np.arange(Min, Max, 0.001)
    Y = np.zeros([np.shape(X)[0]])
    tmp = []
    for x in range(Order+1):
        tmp.append(np.power(X, x))
    New_Mat = np.stack(tmp, axis = 0)
    Y = np.dot(np.transpose(New_Mat), Coeffs)
    Lines = np.stack((X,Y),axis = 0)
    return(Lines)
    
def plot_fit(Data, X, Sparse, Order):
    tmp = len(Data[0,:]) / Sparse
    MinVecSparse = np.zeros([2, int(np.ceil(tmp))])
    MinVec = np.zeros(len(Data[0,:]))
    VolVecSparse = np.zeros(int(np.ceil(tmp)))
    VolVec = np.zeros(len(Data[0,:]))
    tmp = 0
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))
    plt.subplots_adjust(wspace = 0.5)
    fig.suptitle('QHA for P2$_1$/c ReN$_2$    [p = 100 GPa]')
    axs[0].set(xlabel = 'Volume [Å$^3$]', ylabel = 'Gibbs Free Energy [eV/Unitcell]', xticks = np.arange(84, 100, 4))
    axs[1].set(xlabel = 'Temperature [K]', ylabel = 'Volume [Å$^3$]', xticks = np.arange(0, 1001, 250))
    axs[2].set(xlabel = 'Temperature [K]', ylabel = 'Energy [eV/Unitcell]', xticks = np.arange(0, 1001, 250))
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    for x in range(len(Data[0,:])):
        Coeffs = fit(X,Data[:,x],Order)
        line = calc_values(Coeffs, Order, X[0], X[-1])
        minpos = np.argmin(line[1,:])
        VolVec[x] = line[0,minpos]
        MinVec[x] = np.min(line[1,:])
        if (x % Sparse) == 0:
            MinVecSparse[0,tmp] = line[0,minpos]
            MinVecSparse[1,tmp] = MinVec[x]
            VolVecSparse[tmp] = VolVec[x]
            axs[0].plot(line[0,:], line[1,:], color = "grey")
            axs[0].plot(X, Data[:,x], 'o', color = "blue")
            tmp += 1
    axs[0].plot(MinVecSparse[0,:], MinVecSparse[1,:], '-o', color = "red")
    TempSparse = np.arange(0,1001,Sparse*10)
    Temp = np.arange(0,1001,10)
    axs[2].plot(TempSparse, MinVecSparse[1,:], 'o', color = "red")
    axs[2].plot(Temp, MinVec, color = "red")
    axs[1].plot(TempSparse, VolVecSparse, 'o', color = "red")
    axs[1].plot(Temp, VolVec, color = "red")
    return(Temp, MinVec)
    
    
def read_file(file_name):
    f = open(file_name, "r")
    Data = f.read()
    f.close()
    lines = Data.split("\n")
    return(lines)

def read_ev(file_name):
    lines = read_file(file_name)
    del(lines[0])
    y = len(lines)
    Arr = np.empty([2, y])
    for x in range(y):
        tmp = lines[x].split()
        Arr[0,x] = float(tmp[0])
        Arr[1,x] = float(tmp[-1])      
    return(Arr)

def read_tp(file_name, MaxT):
    lines = read_file(file_name)
    for x in range(16):
        del(lines[0])
    Arr = np.empty([5, int(np.ceil(MaxT / 10) + 1)])
    for x in range(101):
        for y in range(5):
            tmp = lines[x * 6 + y].split()
            Arr[y][x] = tmp[-1]
    return(Arr)
    
def write(Temp, GFE):
    f = open('P21c_ReN2.txt','w')
    f.write('#T            G\n')
    for x in range(len(GFE)):
        f.write(str(Temp[x]) + ' ' + str(GFE[x]) + '\n')
    f.close()

def Gibbs(U, Vol, F):
    p = 100 / 160.2176621     #100 GPa to eV/Å^3
    conv = 6.2415064799632e+21 /  6.02214076e+23      #kj/mol to eV/UC
    GFE = U + Vol*p + F*conv
    return(GFE)

def QHA(EV_name, TP_name, TP_number, Order, MaxT):
    EV_Data = read_ev(EV_name)   
    ls = []
    for x in range(TP_number):
        ls.append(read_tp(TP_name + "-" + str(x + 1), MaxT))
    TP_Data = np.stack((ls), axis = 2)
    del(ls)
    Data = np.empty([TP_number, int(np.ceil(MaxT / 10) + 1)])
    for x in range(TP_number):
        for y in range(int(np.ceil(MaxT / 10) + 1)):
            Data[x][y] = Gibbs(EV_Data[0][x], EV_Data[1][x], TP_Data[1][y][x])        
    Temp, MinVec = plot_fit(Data,EV_Data[1,:],10, Order)
    write(Temp, MinVec)


QHA("e-v.dat", "thermal_properties.yaml", 5, 4, 1000)

import numpy as np
import matplotlib.pyplot as plt

def read_file(file_name):
    f = open(file_name, "r")
    Data = f.read()
    f.close()
    lines = Data.split("\n")
    return(lines)
    
    
def read_usage(num_gpus, lines):
    arr_ls = []
    t = 0
    arr = np.zeros([num_gpus])
    for line in lines:
        tmp = line.split()
        if len(tmp) > 1:
            if tmp[1] == "N/A":
                data = tmp[8].split('M')
                arr[t] = data[0]
                if t == (num_gpus - 1):
                    t = 0
                    arr_ls.append(arr)
                    arr = np.zeros([num_gpus])
                else:
                    t += 1
    Data = np.stack(arr_ls, axis = 0)
    return(Data)
    

def read_name(line):
    tmp = line.split()
    name = tmp[2] + ' ' + tmp[3]
    if '-PCIE...' in name:
        tmp = name.split('-')
        name = tmp[0]
    return(name)

def read_hardware(lines):
    num_gpus = 0
    GPUs = []
    prevLine = ''
    for line in lines:
        tmp = line.split()
        if len(tmp) > 1:
            if tmp[1] == "N/A":
                tmp = tmp[10].split('M')
                max_mem = tmp[0]
                num_gpus += 1
                GPUs.append(read_name(prevLine))
        prevLine = line
        if not line.strip():
            print('Found ' + str(num_gpus) + ' GPUs:')
            for x in range(num_gpus):
                print('GPU #' + str(x+1) + ': ' + GPUs[x])
            break
    return(GPUs, num_gpus, max_mem)
    
def plot_usage(Data, num_gpus, max_mem, GPUs):
    X = range(0, len(Data) * 5, 5)
    y_max = np.empty([len(X)])
    for x in range(len(X)):
        y_max[x] = max_mem
    fig, axs = plt.subplots()
    for x in range(num_gpus):
        axs.plot(X, Data[:,x])
    axs.plot(X,y_max, '-r')
    axs.set(xlabel = "Time [s]", ylabel = "GPU-Usage [MByte]", ylim = (0, int(float(max_mem) * 1.05)))
    GPUs.append("Maximum RAM / GPU")
    axs.legend(GPUs)
    
lines = read_file('gpu_info-9737747.txt')
#lines = read_file('gpu_info-9560137.txt')
#Change this text to match your GPU_info-file

GPUs, num_gpus, max_mem = read_hardware(lines)

Data = read_usage(num_gpus,lines)

plot_usage(Data, num_gpus, max_mem, GPUs)

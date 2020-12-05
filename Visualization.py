import matplotlib.pyplot as plt

def get_train_procedure():
    f = open("train_procedure")
    line = f.readline()
    step = []
    accurance = []
    while line:
        data = line.split(' ')
        step.append(int(data[8]))
        accurance.append(float(data[11][:-1]))
        line = f.readline()
    return step, accurance


if __name__ == '__main__':
    step, accurance = get_train_procedure()
    # for i in range(len(step)):
    #     print(step[i],accurance[i])
    plt.plot(step,accurance,color='blue',linewidth=0.5)
    plt.title('Trainning Procedure')
    plt.xlabel('Step')
    plt.ylabel('Accurance')
    plt.rcParams['figure.figsize'] = (16.0, 4.0)
    plt.show()
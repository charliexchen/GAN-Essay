import time
import math
import pickle


class Timer():
    # This object tracks the timing of the GAN
    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def elapsed(self,l_time):
        output = ""
        if l_time >= 3600:
            hours = int(math.floor(l_time / (60 * 60)))
            l_time -= hours*60*60
            output += "{} hr ".format(hours)
        if l_time >=60:
            minuites = int(math.floor(l_time / (60)))
            l_time -= minuites*60
            output += "{} min ".format(minuites)
        return output+ "{0:.2f} sec ".format(l_time)

    def elapsed_time(self):
        print("Time Elapsed: {}".format(self.elapsed(time.time() - self.start_time)))


class DataLogger():
    # This object logs the performance of the GAN
    def __init__(self):
        self.D_acc=[]
        self.G_acc=[]
        self.D_loss=[]
        self.G_loss=[]

    def log(self, data):
        self.D_acc.append(data[0])
        self.G_acc.append(data[1])
        self.D_loss.append(data[2])
        self.G_loss.append(data[3])

    def save(self, filename=False):
        if not filename:
            filename = "MNIST-Logdata-{}-Epochs".format(len(self.D_acc))
        savedfile = open(filename,'wb')
        data=[self.D_acc,self.G_acc,self.D_loss,self.G_loss]
        pickle.dump(data,savedfile)
        savedfile.close()
        print("Saved log files to {}".format(filename))

    def load(self, filename):
        loadfile = open(filename,'rb')
        data = pickle.load(loadfile)
        self.D_acc=data[0]
        self.G_acc=data[1]
        self.D_loss=data[2]
        self.G_loss=data[3]
        print("Loaded log files from {}".format(filename))
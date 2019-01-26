from matplotlib import pyplot, dates
from csv import reader
from dateutil import parser
import pdb

with open('logs/epoch_loss_val_2.txt', 'r') as f:
    data = list(reader(f))

epoch_loss = [float(i[1]) for i in data]
epoch = [float(i[0]) for i in data]

pyplot.plot(epoch, epoch_loss)
pyplot.title('Epoch Loss Val')
pyplot.xlabel('Number of epochs')
pyplot.ylabel('Epoch loss')
pyplot.show()
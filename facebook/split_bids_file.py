import numpy

train = numpy.genfromtxt('train.csv', delimiter=',', skiprows=1, usecols=[0], dtype=str)
test = numpy.genfromtxt('test.csv', delimiter=',', skiprows=1, usecols=[0], dtype=str)

with open('bids.csv', 'r') as f_bids, \
     open('train_bids.csv', 'w') as f_train_bids, \
     open('test_bids.csv', 'w') as f_test_bids:
    for row in f_bids:
        bidder_id = row.split(',')[1]
        if bidder_id in train:
            f_train_bids.write(row)
        elif bidder_id in test:
            f_test_bids.write(row)
        else:
            print("Unfound id %s" % bidder_id)

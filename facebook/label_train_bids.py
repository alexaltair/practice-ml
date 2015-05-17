import numpy

robot_bidders = numpy.genfromtxt('robot_bidders.csv', dtype=str)

with open('train_bids.csv', 'r') as f_train_bids, \
     open('train_bids2.csv', 'w') as f_train_bids2:
    for row in f_train_bids:
        bidder_id = row.split(',')[1]
        if bidder_id in robot_bidders:
            f_train_bids2.write(''.join([row.rstrip(), ',1.0\n']))
        else:
            f_train_bids2.write(''.join([row.rstrip(), ',0.0\n']))

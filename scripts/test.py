import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', action='store', dest='i', default='i.log')
parser.add_argument('-o', action='store', dest='o', default='o.log')
args, unknown = parser.parse_known_args()

adict = dict(zip(unknown[:-1:2],unknown[1::2]))

print args.i
print args.o
print args
print unknown
print adict
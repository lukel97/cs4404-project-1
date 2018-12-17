import sys
filename = sys.stdin.readline().strip()
f = open(filename, 'r+')
f2 = open(filename[:-4]+'_2.csv', 'a+')
f.readline()

lines = []
losses = []
for line in f:
    line = line.split(',')
    lines.append([line[0], float(line[1])])
    losses.append(float(line[1]))

max_loss = max(losses)
f2.write('Epoch,Value\n')
for line in lines:
    f2.write(line[0]+','+ str(line[1]/max_loss)+ '\n')

f.close()
f2.close()

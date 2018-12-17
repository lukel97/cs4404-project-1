import sys
filename = sys.stdin.readline().strip()
f = open(filename, 'r+')
f2 = open(filename+'_2.csv', 'a+')
f.readline()

lines = []
losses = []
for line in f:
    line = line.split(',')
    lines.append([line[0], line[1], float(line[2])])
    losses.append(float(line[2]))

max_loss = max(losses)
f2.write('Epoch,Step,Value\n')
for line in lines:
    f2.write(line[0]+','+ line[1] +','+ str(line[2]/max_loss)+ '\n')

f.close()
f2.close()

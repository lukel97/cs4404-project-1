import sys
import re

lines = [line for line  in open(sys.stdin.readline().strip())]
parens_matcher = r'\((.*?)\)'
translate = { 'D_A' : 'W2SDiscLoss',
                'G_A':'W2SGenLoss',
                'cycle_A':'CycleConsistencySLoss',
                'idt_A':'IdtS',
                'D_B':'S2WDiscLoss',
                'G_B':'S2WGenLoss',
                'cycle_B':'CycleConsistencyWLoss',
                'idt_B': 'IdtW'}
files = {}
column_names = 'Epoch,Step,Value\n'
for key in translate:
    f = open("pyTorch" + translate[key] + ".csv", "a+")
    f.write(column_names)
    files[key] = f

print_every = 20
for i, line in enumerate(lines):
    if not (i % print_every):
        print(str(i) +' lines processed'.format(i))
    line_match = re.findall(parens_matcher, line).pop()
    matches = [s for s in line_match.split(' ') if s]
    
    epoch, iteration = matches[1].strip()[:-1], matches[3][:-1] 
    exclude_parens_match = '(' + line_match+')'
    matches_losses = line[len(exclude_parens_match):].strip().split(' ')
    parsed_losses = {}
    i = 1
    while i < len(matches_losses):
        parsed_losses[matches_losses[i - 1][:-1]] =  (matches_losses[i].strip())
        i += 2

    for key in translate.keys():
        f = files[key]
        f.write(epoch+','+iteration+','+parsed_losses[key]+'\n')


for key in translate.keys():
    files[key].close()

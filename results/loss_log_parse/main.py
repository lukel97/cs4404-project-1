import sys
import re

lines = [line for line  in open(sys.stdin.readline().strip())]
parens_matcher = r'\((.*?)\)'
translate = { 'D_A' : 'W2SDiscLoss',
                'G_A':'W2SGenLoss',
                'idt_A':'IdtS',
                'D_B':'S2WDiscLoss',
                'G_B':'S2WGenLoss',
                'idt_B': 'IdtW',  
                'CCL':'CycleConsistencyLoss'}
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
    
    epoch, iteration = matches[1][:-1], matches[3][:-1]

    exclude_parens_match = '(' + line_match+')'
    matches_losses = line[len(exclude_parens_match):].strip().split(' ')
    parsed_losses = {}
    i = 1
    while i < len(matches_losses):
        parsed_losses[matches_losses[i - 1][:-1]] =  float((matches_losses[i].strip()))
        i += 2
    
    # average the cycle consistency losses
    parsed_losses['CCL'] = (parsed_losses['cycle_A'] + parsed_losses['cycle_B'])/2
    del parsed_losses['cycle_A']
    del parsed_losses['cycle_B']


    for key in translate.keys():
        f = files[key]
        f.write(epoch+',' + iteration + ','+  str(parsed_losses[key])+'\n')


for key in translate.keys():
    files[key].close()

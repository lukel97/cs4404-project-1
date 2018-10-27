import sys
import re

lines = [line for line  in open(sys.stdin.readline().strip())]
parens_matcher = r'\((.*?)\)'
translate = { 'D_A' : 'y2x_discriminator',
                'G_A':'y2x_generator',
                'cycle_A':'x2x_cycle_consistency',
                'idt_A':'idtA',
                'D_B':'x2y_discriminator',
                'G_B':'x2y_generator',
                'cycle_B':'y2y_cycle_consistency',
                'idt_B': 'idt_B'}
print_every = 20
for i, line in enumerate(lines):
    if not (i % print_every):
        print(str(i) +' lines processed'.format(i))
    line_match = re.findall(parens_matcher, line).pop()
    matches = [s for s in line_match.split(' ') if s]
    
    parsed_arr = []
    i = 1
    while i < len(matches):
        parsed_arr.append(matches[i].strip()[:-1])
        i += 2
    
    epoch, iteration = parsed_arr[:2]
    
    exclude_parens_match = '(' + line_match+')'
    matches_losses = line[len(exclude_parens_match):].strip().split(' ')
    parsed_losses = {}
    i = 1
    while i < len(matches_losses):
        parsed_losses[matches_losses[i - 1][:-1]] =  (matches_losses[i].strip())
        i += 2

    for key in translate.keys():
        f = open("summer2winter_yosemite_" + translate[key]+'.csv','a+' )
        f.write(epoch+','+iteration+','+parsed_losses[key]+'\n')
        f.close()

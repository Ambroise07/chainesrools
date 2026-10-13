"""
run usecases.py or test.py.

date : 9.10.2026
author: Gnabro Israel

"""
import argparse
import subprocess
import sys
import pathlib

# dir
dir = pathlib.Path(__file__).parent.resolve()

# values of file --- > test | vnet | lessnet
cmd = argparse.ArgumentParser(
    prog='chaines rools', 
    description='Show some demo of chaines rools.', 
    epilog='ambroiseisrael5@gmail.com'
)

demos = ['test', 'vnet', 'lessnet']
cmd.add_argument('--out', default='test', choices=demos, 
                  type=str, help='run  demos of chainesrools')
value = cmd.parse_args()



match value.out.lower():
  case 'test':
    # do test stuff here
    path = dir / 'usecases.py'
    subprocess.run([sys.executable, path])

  case 'vnet':
    # do vnet stuff here
    pass
  case 'lessnet':
    # do lessnet stuf here
    pass     

print('work done.')

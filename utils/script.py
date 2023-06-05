import os
import argparse

parser = argparse.ArgumentParser(description="Train network")
parser.add_argument('--root', type=str, help='root of experiment',
    default='/home/tyb/anaconda/exp07h/080110drive-lunetXlunetdi0.1ds0.5-fr')
parser.add_argument('--tag', type=str, help='tag of experiment', default='a')
args = parser.parse_args()


if os.path.isdir('/home/tan/datasets/FastSS'):
    root = '/home/tan/datasets/FastSS/'
elif os.path.isdir('/home/tyb/datasets/FastSS'):
    root = '/home/tyb/datasets/FastSS/'
else:
    root = r'G:\Objects\expSeg\MetricCAL\FastSS/'

os.system('conda deactivate')

os.system('cd '+root)
print('listdir of:', root)
print(os.listdir(root))


# /home/tyb/anaconda/exp07h/080110drive-lunetXlunetdi0.1ds0.5-fr
# /home/tan/anaconda/exp08a/080902drive-lunetXlunet_eafds0.5-fr

cmd = '''matlab -nodisplay -r "fastss('{}', '{}')"'''.format(args.root, args.tag)
print(cmd)
os.system(cmd)
os.system('exit()')
# os.system('''matlab -nodisplay --nojvm -r "fastss('{}', '{}')" '''.format(args.root, args.tag))
# os.system('''matlab -nosplash -nodesktop -r "fastss('{}', '{}')" '''.format(args.root, args.tag))

# matlab -nodisplay -r "fastss('/home/tyb/anaconda/exp07h/080110drive-lunetXlunetdi0.1ds0.5-fr', 'a')"
# matlab -nodisplay -r "fastss('/home/tan/anaconda/exp08a/080823drive-lunetXlunet_csmds0.5-fr', 'a')"
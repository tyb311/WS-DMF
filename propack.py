import os, time
# print('__package__:', __package__, '__name__:', __name__)
##################################################################

root = 'exp/'#r'expinc\exp'
# os.chdir(r'segtan\codes')

PY_CONTENT = '# -*- encoding:utf-8 -*-\n' 
PY_COUNTER = 0


##################################################################
def pyFile2Txt(listName, files, strStart='#start#', strEnd='#end#', toTxt='seg'):
	content = '###Code for '+listName
	lineCnt = 0    
	
	for file in files:
		# print('Opening', file)
		pyCnt = 0
		flag_write = False
		for line in open(file, 'rb').readlines():
			line = line.decode().replace('\r\n','\n')
			if line.__contains__(strStart):
				# print('start', file)
				flag_write=True
				content+='\n'#'\n{}#{}{}\n'.format(seperator,file,seperator)
				continue
			elif line.__contains__(strEnd):
				# print('end',file)
				flag_write=False
			# print(line, end='')
			
			if flag_write:
				content+=line
				pyCnt+=1

				# if file.__contains__('loss'):
				#     print(line)
		lineCnt+=pyCnt+2
		if pyCnt>0:
			print('\tCopied{:4d} lines @ {}'.format(pyCnt, file))

	# print(content)
	print('{}-->{} has {} lines!!!'.format(listName, toTxt, lineCnt))
	print('*'*64)
	# return content, lineCnt
	global PY_CONTENT, PY_COUNTER
	PY_CONTENT += content+"\n"
	PY_COUNTER += lineCnt

	# with open(os.path.join(root, toTxt+'.txt'), 'w', encoding="utf-8") as f:
	# 	f.write(content)


##################################################################
# def pyProj2Txt(codeType='eye'):
# 	print('*'*64)
# 	print('Code for jupyter from ', codeType)
# 	print('*'*64)

# 	# 实验部分
# 	pyFile2Txt('kite', ['{}/{}'.format(kite.NAME_FOLDER, py) for py in kite.NAME_FILES[codeType]], toTxt='txt_kite')
# 	# pyFile2Txt('kite', ['{}/{}'.format(kite.NAME_FOLDER, py) for py in kite.NAME_FILES[codeType]],
# 	# 	strStart='#EXPS#', strEnd='#EXPE', toTxt='txt_exp')

# 	# 完结撒花
# 	print('Total Lines:', PY_COUNTER, codeType)
# 	timeStr = time.strftime("%d", time.localtime())
# 	with open(root+'{}{}.py'.format(timeStr, codeType, ), 'w', encoding="utf-8") as f:
# 		f.write(PY_CONTENT)
# 	# with open(root+'{}.py'.format(codeType), 'w', encoding="utf-8") as f:
# 	# 	f.write(PY_CONTENT)


FILES = [

	'data/eyeset.py',

	'utils/optim.py',
	'utils/loss.py',

	'nets/modules/activation.py',
	'nets/modules/attention.py',

	'nets/conv.py',
	'nets/lunet.py',
	'nets/rot.py',
	'nets/dmfu.py',

	'build.py',
	'grad.py',
	'loop.py',
	'main.py'
]


##################################################################
if __name__ == '__main__':
	print('packing')
	# pyProj2Txt('eye')


	pyFile2Txt('segcl', files=FILES, strStart='#start#', strEnd='#end#', toTxt='seg')



	# 完结撒花
	print('Total Lines:', PY_COUNTER)
	timeStr = time.strftime("%d", time.localtime())
	with open(root+'{}cl.py'.format(timeStr), 'w', encoding="utf-8") as f:
		f.write(PY_CONTENT)

	# from proclean import pyProjClean
	# pyProjClean()
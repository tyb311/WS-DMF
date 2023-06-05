gpu=$1
top=$2

pyf="prob.py"
net='scls'


echo '***************************************************************************'
echo "画出不同样本构造策略的概率图"
echo "arg=" $#
echo "cmd=" $0 

if [ -n "$1" ]; then
    echo "gpu=" $gpu
else
    echo "请输入gpu编号"
    sleep 3
    exit
fi


if [ -n "$2" ]; then
    echo "top=" $top
else
    echo "请输入top编号"
    sleep 3
    exit
fi
echo '***************************************************************************'


sleep 4h
for((i=0;i<${top};i++));  
do
    echo '***************************************************************************'
    echo "python ${pyf} --gpu=${gpu} --sss=prob --top=${top} --low=${i} --net=${net} --arch=siam --bs=12"
    echo '***************************************************************************'
    # sleep 1

    python ${pyf} --gpu=${gpu} --sss=prob --top=${top} --low=${i} --net=${net} --arch=siam --bs=12
    sleep 5
done

sleep 5
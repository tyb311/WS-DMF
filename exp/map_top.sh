gpu=$1
top=$2

pyf="21cl.py"
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


for i in 0 1 2 3 4;
do
    echo '***************************************************************************'
    echo "python ${pyf} --gpu=${gpu} --sss=part --top=${top} --low=${i} --net=${net} --arch=siam"
    echo '***************************************************************************'
    # sleep 1

    python ${pyf} --gpu=${gpu} --sss=part --top=${top} --low=${i} --net=${net} --arch=siam 
    sleep 5
done

# sleep 5
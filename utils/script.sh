root=$1
tag=$2

####################################################################################
# variable 是变量名，value 是赋给变量的值。
# 如果 value 不包含任何空白符（例如空格、Tab 缩进等），那么可以不使用引号；
# 如果 value 包含了空白符，那么就必须使用引号包围起来。使用单引号和使用双引号也是有区别的
####################################################################################
if [  -n "$1" ]; then
    echo "root="$root
else
    echo "请输入root"
    # sleep 3
    root="/home/tyb/anaconda/exp07h/080110drive-lunetXlunetdi0.1ds0.5-fr"
    echo "root="$root
    # exit
fi
####################################################################################


####################################################################################
if [  -n "$2" ]; then
    echo "tag="$tag
else
    echo "请输入tag"
    # sleep 3
    tag="a"
    echo "tag="$tag
    # exit
fi
####################################################################################



####################################################################################
if [ -d "/home/tan/datasets/FastSS" ]; then
    cd /home/tan/datasets/FastSS
fi

if [ -d "/home/tyb/datasets/FastSS" ]; then
    cd /home/tyb/datasets/FastSS
fi

if [ -d "G:\Objects\expSeg\MetricCAL\FastSS" ]; then
    cd G:\Objects\expSeg\MetricCAL\FastSS
fi
####################################################################################



####################################################################################
# /home/tyb/anaconda/exp07h/080110drive-lunetXlunetdi0.1ds0.5-fr
# /home/tan/anaconda/exp08a/080902drive-lunetXlunet_eafds0.5-fr
matlab -nodisplay -r "fastss('${root}', '${tag}'), exit"
# matlab -nodisplay -r "exit"
####################################################################################

# 调用方法
# cmd = f'bash script.sh {os.getcwd()}/{self.root} {key}'
# print(cmd)
# os.system(cmd)
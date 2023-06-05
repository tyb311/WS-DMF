
echo '***************************************************************************'
echo "python thsh"
echo '***************************************************************************'
# sleep 1



for((i=1;i<9;i++));  
do
    echo '***************************************************************************'
    echo "python thsh.py --gpu=1 --sss=thsh --dis=${i}"
    echo '***************************************************************************'
    # sleep 1

    python thsh.py --gpu=1 --sss=thsh --dis=${i} 
    sleep 5
done


# python thsh.py --gpu=1 --sss=thsh --loss_cl=sim2 --roma=True
# sleep 5
# python thsh.py --gpu=1 --sss=thsh --loss_cl=sim2 --roma=True
# sleep 5
# python thsh.py --gpu=1 --sss=thsh --loss_cl=sim2 --roma=False
# sleep 5
# python thsh.py --gpu=1 --sss=thsh --loss_cl=sim2 --roma=False

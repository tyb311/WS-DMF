
echo '***************************************************************************'
echo "python munet sunet lunet"
echo '***************************************************************************'


# for((i=1;i<=5;i+=2));  
for((i=7;i<=9;i+=2));  
do
    # ds=`expr ${i} \* 0.1` 
    # ds=$(echo " ${i} \* 0.1" | bc -l)
    echo "python srp.py --gpu=1 --net=lunet --seg=lunet  --coff_ct=${i} --ct=True"
    python 14cl.py --gpu=1 --net=lunet --seg=lunet  --coff_ct=${i}  --ct=True
    sleep 5 
done


# for((i=1;i<=5;i+=2));  
# do
#     # ds=`expr ${i} \* 0.1` 
#     # ds=$(echo " ${i} \* 0.1" | bc -l)
#     echo "python srp.py --gpu=2 --net=lunet --seg=lunet  --coff_ct=0.${i}"
#     python srp.py --gpu=2 --net=lunet --seg=lunet  --coff_ct=0.${i} 
#     sleep 5 
# done


sleep 5 

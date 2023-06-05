


for smp in 'half' 'hard' 
do
    echo "python valt.py --gpu=0 --sss=$smp  --arch=siam --loss_cl=sim3  --coff_cl=0.1"
    # python valt.py --gpu=0 --sss=$smp  --arch=siam --loss_cl=sim3  --coff_cl=0.1 --inc=ValT
    # sleep 5 
    python valt.py --gpu=1 --sss=$smp  --arch=siam --loss_cl=sim2  --coff_cl=0.1 --inc=ValT
    sleep 5 
done
sleep 5 

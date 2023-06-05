
echo '***************************************************************************'
echo "python hard"
echo '***************************************************************************'
# sleep 1

#  --bug=True --board=False

python 04cl.py --gpu=1 --net=smf --seg=lunet --db=stare --inc=RE --los=fr
python 04cl.py --gpu=1 --net=smf --seg=lunet --db=chase --inc=RE --los=fr
python 04cl.py --gpu=1 --net=smf --seg=lunet --db=hrf   --inc=RE --los=fr
python 04cl.py --gpu=1 --net=smf --seg=lunet --db=drive --inc=RE --los=fr

# python siam.py --gpu=1 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=0.1
# sleep 5 
# python siam.py --gpu=1 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=0.3
# sleep 5 
# python siam.py --gpu=1 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=0.5
# sleep 5 
# python siam.py --gpu=1 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=0.7
# sleep 5 
# python siam.py --gpu=1 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=0.9
# sleep 5 


# python siam.py --gpu=0 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=1
# sleep 5 
# python siam.py --gpu=0 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=3
# sleep 5 
# python siam.py --gpu=0 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=4
# sleep 5 
# python siam.py --gpu=0 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=5
# sleep 5 
# python siam.py --gpu=0 --sss=hard  --arch=siam --loss_cl=sim3  --coff_cl=9
# sleep 5 
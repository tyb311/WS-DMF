
echo '***************************************************************************'
echo "python shpere"
echo '***************************************************************************'

python 31cl.py --gpu=2 --loss_cl= --net=spu --seg= --arch= --sss=hard --coff_cl=0.1 --inc=Sphere
sleep 5
python 31cl.py --gpu=2 --loss_cl= --net=spu --seg= --arch= --sss=hard --coff_cl=0.5 --inc=Sphere
sleep 5
python 31cl.py --gpu=2 --loss_cl= --net=spu --seg= --arch= --sss=hard --coff_cl=0.07 --inc=Sphere
sleep 5
python 31cl.py --gpu=2 --loss_cl= --net=spu --seg= --arch= --sss=hard --coff_cl=0.9 --inc=Sphere
sleep 5
python 31cl.py --gpu=2 --loss_cl= --net=spu --seg= --arch= --sss=hard --coff_cl=0.05 --inc=Sphere
sleep 5


# python siam.py --gpu=0 --loss_cl=sim3 --net=sunet --seg=sunet --arch=siam --sss=hard --coff_cl=0.1
# sleep 5 
# python siam.py --gpu=1 --loss_cl=sim3 --net=sunet --seg=sunet --arch=siam --sss=hard --coff_cl=1
# sleep 5 

# python siam.py --gpu=0 --loss_cl=nce3 --net=sunet --seg=sunet --arch=siam --sss=hard --coff_cl=0.1
# sleep 5 
# python siam.py --gpu=0 --loss_cl=nce3 --net=sunet --seg=sunet --arch=siam --sss=hard --coff_cl=1
# sleep 5 

# python 31cl.py --gpu=0 --loss_cl= --net=spu --seg= --arch= --sss=hard --coff_cl=0.1 --inc=Sphere --bs=0 --ds=64


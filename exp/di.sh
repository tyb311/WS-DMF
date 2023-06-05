
echo '***************************************************************************'
echo "python di"
echo '***************************************************************************'
# sleep 30m

# python di.py --gpu=2 --coff_ct=0.1 --loss_ct=di --inc=D2N2 --net=scdu --seg=scdu
# sleep 5 
# python di.py --gpu=2 --coff_ct=0.1 --loss_ct=di --inc=D2N2 --net=sdou --seg=sdou
# sleep 5 
# python di.py --gpu=2 --coff_ct=0.1 --loss_ct=di --inc=D2N2 --net=sbau --seg=sbau
# sleep 5 
# python di.py --gpu=2 --coff_ct=0.1 --loss_ct=di --inc=D2N2 --net=spyu --seg=spyu
# sleep 5 



python 25cl.py --gpu=0 --coff_ds=0.5 --coff_ct=0.9 --loss_ct=di --net=sunet --seg=sunet --sss=half
sleep 5 
python 25cl.py --gpu=0 --coff_ds=0.5 --coff_ct=0.9 --loss_ct=di --net=sunet --seg=sunet --sss=hard
sleep 5 
python 25cl.py --gpu=0 --coff_ds=0.5 --coff_ct=0.5 --loss_ct=di --net=sunet --seg=sunet --sss=half
sleep 5 
python 25cl.py --gpu=0 --coff_ds=0.5 --coff_ct=0.5 --loss_ct=di --net=sunet --seg=sunet --sss=hard
sleep 5 


# python di.py --gpu=0 --coff_ds=0.7 --coff_ct=0.5 --loss_ct=di
# sleep 5 
# python di.py --gpu=0 --coff_ds=0.7 --coff_ct=0.7 --loss_ct=di
# sleep 5 
# python di.py --gpu=0 --coff_ds=0.7 --coff_ct=0.9 --loss_ct=di
# sleep 5 


# python di.py --gpu=0 --coff_ds=0.9 --coff_ct=0.5 --loss_ct=di
# sleep 5 
# python di.py --gpu=0 --coff_ds=0.9 --coff_ct=0.7 --loss_ct=di
# sleep 5 
# python di.py --gpu=0 --coff_ds=0.9 --coff_ct=0.9 --loss_ct=di
# sleep 5 
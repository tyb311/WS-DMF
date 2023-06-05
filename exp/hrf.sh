
echo '***************************************************************************'
echo "python CS-BME"
echo '***************************************************************************'
# sleep 1


# python hrf.py --gpu=0 --net=bunet --seg=bunet --csm=False --eaf=False
# sleep 5 
# python hrf.py --gpu=0 --net=lunet --seg=lunet --csm=False --eaf=False
# sleep 5 
# python hrf.py --gpu=0 --net=bunet --seg=bunet --csm=True --eaf=False
# sleep 5 
# python hrf.py --gpu=0 --net=bunet --seg=bunet --csm=False --eaf=True
# sleep 5 

# python 19cl.py --gpu=0 --net=lunet --seg=lunet --db=rcslo --sss=half
# python 19cl.py --gpu=0 --net=lunet --seg=lunet --db=rcslo --sss=hard

# python 19cl.py --gpu=0 --net=lunet --seg=lunet --db=rcslo --sss=half
# python 19cl.py --gpu=0 --net=lunet --seg=lunet --db=rcslo --sss=hard


# python 20cl.py --gpu=1 --net=lunet --seg=lunet --db=iostar --sss=half
# python 20cl.py --gpu=1 --net=lunet --seg=lunet --db=iostar --sss=hard

python tmi.py --gpu=0 --net=lunet --seg=lunet --db=uoadr --sss=half
python tmi.py --gpu=0 --net=lunet --seg=lunet --db=uoadr --sss=hard
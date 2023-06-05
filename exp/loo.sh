
echo '***************************************************************************'
echo "python 30cl  --inc=TAN"
echo '***************************************************************************'
# sleep 1h
 
# python 30cl.py --gpu=0 --coff_ct=0.9 --sss=half --coff_cl=0.1 --loss_ct=di  --net=lunet --seg=lunet --db=chase
# sleep 5  
# python 30cl.py --gpu=0 --coff_ct=0.9 --sss=half --coff_cl=0.1 --loss_ct=di  --net=lunet --seg=lunet --db=hrf
# sleep 5
# python 30cl.py --gpu=0 --coff_ct=0.9 --sss=half --coff_cl=0.1 --loss_ct=di  --net=lunet --seg=lunet --db=drive
# sleep 5   


# for i in 19 18 17 16 15 14 13 12 11 10
# for i in 0 1 2 3 4 5 6 7 8 9;
for i in 0 1 2 3 4 5 9 15 17 18 19
do
    echo "python 30cl.py --gpu=0 --coff_ct=0.9 --sss=half --coff_cl=0.1 --loss_ct=di  --net=lunet --seg=lunet --db=stare --loo=${i} --inc=TAN"
    python 30cl.py --gpu=0 --coff_ct=0.9 --sss=half --coff_cl=0.1 --loss_ct=di  --net=lunet --seg=lunet --db=stare --loo=${i} --inc=TAN
    sleep 5 
done


# 重点训练0,1,2,3,4,5,9,15,17，18,19
#0  F=77.58,C=99.77,A=88.84,L=87.53,rSe=82.29,rSp=98.91,rAcc=96.07,SS=82.29,Rnc=14.08
#1  F=72.05,C=99.89,A=86.42,L=83.46,rSe=72.98,rSp=99.34,rAcc=95.35,SS=72.98,Rnc=5.05
#2  F=76.01,C=99.82,A=90.56,L=84.08,rSe=83.41,rSp=98.69,rAcc=96.63,SS=83.41,Rnc=15.42
#3  F=74.63,C=99.87,A=86.85,L=86.05,rSe=80.18,rSp=98.75,rAcc=96.27,SS=80.18,Rnc=14.26
#4  F=77.60,C=99.83,A=89.59,L=86.76,rSe=85.91,rSp=98.78,rAcc=96.63,SS=85.91,Rnc=15.71
#5  F=82.64,C=99.85,A=92.41,L=89.56,rSe=93.92,rSp=98.89,rAcc=98.21,SS=93.92,Rnc=15.23
#6  F=88.87,C=99.91,A=95.96,L=92.70,rSe=94.18,rSp=99.09,rAcc=98.16,SS=94.18,Rnc=10.49
#7  F=86.92,C=99.83,A=95.11,L=91.54,rSe=90.93,rSp=98.98,rAcc=97.57,SS=90.93,Rnc=10.46
#8  F=91.02,C=99.85,A=97.14,L=93.84,rSe=89.50,rSp=99.46,rAcc=97.52,SS=89.50,Rnc=4.78
#9  F=79.84,C=99.88,A=90.92,L=87.91,rSe=86.60,rSp=98.35,rAcc=96.17,SS=86.60,Rnc=12.66
#10 F=91.29,C=99.83,A=97.03,L=94.24,rSe=88.52,rSp=99.87,rAcc=97.33,SS=88.52,Rnc=1.34
#11 F=91.79,C=99.91,A=97.17,L=94.55,rSe=93.70,rSp=99.18,rAcc=98.20,SS=93.70,Rnc=7.70
#12 F=90.12,C=99.90,A=96.62,L=93.36,rSe=88.06,rSp=99.39,rAcc=97.23,SS=88.06,Rnc=5.20
#13 F=87.52,C=99.89,A=95.35,L=91.89,rSe=88.19,rSp=99.18,rAcc=97.11,SS=88.19,Rnc=7.03
#14 F=86.01,C=99.90,A=94.37,L=91.23,rSe=84.14,rSp=99.21,rAcc=96.26,SS=84.14,Rnc=3.31
#15 F=82.25,C=99.95,A=90.64,L=90.79,rSe=88.51,rSp=98.44,rAcc=96.61,SS=88.51,Rnc=11.31
#16 F=84.33,C=99.81,A=93.94,L=89.93,rSe=79.81,rSp=99.60,rAcc=95.19,SS=79.81,Rnc=2.72
#17 F=72.32,C=99.83,A=86.20,L=84.04,rSe=76.58,rSp=99.27,rAcc=96.76,SS=76.58,Rnc=9.59
#18 F=80.36,C=99.76,A=89.79,L=89.71,rSe=89.87,rSp=99.22,rAcc=98.28,SS=89.87,Rnc=12.32
#19 F=81.37,C=99.89,A=90.78,L=89.73,rSe=84.77,rSp=98.32,rAcc=96.64,SS=84.77,Rnc=13.28
# mean score:
# 82.73,99.86,92.28,89.65,86.10,99.05,96.91,86.10,9.60,99.58
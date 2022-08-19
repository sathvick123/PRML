import math

def radius(wl,d1,d2):
    return math.sqrt((wl*d1*d2)/(d1+d2))

def height_calculator(percentage,wl,d1,d2):
    return percentage*(radius(wl,d1,d2))

def get_height(dist,percent,ibd,wl,ht):
    maxh=ht
    for p in dist:
        maxh = max(maxh, p[1]+ height_calculator(percent,wl,p[0],ibd-p[0]) )
    return maxh

def get_LOS(dist,ibd,hA,hB,ht):
    ta=0
    tb=0
    h=get_height(dist,0.4,ibd,wl,ht)
    ta=max(ta,h-hA)
    tb=max(tb,h-hB)
    return [ta,tb]

def get_nearLOS(dist,ibd,hA,hB,ht):
    ta = 0
    tb = 0
    h = get_height(dist, 0.6, ibd,wl,ht)
    ta = max(ta, h - hA)
    tb = max(tb, h - hB)
    return [ta, tb]

input=open("inp.txt","r")

f=input.readlines()
hA=float(f[0])
hB=float(f[1])
ibd=float(f[2])
freq=float(f[3])
nob=int(f[4])
dist=[]
wl=(3/(freq*10))

h=max(hA,hB)

for i in range(0,nob):
    l=f[i+5]
    lis=l.split()
    fl = [float(x) for x in lis]
    dist.append(fl)


att=92.5+20*math.log((freq*ibd)/1000)
P_in=10**(-5)
P_out=P_in*math.exp(att/10)

los=get_LOS(dist,ibd,hA,hB,h)
print("Solution type: Line of Sight")
print("Feasibility: ",end="")

if los[0]>20:
    print("Not Feasible")
else:
    print("Feasible")
print("Frequency used: ",freq)
print("Cell Tower A height: ",round(los[0],2))
print("Cell Tower B height: ",round(los[1],2))
Gaps =[(h-d[1]-(0.4*radius(wl,d[0],ibd-d[0]))) for d in dist]
print(Gaps)
print("Transmit power at Cell Tower A, if the power received at -50dbm",end=" ")
print(round(P_out,3))

print("\n")

nlos=get_nearLOS(dist,ibd,hA,hB,h)
print("Solution type: Near Line of Sight")
print("Feasibility: ",end="")
if nlos[0]>20:
    print("Not Feasible")
else:
    print("Feasible")
print("Frequency used: ",freq)
print("Cell Tower A height: ",round(nlos[0],2))
print("Cell Tower B height: ",round(nlos[1],2))
Gaps =[(h-d[1]-(0.6*radius(wl,d[0],ibd-d[0]))) for d in dist]
print(Gaps)
print("Transmit power at Cell Tower A, if the power received at -50dbm:",end=" ")
print(round(P_out,3))
#92.5+20ln(f*d)
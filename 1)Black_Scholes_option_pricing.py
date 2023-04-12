import numpy as np
from scipy.stats import norm

#defining variable
r=float(input('Please enter Interest rate(decimal):'))
s=float(input('Please enter Underlying price:'))
k=float(input('Please enter Strike price:'))
time=int(input('Please enter time remaining days:'))
t=float(time/365)
sigma=float(input('Please enter volatility (decimal):'))
price=None

def blackscholes(r,s,k,t,sigma,price,type=""):
#calculate B-S option price for a call/put

    d1=(np.log(s/k)+((r+sigma**2)/2)*t)/(sigma*np.sqrt(t))
    d2=d1-sigma*np.sqrt(t)

    try: 
        type=input(f'Please Enter c for call and p for put:')
        if type=="c":
            price=s*norm.cdf(d1,0,1)-k*np.exp(-r*t)*norm.cdf(d2,0,1)
        elif type=="p":
            price=k*np.exp(-r*t)*norm.cdf(-d2,0,1)-s*(norm.cdf(-d1,0,1))
        return price
    except:
        print('please confirm all the parameters!')

print('Option price is:',round(blackscholes(r,s,k,t,sigma,price,type=""),2))

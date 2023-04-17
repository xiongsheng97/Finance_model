import yfinance as yf
import numpy as np
import datetime as dt
import pandas as pd
from pandas_datareader import data as pdr
import scipy as sc
import plotly.graph_objects as go
yf.pdr_override()

stock=[]

a=int(input('Number of stocks:'))
for i in range (a):
   stocklist=input("Please enter stock:").upper()
   stock.append(stocklist)
end=dt.datetime.now()
start=dt.datetime(2015,12,15) #start=end-dt.timedelta(days=500) if want by days

def getdata(stock,start,end):
   df= pdr.get_data_yahoo(stock,start,end)
   df=df['Close']

   returns=df.pct_change()
   meanreturn=returns.mean()
   covmatrix=returns.cov()
   return meanreturn,covmatrix

meanreturn,covmatrix=getdata(stock,start,end)
#weights=np.array([0.5,0.3,0.2])
def portfolioPerformance(weights,meanreturn,covmatrix):
    returns=np.sum(meanreturn*weights)*252
    #np.dot= matrix multiply, .T= transpose , Trading days=252
    std= np.sqrt(np.dot(weights.T,np.dot(covmatrix,weights)))*np.sqrt(252)
    return returns, std


def negativeSR(weights,meanreturn,covmatrix,riskfreerate=0):
    preturns,pstd= portfolioPerformance(weights,meanreturn,covmatrix)
    return -(preturns-riskfreerate)/pstd
def maxSr(meanreturn,covmatrix,riskfreerate=0,constraintset=(0,1)):
    #Focusing on trade off between volatility and return
    #minimise the negative SR by altering the weight of portfolio)
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    numstock=len(meanreturn)
    args=(meanreturn,covmatrix,riskfreerate)
    constraints=({'type':'eq','fun':lambda x:np.sum(x)-1})
    bound=constraintset
    bounds=tuple(bound for asset in range(numstock))
    results=sc.optimize.minimize(negativeSR,numstock*[1./numstock],args=args,
                                     method='SLSQP',bounds=bounds,constraints=constraints)
    return results
#Minimium Portfolio Variance
def portfolioVariance(weights,meanreturn,covmatrix):
    return portfolioPerformance(weights,meanreturn,covmatrix)[1]
def minimizeVariance(meanreturn,covmatrix,constraintset=(0,1)):
    #what is the absolute mini volatility we have by having the portfolio
    #minimize portfolio variance by altering the weights/allocation of assets
    numstock=len(meanreturn)
    args=(meanreturn,covmatrix)
    constraints=({'type':'eq','fun':lambda x:np.sum(x)-1})
    bound=constraintset
    bounds=tuple(bound for asset in range(numstock))
    results=sc.optimize.minimize(portfolioVariance,numstock*[1./numstock],args=args,
                                     method='SLSQP',bounds=bounds,constraints=constraints)
    return results

#returns,std= portfolioPerformance(weights,meanreturn,covmatrix)
#returns,std= round(returns*100,2),round(std*100,2)
#print(returns,std)
#result=maxSr(meanreturn,covmatrix)
#sr,maxweight=result['fun'],result['x']
#print(sr,maxweight)

#minvarresult=minimizeVariance(meanreturn,covmatrix)
#minvar,minvarweight=minvarresult['fun'],minvarresult['x']
#print(minvar,minvarweight)
def portfolioReturn(weights,meanreturn,covmatrix):
    return portfolioPerformance(weights,meanreturn,covmatrix)[0]

def efficientOpt(meanreturn, covmatrix,returnTarget,constraintSet=(0,1)):
    # For each returnTarget, we want to optimise portfolio for min variance
    numstock=len(meanreturn)
    args=(meanreturn,covmatrix)
    constraints=({'type':'eq','fun':lambda x:portfolioReturn(x,meanreturn, covmatrix)-returnTarget},
                 {'type':'eq','fun':lambda x:np.sum(x)-1})
    bound=constraintSet
    bounds=tuple(bound for stock in range(numstock))
    effOpt=sc.optimize.minimize(portfolioVariance,numstock*[1./numstock],args=args,
                        method='SLSQP',bounds=bounds,constraints=constraints)
    return effOpt 


def calculatedResults(meanreturn, covmatrix, riskFreeRate=0, constraintSet=(0,1)):
    #Read in mean, cov matrix, and other financial information. Output:Max SR , Min Volatility, efficient frontier
    #Max sharpe Ratio portfolio
    maxSR_Portfolio= maxSr(meanreturn,covmatrix)
    maxSR_returns,maxSR_std= portfolioPerformance(maxSR_Portfolio['x'],meanreturn,covmatrix)
    maxSR_allocation=pd.DataFrame(maxSR_Portfolio['x'],index=meanreturn.index,columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
    #Min volatility portfolio
    minvol_Portfolio= minimizeVariance(meanreturn,covmatrix)
    minvol_returns,minvol_std= portfolioPerformance(minvol_Portfolio['x'],meanreturn,covmatrix)
    minvol_allocation=pd.DataFrame(minvol_Portfolio['x'],index=meanreturn.index,columns=['allocation'])
    minvol_allocation.allocation = [round(i*100,0) for i in minvol_allocation.allocation]

    #Efficient Frontier
    efficientList=[]
    targetReturn=np.linspace(minvol_returns,maxSR_returns,20)
    maxSR_returns,maxSR_std=round(maxSR_returns*100,2),round(maxSR_std*100,2)
    minvol_returns,minvol_std=round(minvol_returns*100,2),round(minvol_std*100,2)
    for target in targetReturn:
        efficientList.append(efficientOpt(meanreturn, covmatrix,target)['fun'])

    return maxSR_returns,maxSR_std,maxSR_allocation,minvol_returns,minvol_std,minvol_allocation,efficientList,targetReturn

print (calculatedResults(meanreturn, covmatrix))
print (efficientOpt(meanreturn, covmatrix,1))

def ef_graph(meanreturn, covmatrix, riskFreeRate=0, constraintSet=(0,1)):
    #return graph ploting the minvol , maxsr and efficient frontier
    maxSR_returns,maxSR_std,maxSR_allocation,minvol_returns,minvol_std,minvol_allocation,efficientList,targetReturn=calculatedResults(meanreturn, covmatrix, riskFreeRate=0, constraintSet=(0,1))

    #Max SR
    maxSharpeRatio=go.Scatter(
        name='Maximum Sharpe Ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_returns],
        marker=dict(color='red',size=14,line=dict(width=3,color='black'))
    )
    #min_vol
    minVol=go.Scatter(
        name='Minimum volatility',
        mode='markers',
        x=[minvol_std],
        y=[minvol_returns],
        marker=dict(color='red',size=14,line=dict(width=3,color='black'))
    )    
    #Efficient Frontier
    ef_Curve=go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100,2) for ef_std in efficientList],
        y=[round(target*100,2) for target in targetReturn],
        line=dict(color='black',width=3,dash='dashdot')
    )
    
    data=[maxSharpeRatio,minVol,ef_Curve]

    layout=go.Layout(
        title='Portfolio optimisation with the Efficient Frontier',
        yaxis=dict(title='Annualised return(%)'),
        xaxis=dict(title='Annualised Volatility(%)'),
        showlegend=True,
        legend=dict(
            x=0.75,y=0,traceorder='normal',
            bgcolor='#E2E2E2',bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)
    
    fig=go.Figure(data=data,layout=layout)
    return fig.write_html("EF.html", auto_open=True)

ef_graph(meanreturn, covmatrix)
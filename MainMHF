%==================================================================
%Morteza Hajitabar Firuzjaei
%Senior Computer Programmer
%==================================================================
load('input.mat');

X = tonndata(inputData(:,(1:3)),false,false); %parameters(Temerature, Relative Humidity, Pressure)
T = tonndata(inputData(:,4),false,false); %Target(Past Wind Speed)
%=================================================================
N = 100; % Multi-step ahead prediction
%=================================================================
inputSeries  = X(1:end);
targetSeries = T(1:end);
inputSeriesVal  = X(end-N+1:end);
targetSeriesVal = T(end-N+1:end); 
%=================================================================
delay = 1; %one hour
neuronsHiddenLayer = 10;
%=================================================================
% Network Creation
net = narxnet(1:delay,1:delay,neuronsHiddenLayer);
[Xs,Xi,Ai,Ts] = preparets(net,inputSeries,{},targetSeries); 
net = train(net,Xs,Ts,Xi,Ai);
view(net)
%=================================================================
%Multi-step Ahead
Y = net(Xs,Xi,Ai); 
perf = perform(net,Ts,Y);
[Xs1,Xio,Aio] = preparets(net,inputSeries(1:end-delay),{},targetSeries(1:end-delay));
[Y1,Xfo,Afo] = net(Xs1,Xio,Aio);
%=================================================================
%Close-loop
[netc,Xic,Aic] = closeloop(net,Xfo,Afo);
[yPred,Xfc,Afc] = netc(inputSeriesVal,Xic,Aic);
multiStepPerformance = perform(net,yPred,targetSeriesVal);
view(netc)
%=================================================================
%Plot
figure;
plot([cell2mat(targetSeries),nan(1,N);
      nan(1,length(targetSeries)),cell2mat(yPred);]')
legend('Original Inputs','Network Predictions')
%==================================================================

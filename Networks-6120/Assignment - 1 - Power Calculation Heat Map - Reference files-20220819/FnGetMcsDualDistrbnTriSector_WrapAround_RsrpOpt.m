function [nMCS, dMCS, UeMCS, UeAssocBs] = FnGetMcsDualDistrbnTriSector_WrapAround_RsrpOpt(visual,OprId,BS_XT,BS_YT,xpos,ypos,xcount,ycount, ...
                                                                                          BSTxPwr,BsBw,BsCf,BsHt,UeHt)
%disType, OprId, CBS_X, CBS_Y, xRange, yRange, incRange, IBD, visual, PLType, NUM_BS, BSTxPwr,BsBw,BsCf,BsHt,UeHt)
%This function determines 37 BS positions for two operators, distributed in
%a hexa layout. Te 37 BSs are located such that the BS are placed in five
%circular rings around the BS. 37 = 1 + 6 + 12 + 18. At each BS position,
%it is assumed the BS supports a tri sector cell, i.e. each cell is an
%individual BS.
%
% Input Parameters:
% disType: The BS placment, 0 means uniform, 1 means random placement
%
% OprId: A Base number to identify the operator, for easy porcessing later.
% Currently this variable will have values 1,2,3 or 4. (For max of four
% operator scenarios. The number will be multiplied with 1000, and the BS
% will be added, to get the BS identifier - equivalent to cell id)
%
% CBS_X: The X co-ordinate of the Center BS - BS1
% CBS_Y: The Y co-ordinate of the Center BS - BS2
% xRange:
% yRange:
% incRange:
% IBD: Inter base station distance in meters
% visual: = 1, draw the output heatmap and bar graph - used only when one of
% checks is done.
% PLType: Pathloss type: 1 - Macro LOS, 2 - Macro NLOS, 3 - Micro NLOS
% NUM_BS: Number of closest BS to be taken into consideration for SINR
% calculation.
% BSTxPwr: The transmission power of the BS in dbm. We are assuming in this
% function the network is homogenous, and the TX power of all BS is same.
%     .1 watt - 20 dbm
%      1 watt - 30 dbm
%     10 watt - 40 dbm
%     20 watt - 43.0103 dbm
%     30 watt - 44.771213 dbm
%     40 watt - 46.0206 dbm
%    100 watt - 50 dbm

%%
% Revision history.
% Author: Manikantan Srinivasan, CS&E Dept, IIT-Madras, Chennai, India.
%
% Made into function: Jan 5 2016.
%
% function modified April 15 2016,
% BsMT : Mechanical Tilt of the BS Antenna - DEPRECATED. Currently SINR
% measured only for horizontal position variation. Vertical variation not
% considered. may be in future.
% BsET : Electrical Tilt of the BS Antenna - DEPRECATED. Currently SINR
% measured only for horizontal position variation. Vertical variation not
% considered. may be in future.

%%
% NOTE: The following is a constant and hence declared in the start.
% Compute noise power per Resource Block
NOISE_FIGURE = 2.5;
RB_WIDTH = 180*1000;       % RB: 180kHz bandwidth
n0 = -174;                 % dBm - boltzman constant multiplied with tempature in kelvin for 1 Hz.
NOISE_WATT_RB = db2pow(NOISE_FIGURE + n0 + 10*log10(RB_WIDTH) - 30);

%%

Ncell = 30;%91;%37; %Number of cells
Nsec = 3; % Number of sectors per Cell.
BSC  = Ncell * Nsec; % Number of Base Stations.

NBS = 2;
NBSC = 4;%NBS*Nsec;

MBAn = [90,210,330]; % Direction of the main beam in each of the sector.

% Deriving the BS ids (cell ids)
BS_ID = (1:1:BSC)+(OprId*1000);

%%
BsPower = ones(1,BSC)*BSTxPwr; % 43 db - 10 watt
Bwdth   = ones(1,BSC)*BsBw; % 20 MHz
Cf      = ones(1,BSC)*BsCf; % 2 GHz
Ht      = ones(1,BSC)*BsHt; % 25 mt
UHt     = ones(1,BSC)*UeHt; % 25 mt
MbAngle = repmat(MBAn,1,Ncell);


PLT  = ones(1,BSC)*1;%PLType; % Path Loss Type - Urban MAcro LOS

%%
% We are looking for SINR heat map made by 7 BS.
% Distributing UE uniformly, separated by a distance of incRange meters.
% The max incRange is 10 meters.

UE_POSXY = zeros(2,ycount,xcount);
UE_POSXY(1,:,:) = repmat(xpos,ycount,1);
UE_POSXY(2,:,:) = repmat(ypos',1,xcount);

%UE_BS_DS = zeros(BSC,ycount,xcount);
UE_BS_DS_S = zeros(BSC,ycount,xcount);
UE_BS_DS_I = zeros(BSC,ycount,xcount);

%UE_BS_Plos = zeros(BSC,ycount,xcount);
%UE_BS_PLP_RVal = zeros(BSC,ycount,xcount);
UE_BS_DS_XDIFF = zeros(BSC,ycount,xcount);
UE_BS_DS_YDIFF = zeros(BSC,ycount,xcount);

% For every UE, the distance with each of BSC BS is calculated, the distance
% is sorted and the index of BS based on nearest distance is tracked. We
% are interested to work with the nearest 7 BS.

parfor i=1:ycount
    for j=1:xcount
        x = zeros(BSC,2,2);
        x(:,1,1) = UE_POSXY(1,i,j);
        x(:,2,1) = UE_POSXY(2,i,j);
        x(:,1,2) = BS_XT((i-1)*ycount+j,:);
        x(:,2,2) = BS_YT((i-1)*ycount+j,:);
        
        disv=sqrt((x(:,1,1)-x(:,1,2)).^2+(x(:,2,1)-x(:,2,2)).^2)';
        B = disv;
        
        UE_BS_DS_S(:,i,j) = B;
        
        UE_BS_DS_XDIFF(:,i,j) = x(:,1,1) - x(:,1,2);
        UE_BS_DS_YDIFF(:,i,j) = x(:,2,1) - x(:,2,2);
    end
end

%%
UE_BS_RSRP = zeros(BSC,ycount,xcount);
UE_BS_SINR = zeros(NBSC,ycount,xcount);
UE_BS_MCI  = zeros(NBSC,ycount,xcount);
UE_BS_BRate  = zeros(NBSC,ycount,xcount);

parfor i=1:ycount
    for j=1:xcount
        % Determing the PLOS to be applied probabilistically.
        UE_BS_RSRP(:,i,j) = arrayfun(@FnRxPoweForTriSec, BsPower,Bwdth,Cf,Ht,MbAngle,UHt, UE_BS_DS_S(:,i,j)', UE_BS_DS_XDIFF(:,i,j)',UE_BS_DS_YDIFF(:,i,j)', PLT);
        
        FRSRP = UE_BS_RSRP(:,i,j);  % RSRP - chaned to FRSRP - to indicate FULL RSRP.
        
        [B,I] = sort(FRSRP,'descend');
        RSRP = B(1:NBSC)';
        
        sumRsrp = ones(1,NBSC)*sum(RSRP);
        exRsrp  = (sumRsrp - RSRP) + NOISE_WATT_RB;
        sinr = arrayfun(@pow2db, (RSRP ./ exRsrp));
        
        UE_BS_SINR(:,i,j) = sinr;
        
        UE_BS_DS_I(:,i,j) = BS_ID(I);
        [UE_BS_MCI(:,i,j), UE_BS_BRate(:,i,j)] = arrayfun(@FnComputeBitrate, sinr);
    end
end

% To track the MCS value at each of the UE positions, currently uniformly
% placed.
UeMCS = zeros(ycount,xcount);
UeAssocBs = zeros(ycount,xcount);
for i=1:ycount
    UeMCS(i,:) = UE_BS_MCI(1,i,:);
    UeAssocBs(i,:) = UE_BS_DS_I(1,i,:);
end

%Determining the number of positions that have identical MCS, and the
%number of ditinct MCS values between 1 and 15.
values = unique(UeMCS);
nMCS =size(values,1);
insval = histc(UeMCS,values);
insvalc = zeros(nMCS,1);
for i =1:nMCS
    insvalc(i,1) = sum(insval(i,:));
end

% The MCS distribution as percentage.
ip = (insvalc / sum(insvalc)) *100;
dMCS = ip';

%%
% To see how the MCS distribution as a heat map
if visual == 1
    FnDrawMcsHeatMap(UeMCS);
    
end
end
